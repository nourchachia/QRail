/**
 * Timetable Engine - Train position calculation and conflict detection
 * 
 * Reads timetable.json and calculates real-time train positions
 * based on the current day type (weekday/weekend/holiday)
 */

const TimetableEngine = {
    timetable: [],
    segments: [],
    stations: [],
    dayType: 'weekday',
    trainColors: new Map(),

    /**
     * Initialize the engine with data
     */
    init(timetable, segments, stations) {
        this.timetable = timetable;
        this.segments = segments;
        this.stations = stations;
        this.dayType = this.getDayType(new Date());
        this.generateTrainColors();
        console.log(`Timetable engine initialized: ${this.dayType} schedule, ${timetable.length} trains`);
    },

    /**
     * Determine if today is weekday, weekend, or holiday
     */
    getDayType(date) {
        const day = date.getDay();
        // Sunday = 0, Saturday = 6
        if (day === 0 || day === 6) {
            return 'weekend';
        }
        // TODO: Add holiday detection based on calendar
        return 'weekday';
    },

    /**
     * Generate unique colors for each train using HSL
     */
    generateTrainColors() {
        this.trainColors.clear();
        const trains = this.timetable;

        trains.forEach((train, index) => {
            const hue = (index / trains.length) * 360;
            // Vary saturation and lightness by service type
            let saturation = 70;
            let lightness = 50;

            if (train.service_type === 'express') {
                saturation = 85;
                lightness = 45;
            } else if (train.service_type === 'local') {
                saturation = 60;
                lightness = 55;
            }

            this.trainColors.set(train.train_id, `hsl(${hue}, ${saturation}%, ${lightness}%)`);
        });
    },

    /**
     * Get color for a specific train
     */
    getTrainColor(trainId) {
        return this.trainColors.get(trainId) || '#3b82f6';
    },

    /**
     * Parse time string "HH:MM" to minutes since midnight
     */
    parseTime(timeStr) {
        const [hours, minutes] = timeStr.split(':').map(Number);
        return hours * 60 + minutes;
    },

    /**
     * Get current time as minutes since midnight
     */
    getCurrentMinutes() {
        const now = new Date();
        return now.getHours() * 60 + now.getMinutes();
    },

    /**
     * Filter stops for current day type
     */
    getStopsForToday(train) {
        return train.stops.filter(stop => stop.daytype === this.dayType);
    },

    /**
     * Find segment between two stations
     */
    findSegment(fromStationId, toStationId) {
        return this.segments.find(seg =>
            (seg.from_station === fromStationId && seg.to_station === toStationId) ||
            (seg.from_station === toStationId && seg.to_station === fromStationId)
        );
    },

    /**
     * Calculate all train positions at current time
     */
    getActiveTrains(currentMinutes = null) {
        if (currentMinutes === null) {
            currentMinutes = this.getCurrentMinutes();
        }

        const activeTrains = [];

        this.timetable.forEach(train => {
            const stops = this.getStopsForToday(train);
            if (stops.length < 2) return;

            // Find where the train is right now
            for (let i = 0; i < stops.length - 1; i++) {
                const currentStop = stops[i];
                const nextStop = stops[i + 1];

                const departureTime = this.parseTime(currentStop.departure_time);
                const arrivalTime = this.parseTime(nextStop.arrival_time);

                // Check if train is at station (dwelling)
                const arriveAtCurrent = this.parseTime(currentStop.arrival_time);
                if (currentMinutes >= arriveAtCurrent && currentMinutes < departureTime) {
                    // Train is at station
                    activeTrains.push({
                        id: train.train_id,
                        service_type: train.service_type,
                        station_id: currentStop.station_id,
                        station_name: currentStop.station_name,
                        platform: currentStop.platform,
                        status: 'stopped',
                        progress: 0,
                        segment_id: null,
                        color: this.getTrainColor(train.train_id),
                    });
                    break;
                }

                // Check if train is between stations (moving)
                if (currentMinutes >= departureTime && currentMinutes <= arrivalTime) {
                    const segment = this.findSegment(currentStop.station_id, nextStop.station_id);
                    const totalTime = arrivalTime - departureTime;
                    const elapsed = currentMinutes - departureTime;
                    const progress = totalTime > 0 ? elapsed / totalTime : 0;

                    // Determine direction
                    const direction = segment && segment.from_station === currentStop.station_id
                        ? 'forward' : 'backward';

                    activeTrains.push({
                        id: train.train_id,
                        service_type: train.service_type,
                        from_station: currentStop.station_id,
                        to_station: nextStop.station_id,
                        segment_id: segment ? segment.id : null,
                        progress: Math.min(1, Math.max(0, progress)),
                        direction: direction,
                        status: 'moving',
                        color: this.getTrainColor(train.train_id),
                    });
                    break;
                }
            }
        });

        return activeTrains;
    },

    /**
     * Detect scheduling conflicts
     */
    detectConflicts() {
        const conflicts = {
            platform: [],
            segment: [],
        };

        // Check every minute of the day for conflicts
        for (let minute = 0; minute < 24 * 60; minute += 5) {
            const trains = this.getActiveTrains(minute);

            // Platform conflicts: same station + platform + overlapping time
            const stoppedTrains = trains.filter(t => t.status === 'stopped');
            for (let i = 0; i < stoppedTrains.length; i++) {
                for (let j = i + 1; j < stoppedTrains.length; j++) {
                    const t1 = stoppedTrains[i];
                    const t2 = stoppedTrains[j];

                    if (t1.station_id === t2.station_id && t1.platform === t2.platform) {
                        const conflictKey = `${t1.id}-${t2.id}-${t1.station_id}-${t1.platform}`;
                        if (!conflicts.platform.some(c => c.key === conflictKey)) {
                            conflicts.platform.push({
                                key: conflictKey,
                                trains: [t1.id, t2.id],
                                station: t1.station_name || t1.station_id,
                                platform: t1.platform,
                                time: `${Math.floor(minute / 60).toString().padStart(2, '0')}:${(minute % 60).toString().padStart(2, '0')}`,
                                severity: 'high',
                            });
                        }
                    }
                }
            }

            // Segment conflicts: same segment at same time
            const movingTrains = trains.filter(t => t.status === 'moving' && t.segment_id);
            for (let i = 0; i < movingTrains.length; i++) {
                for (let j = i + 1; j < movingTrains.length; j++) {
                    const t1 = movingTrains[i];
                    const t2 = movingTrains[j];

                    if (t1.segment_id === t2.segment_id) {
                        // Only conflict if going same direction and too close
                        if (t1.direction === t2.direction && Math.abs(t1.progress - t2.progress) < 0.2) {
                            const conflictKey = `${t1.id}-${t2.id}-${t1.segment_id}`;
                            if (!conflicts.segment.some(c => c.key === conflictKey)) {
                                conflicts.segment.push({
                                    key: conflictKey,
                                    trains: [t1.id, t2.id],
                                    segment: t1.segment_id,
                                    time: `${Math.floor(minute / 60).toString().padStart(2, '0')}:${(minute % 60).toString().padStart(2, '0')}`,
                                    severity: 'medium',
                                });
                            }
                        }
                    }
                }
            }
        }

        return conflicts;
    },

    /**
     * Get summary of conflicts for display
     */
    getConflictSummary() {
        const conflicts = this.detectConflicts();
        return {
            totalPlatformConflicts: conflicts.platform.length,
            totalSegmentConflicts: conflicts.segment.length,
            platformConflicts: conflicts.platform,
            segmentConflicts: conflicts.segment,
            hasConflicts: conflicts.platform.length > 0 || conflicts.segment.length > 0,
        };
    },
};

// Export for use in other modules
window.TimetableEngine = TimetableEngine;
