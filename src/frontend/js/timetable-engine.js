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
        this.buildGraph(); // Build initial graph
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
     * Get current time as minutes since midnight (with fractional seconds)
     */
    getCurrentMinutes() {
        const now = new Date();
        return now.getHours() * 60 + now.getMinutes() + (now.getSeconds() / 60);
    },

    /**
     * Get current seconds since midnight
     */
    getCurrentSeconds() {
        const now = new Date();
        return now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
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
                        // Context for rotation: Face the NEXT station
                        from_station: currentStop.station_id,
                        to_station: nextStop.station_id,
                        progress: 0,
                        segment_id: null,
                        color: this.getTrainColor(train.train_id),
                    });
                    break;
                }

                // Check if train is between stations (moving)
                if (currentMinutes >= departureTime && currentMinutes <= arrivalTime) {
                    let segment = this.findSegment(currentStop.station_id, nextStop.station_id);
                    let segmentId = segment ? segment.id : null;
                    let segmentDirection = segment && segment.from_station === currentStop.station_id ? 'forward' : 'backward';

                    const totalTime = arrivalTime - departureTime;
                    const elapsed = currentMinutes - departureTime;
                    // Global progress for the entire trip leg (A->C)
                    let globalProgress = totalTime > 0 ? elapsed / totalTime : 0;
                    globalProgress = Math.min(1, Math.max(0, globalProgress));

                    let localProgress = globalProgress;

                    // PATHFINDING LOGIC: If no direct segment, find multi-segment path
                    if (!segment) {
                        const path = this.findPath(currentStop.station_id, nextStop.station_id);

                        if (path && path.length > 0) {
                            // Calculate total length of the path
                            // Note: We use length_km from segments. If missing, assume 1.
                            const totalPathLen = path.reduce((sum, item) => sum + (item.segment.length_km || 1), 0);
                            const distanceTraveled = globalProgress * totalPathLen;

                            let distanceCoveredSoFar = 0;

                            // Find which segment we are currently on
                            for (const item of path) {
                                const segLen = item.segment.length_km || 1;

                                if (distanceTraveled >= distanceCoveredSoFar && distanceTraveled <= distanceCoveredSoFar + segLen) {
                                    // Found the active segment
                                    segment = item.segment;
                                    segmentId = segment.id;
                                    segmentDirection = item.direction; // 'forward' or 'backward' determined by BFS

                                    // Calculate progress relative to THIS segment only
                                    const distOnSegment = distanceTraveled - distanceCoveredSoFar;
                                    localProgress = distOnSegment / segLen;
                                    localProgress = Math.min(1, Math.max(0, localProgress));
                                    break;
                                }

                                distanceCoveredSoFar += segLen;
                            }

                            // Edge case: if loop finished (e.g. progress=1), snap to last segment
                            if (!segmentId && path.length > 0) {
                                const lastItem = path[path.length - 1];
                                segment = lastItem.segment;
                                segmentId = segment.id;
                                segmentDirection = lastItem.direction;
                                localProgress = 1;
                            }
                        } else {
                            // Still no path? Log warning (only occasionally)
                            if (Math.random() < 0.001) console.warn(`No path found: ${currentStop.station_id} -> ${nextStop.station_id}`);
                        }
                    }

                    activeTrains.push({
                        id: train.train_id,
                        service_type: train.service_type,
                        from_station: currentStop.station_id,
                        to_station: nextStop.station_id,
                        segment_id: segmentId,
                        progress: localProgress,
                        direction: segmentDirection,
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

    // --- PATHFINDING & GRAPH UTILS ---

    graph: null, // Adjacency list: { stationId: [ { segment, to } ] }
    pathCache: new Map(), // Cache for BFS results: "from-to" -> [segments]

    /**
     * Build the network graph for pathfinding
     */
    buildGraph() {
        this.graph = {};
        this.pathCache.clear();

        if (!this.segments) return;

        this.segments.forEach(seg => {
            // Forward link
            if (!this.graph[seg.from_station]) this.graph[seg.from_station] = [];
            this.graph[seg.from_station].push({ segment: seg, to: seg.to_station });

            // Backward link (assuming tracks are traversable unless one-way, but segments says bidirectional)
            if (seg.bidirectional) {
                if (!this.graph[seg.to_station]) this.graph[seg.to_station] = [];
                this.graph[seg.to_station].push({ segment: seg, to: seg.from_station });
            }
        });
        console.log('Network graph built for pathfinding.');
    },

    /**
     * Find shortest path between two stations (BFS)
     * Returns array of segments
     */
    findPath(fromId, toId) {
        if (!this.graph) this.buildGraph();

        const cacheKey = `${fromId}-${toId}`;
        if (this.pathCache.has(cacheKey)) return this.pathCache.get(cacheKey);

        const queue = [{ id: fromId, path: [] }];
        const visited = new Set([fromId]);

        while (queue.length > 0) {
            const { id, path } = queue.shift();

            if (id === toId) {
                this.pathCache.set(cacheKey, path);
                return path;
            }

            const neighbors = this.graph[id] || [];
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor.to)) {
                    visited.add(neighbor.to);
                    // Append segment to path
                    queue.push({
                        id: neighbor.to,
                        path: [...path, {
                            segment: neighbor.segment,
                            direction: neighbor.to === neighbor.segment.to_station ? 'forward' : 'backward'
                        }]
                    });
                }
            }
        }

        // No path found
        this.pathCache.set(cacheKey, null);
        return null;
    }
};

// Export for use in other modules
window.TimetableEngine = TimetableEngine;
