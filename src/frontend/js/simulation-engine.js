/**
 * Simulation Engine - Real-Time Clock & Train Lifecycle
 * 
 * Implements Option C: Hybrid time-based simulation
 * - Load initial snapshot from time selector
 * - Advance virtual clock (1 real second = 1 sim minute)
 * - Calculate train positions from timetable in real-time
 * - Integrate with incident resolution flow
 */

// ============================================================================
// SIMULATION STATE
// ============================================================================

let simulationState = {
    isRunning: false,
    isPaused: false,
    currentTime: null,  // Date object for current simulation time
    speed: 60,  // Default: 1 real second = 60 sim seconds (1 minute)
    baseTime: null,  // Starting point for simulation
    intervalId: null,
    dayType: 'weekday',  // Current day type: 'weekday', 'weekend', 'holiday'

    // Incident tracking
    incidentStartTime: null,  // When incident was triggered
    activeIncident: null,     // Current incident data
    baselineTrains: null,     // Trains state before incident
};

// Expose generically for other modules
window.simulationState = simulationState;

/**
 * Set the active day type (weekday/weekend/holiday)
 */
function setDayType(dayType) {
    simulationState.dayType = dayType;
    console.log(`📅 Day type set to: ${dayType}`);

    // Update active button
    document.querySelectorAll('.day-type-btn').forEach(btn => {
        if (btn.dataset.day === dayType) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Recalculate trains with new day type
    if (simulationState.isRunning || simulationState.currentTime) {
        updateTrainsFromTimetable();
    }
}

/**
 * Initialize simulation at a specific time
 * @param {string} timeStr - Format: "HH:MM" (e.g., "08:00")
 */
function initializeSimulation(timeStr) {
    const [hours, minutes] = timeStr.split(':').map(Number);

    // Create date object for today at specified time
    const now = new Date();
    now.setHours(hours, minutes, 0, 0);

    simulationState.currentTime = now;
    simulationState.baseTime = new Date(now);
    simulationState.isRunning = false;
    simulationState.isPaused = false;

    console.log(`🕐 Simulation initialized at ${timeStr}`);
    updateClockDisplay();

    return now;
}

/**
 * Start the simulation clock
 */
function startSimulation() {
    // Initialize if not already done
    if (!simulationState.currentTime) {
        console.log('⚠️ Simulation not initialized, initializing to 08:00');
        initializeSimulation('08:00');
    }

    if (simulationState.intervalId) {
        stopSimulation(); // Clear any existing interval
    }

    simulationState.isRunning = true;
    simulationState.isPaused = false;

    // Remove any existing interval just in case
    if (simulationState.intervalId) {
        clearInterval(simulationState.intervalId);
        simulationState.intervalId = null;
    }

    // Real-time simulation: 1 real second = 1 sim second
    // Update every 1000ms and advance by 1 second
    // Update loop runs every 100ms for smoother animation
    const tickInterval = 100;

    simulationState.intervalId = setInterval(() => {
        if (!simulationState.isPaused && simulationState.currentTime) {
            // Advance time based on speed factor (default 60x = 1 min per sec)
            // We want smooth updates, so we update every 100ms

            // If speed is 60x:
            // 1 sec real time = 60 sec sim time
            // 100ms real time = 6 sec sim time

            // Calculate seconds to advance per tick (assuming 100ms interval)
            const secondsToAdvance = simulationState.speed * 0.1;

            simulationState.currentTime.setSeconds(
                simulationState.currentTime.getSeconds() + secondsToAdvance
            );

            // Update all displays
            updateClockDisplay();
            updateTrainsFromTimetable();

            // Update telemetry every 10 seconds
            if (simulationState.currentTime.getSeconds() % 10 === 0) {
                updateSimulationMetrics();
            }
        }
    }, tickInterval);

    updatePlayPauseButton();
}

/**
 * Pause the simulation (time stops)
 */
function pauseSimulation() {
    simulationState.isPaused = true;
    updatePlayPauseButton();
}

/**
 * Resume the simulation
 */
function resumeSimulation() {
    simulationState.isPaused = false;
    updatePlayPauseButton();
}

/**
 * Stop the simulation completely
 */
function stopSimulation() {
    if (simulationState.intervalId) {
        clearInterval(simulationState.intervalId);
        simulationState.intervalId = null;
    }
    simulationState.isRunning = false;
    simulationState.isPaused = false;
    updatePlayPauseButton();
}

/**
 * Set simulation speed
 * @param {number} speed - Multiplier (1, 2, 4)
 */
function setSimulationSpeed(speed) {
    const wasRunning = simulationState.isRunning && !simulationState.isPaused;

    if (wasRunning) {
        stopSimulation();
    }

    simulationState.speed = speed;

    if (wasRunning) {
        startSimulation();
    } else {
        // If it wasn't running, just update the state but don't start the clock
        updateSpeedButtons();
    }
}

// ============================================================================
// TRAIN LIFECYCLE MANAGEMENT
// ============================================================================

/**
 * Calculate which trains should be active at current simulation time
 * Based on timetable.json schedules
 */
function updateTrainsFromTimetable() {
    if (!window.staticNetworkData || !window.staticNetworkData.segments) {
        console.warn('No network data available for train calculation');
        return;
    }

    if (!simulationState.currentTime) return;

    // Get timetable data (should be loaded globally or fetched)
    const timetable = window.timetableData || [];
    if (timetable.length === 0) {
        console.warn('No timetable data available');
        return;
    }

    const segments = window.staticNetworkData.segments;
    const stations = window.staticNetworkData.stations;
    const currentTime = simulationState.currentTime;
    const dayType = simulationState.dayType;  // Use user-selected day type

    // Calculate train positions
    const activeTrains = calculateActiveTrains(timetable, currentTime, dayType, segments, stations);

    // Store in simulation state for tracking
    simulationState.activeTrainCount = activeTrains.length;

    // Update network view - CRITICAL: Must re-render trains each tick
    if (window.networkView) {
        // Render trains at new positions (D3 handles enter/update/exit)
        window.networkView.renderTrains(activeTrains, segments, stations);
    } else {
        console.warn('⚠️ window.networkView not available');
    }

    // Update metrics - ALWAYS call this to update the count display
    updateTrainCount(activeTrains.length);
    updateNetworkLoad(activeTrains.length);
}

/**
 * Calculate active trains at a specific time
 */
function calculateActiveTrains(timetable, currentTime, dayType, segments, stations) {
    const currentHours = currentTime.getHours();
    const currentMinutes = currentTime.getMinutes();
    const currentSeconds = currentTime.getSeconds();

    // Total seconds since midnight for comparison
    const currentTotalSeconds = currentHours * 3600 + currentMinutes * 60 + currentSeconds;

    const activeTrains = [];
    let debugCount = 0;

    for (const train of timetable) {
        const stops = train.stops.filter(s => s.daytype === dayType);
        if (stops.length < 2) continue;

        // Helper to get total seconds from HH:MM string for sorting
        const getTimeInSeconds = (timeStr) => {
            if (!timeStr) return 0;
            const [h, m] = timeStr.split(':').map(Number);
            return (h * 3600) + (m * 60);
        };

        // Sort by first available time (arrival or departure)
        const sortedStops = stops.sort((a, b) => {
            const timeA = getTimeInSeconds(a.arrival_time || a.departure_time);
            const timeB = getTimeInSeconds(b.arrival_time || b.departure_time);
            return timeA - timeB;
        });

        const firstStop = sortedStops[0];
        const lastStop = sortedStops[sortedStops.length - 1];

        // Parse first/last times (HH:MM format)
        const [startH, startM] = firstStop.departure_time.split(':').map(Number);
        const [endH, endM] = lastStop.arrival_time.split(':').map(Number);

        const startTotalSeconds = startH * 3600 + startM * 60;
        const endTotalSeconds = endH * 3600 + endM * 60;

        // Check if train is active (between first departure and last arrival)
        if (currentTotalSeconds >= startTotalSeconds && currentTotalSeconds <= endTotalSeconds) {
            debugCount++;
            // Find current segment/position
            const position = calculateTrainPosition(train, sortedStops, currentTotalSeconds, segments, stations);

            if (position) {
                // Check if train is affected by current incident
                let status = (position.station_id) ? 'stopped' : 'moving';
                let delay = 0;

                const incident = window.appState && window.appState.analysisResult;
                const isResolved = window.appState && window.appState.status === 'resolved';

                if (incident && incident.parsed) {
                    const affectedStations = incident.parsed.station_ids || [];
                    const affectedSegments = incident.parsed.segment_ids || [];

                    const isStationAffected = affectedStations.includes(position.from_station) ||
                        affectedStations.includes(position.to_station) ||
                        affectedStations.includes(position.station_id);

                    const isSegmentAffected = affectedSegments.includes(position.segment_id);

                    if (isStationAffected || isSegmentAffected) {
                        // Get current delay from incident progression
                        const progression = simulateIncidentProgression();
                        delay = progression.delayMinutes;
                        status = isResolved ? 'recovering' : 'delayed';
                    }
                }

                activeTrains.push({
                    id: train.train_id,
                    ...position,
                    service_type: train.service_type,
                    route: train.route,
                    status: status,
                    delay: delay,
                    speed: getSpeedForServiceType(train.service_type),
                    color: getColorForServiceType(train.service_type)
                });
            }
        }
    }

    return activeTrains;
}

/**
 * Calculate exact position of a train at current time
 */
function calculateTrainPosition(train, sortedStops, currentTotalSeconds, segments, stations) {
    // Find which leg of the journey we're on
    for (let i = 0; i < sortedStops.length - 1; i++) {
        const fromStop = sortedStops[i];
        const toStop = sortedStops[i + 1];

        const [depH, depM] = fromStop.departure_time.split(':').map(Number);
        const [arrH, arrM] = toStop.arrival_time.split(':').map(Number);

        const departureSeconds = depH * 3600 + depM * 60;
        const arrivalSeconds = arrH * 3600 + arrM * 60;

        // Check if we're on this segment
        if (currentTotalSeconds >= departureSeconds && currentTotalSeconds <= arrivalSeconds) {
            // Find the segment - be direction-blind
            const segment = segments.find(s =>
                (s.from_station === fromStop.station_id && s.to_station === toStop.station_id) ||
                (s.from_station === toStop.station_id && s.to_station === fromStop.station_id)
            );

            if (!segment) continue;

            // Determine if we're moving along or against the defined segment direction
            const direction = (segment.from_station === fromStop.station_id) ? 'forward' : 'backward';

            // Calculate progress along segment
            const travelTime = arrivalSeconds - departureSeconds;
            const elapsed = currentTotalSeconds - departureSeconds;
            let progress = travelTime > 0 ? elapsed / travelTime : 0;

            // CRITICAL: If moving backward relative to segment definition, invert progress
            if (direction === 'backward') {
                progress = 1 - progress;
            }

            return {
                segment_id: segment.id,
                progress: Math.min(1, Math.max(0, progress)),
                from_station: fromStop.station_id,
                to_station: toStop.station_id,
                direction: direction,
                delay: 0
            };
        }

        // Check if at station (dwelling)
        const arriveSeconds = arrH * 3600 + arrM * 60;
        if (currentTotalSeconds >= arriveSeconds && currentTotalSeconds < departureSeconds) {
            return {
                station_id: fromStop.station_id,
                from_station: fromStop.station_id, // Keep orientation stations even when stopped
                to_station: toStop.station_id,
                progress: 0,
                delay: 0
            };
        }
    }

    // Special case: Is this the LAST station of the journey?
    const lastStop = sortedStops[sortedStops.length - 1];
    const [lastArrH, lastArrM] = lastStop.arrival_time.split(':').map(Number);
    const lastArrivalSeconds = lastArrH * 3600 + lastArrM * 60;

    if (currentTotalSeconds >= lastArrivalSeconds) {
        return {
            station_id: lastStop.station_id,
            from_station: sortedStops[sortedStops.length - 2].station_id,
            to_station: lastStop.station_id,
            progress: 1, // Stay at the end
            delay: 0
        };
    }

    return null;
}

// ============================================================================
// INCIDENT PROGRESSION SIMULATION
// ============================================================================

/**
 * Simulate incident impact over time
 * Returns delay multiplier and affected train states
 */
function simulateIncidentProgression() {
    if (!simulationState.incidentStartTime || !simulationState.activeIncident) {
        return {
            delayMinutes: 0,
            severity: 'none',
            affectedTrainCount: 0
        };
    }

    // Calculate elapsed time since incident start
    const elapsedMs = simulationState.currentTime - simulationState.incidentStartTime;
    const elapsedMinutes = elapsedMs / 1000 / 60;

    // Get incident severity parameters
    const incident = simulationState.activeIncident;
    const isResolved = window.appState && window.appState.status === 'resolved';

    // Delay progression parameters based on severity
    const severityParams = {
        high: { peakDelay: 25, peakTime: 15, growthRate: 0.3, decayRate: 0.05 },
        medium: { peakDelay: 15, peakTime: 12, growthRate: 0.25, decayRate: 0.06 },
        low: { peakDelay: 8, peakTime: 10, growthRate: 0.2, decayRate: 0.08 }
    };

    const params = severityParams[incident.severity] || severityParams.medium;

    let delayMinutes = 0;
    let severity = 'none';

    if (isResolved) {
        // Faster recovery after resolution
        const recoveryRate = 0.15; // Faster decay
        delayMinutes = params.peakDelay * Math.exp(-recoveryRate * elapsedMinutes);
    } else {
        // Normal cascade progression
        if (elapsedMinutes <= params.peakTime) {
            // Growing phase - exponential growth to peak
            delayMinutes = params.peakDelay * (1 - Math.exp(-params.growthRate * elapsedMinutes));
        } else {
            // Decay phase - exponential decay from peak
            delayMinutes = params.peakDelay * Math.exp(-params.decayRate * (elapsedMinutes - params.peakTime));
        }
    }

    // Determine severity level based on current delay
    if (delayMinutes > 15) severity = 'critical';
    else if (delayMinutes > 8) severity = 'major';
    else if (delayMinutes > 3) severity = 'minor';

    return {
        delayMinutes: Math.round(delayMinutes * 10) / 10, // Round to 1 decimal
        severity,
        elapsedMinutes: Math.round(elapsedMinutes),
        isResolved
    };
}

/**
 * Trigger an incident in the simulation
 */
function triggerIncident(incident) {
    simulationState.incidentStartTime = new Date(simulationState.currentTime);
    simulationState.activeIncident = incident;

    console.log(`🚨 Incident triggered at ${simulationState.currentTime.toLocaleTimeString()}`);

    // Store baseline for comparison
    if (window.networkView && window.networkView.getTrains) {
        simulationState.baselineTrains = window.networkView.getTrains();
    }
}

/**
 * Clear active incident
 */
function clearIncident() {
    simulationState.incidentStartTime = null;
    simulationState.activeIncident = null;
    simulationState.baselineTrains = null;
    console.log('✅ Incident cleared');
}

// ============================================================================
// UI UPDATES
// ============================================================================

function updateClockDisplay() {
    const clockEl = document.getElementById('sim-clock');
    if (!clockEl || !simulationState.currentTime) return;

    const hours = simulationState.currentTime.getHours().toString().padStart(2, '0');
    const minutes = simulationState.currentTime.getMinutes().toString().padStart(2, '0');
    const seconds = simulationState.currentTime.getSeconds().toString().padStart(2, '0');
    clockEl.textContent = `${hours}:${minutes}:${seconds}`;
}

function updatePlayPauseButton() {
    const playBtn = document.getElementById('play-pause-btn');
    if (!playBtn) return;

    if (!simulationState.isRunning || simulationState.isPaused) {
        playBtn.textContent = '▶ Play';
        playBtn.classList.remove('playing');
    } else {
        playBtn.textContent = '⏸ Pause';
        playBtn.classList.add('playing');
    }
}

function updateSpeedButtons() {
    document.querySelectorAll('.speed-btn').forEach(btn => {
        const speed = parseInt(btn.dataset.speed);
        if (speed === simulationState.speed) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

function updateTrainCount(count) {
    const trainValueEl = document.getElementById('trains-value');
    if (trainValueEl) {
        trainValueEl.textContent = count;
    }

    const trainStatusEl = document.getElementById('trains-status');
    if (trainStatusEl) {
        trainStatusEl.textContent = `🚂 Trains: ${count}`;
    }
}

function updateNetworkLoad(trainCount) {
    // Simple heuristic: network load based on train density
    // Assume max capacity is ~50 trains for the network
    const loadPct = Math.min(100, Math.round((trainCount / 50) * 100));

    const loadValueEl = document.getElementById('load-value');
    if (loadValueEl) {
        loadValueEl.textContent = `${loadPct}%`;
    }

    const loadStatusEl = document.getElementById('load-status');
    if (loadStatusEl) {
        loadStatusEl.textContent = `📊 Load: ${loadPct}%`;
    }

    // Update telemetry if available
    if (window.timeline && window.timeline.updateMetrics) {
        window.timeline.updateMetrics({
            network_load_pct: loadPct,
            active_trains: trainCount,
            weather: { condition: 'clear' }
        });
    }
}

function updateSimulationMetrics() {
    // This is called periodically to refresh all metrics
    const trainCount = simulationState.activeTrainCount || 0;
    updateNetworkLoad(trainCount);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getDayType(date) {
    const day = date.getDay(); // 0 = Sunday, 6 = Saturday
    return (day === 0 || day === 6) ? 'weekend' : 'weekday';
}

function getSpeedForServiceType(type) {
    const speeds = {
        express: 120,
        regional: 80,
        local: 60,
        halt: 40
    };
    return speeds[type] || 80;
}

function getColorForServiceType(type) {
    const colors = {
        express: '#3b82f6',  // Blue
        regional: '#8b5cf6', // Purple  
        local: '#10b981',    // Green
        halt: '#f59e0b'      // Amber
    };
    return colors[type] || '#6366f1';
}

// ============================================================================
// EXPORTS
// ============================================================================

window.simulation = {
    init: initializeSimulation,
    start: startSimulation,
    pause: pauseSimulation,
    resume: resumeSimulation,
    stop: stopSimulation,
    setSpeed: setSimulationSpeed,
    setDayType: setDayType,
    getState: () => simulationState,
    updateTrains: updateTrainsFromTimetable,
    triggerIncident: triggerIncident,
    clearIncident: clearIncident,
    getIncidentProgression: simulateIncidentProgression
};
