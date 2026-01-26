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
    speed: 1,  // 1x, 2x, 4x multiplier
    baseTime: null,  // Starting point for simulation
    intervalId: null,
    dayType: 'weekday',  // Current day type: 'weekday', 'weekend', 'holiday'
};

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
    if (simulationState.intervalId) {
        stopSimulation(); // Clear any existing interval
    }

    simulationState.isRunning = true;
    simulationState.isPaused = false;

    // Update every 1000ms (1 second real time = 1 minute sim time)
    const tickInterval = 100 / simulationState.speed; //ms per tick

    simulationState.intervalId = setInterval(() => {
        if (!simulationState.isPaused) {
            // Advance time by 1 minute
            simulationState.currentTime.setMinutes(simulationState.currentTime.getMinutes() + 1);

            // Update all displays
            updateClockDisplay();
            updateTrainsFromTimetable();

            // Update telemetry every 3 minutes
            if (simulationState.currentTime.getMinutes() % 3 === 0) {
                updateSimulationMetrics();
            }
        }
    }, tickInterval);

    console.log(`▶️ Simulation started at ${simulationState.speed}x speed`);
    updatePlayPauseButton();
}

/**
 * Pause the simulation (time stops)
 */
function pauseSimulation() {
    simulationState.isPaused = true;
    console.log('⏸️ Simulation paused');
    updatePlayPauseButton();
}

/**
 * Resume the simulation
 */
function resumeSimulation() {
    simulationState.isPaused = false;
    console.log('▶️ Simulation resumed');
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
    console.log('⏹️ Simulation stopped');
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
    console.log(`⏩ Speed set to ${speed}x`);

    if (wasRunning) {
        startSimulation();
    }

    updateSpeedButtons();
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

    // Update network view
    if (window.networkView && window.networkView.updateTrains) {
        // Update existing train positions (smooth animation)
        window.networkView.updateTrains(activeTrains, segments, stations);
    } else if (window.networkView && window.networkView.renderTrains) {
        // Re-render trains (less smooth but works)
        window.networkView.clearTrains && window.networkView.clearTrains();
        window.networkView.renderTrains(activeTrains, segments, stations);
    }

    // Update metrics
    updateTrainCount(activeTrains.length);
}

/**
 * Calculate active trains at a specific time
 */
function calculateActiveTrains(timetable, currentTime, dayType, segments, stations) {
    const currentMinutes = currentTime.getHours() * 60 + currentTime.getMinutes();
    const activeTrains = [];

    for (const train of timetable) {
        const stops = train.stops.filter(s => s.daytype === dayType);
        if (stops.length < 2) continue;

        // Sort by arrival time
        const sortedStops = stops.sort((a, b) => {
            const [aH, aM] = a.arrival_time.split(':').map(Number);
            const [bH, bM] = b.arrival_time.split(':').map(Number);
            return (aH * 60 + aM) - (bH * 60 + bM);
        });

        const firstStop = sortedStops[0];
        const lastStop = sortedStops[sortedStops.length - 1];

        const [startH, startM] = firstStop.departure_time.split(':').map(Number);
        const [endH, endM] = lastStop.arrival_time.split(':').map(Number);

        const startMinutes = startH * 60 + startM;
        const endMinutes = endH * 60 + endM;

        // Check if train is active (between departure and arrival)
        if (currentMinutes >= startMinutes && currentMinutes <= endMinutes) {
            // Find current segment
            const position = calculateTrainPosition(train, sortedStops, currentMinutes, segments, stations);

            if (position) {
                activeTrains.push({
                    id: train.train_id,
                    ...position,
                    service_type: train.service_type,
                    route: train.route,
                    status: 'moving',
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
function calculateTrainPosition(train, sortedStops, currentMinutes, segments, stations) {
    // Find which leg of the journey we're on
    for (let i = 0; i < sortedStops.length - 1; i++) {
        const fromStop = sortedStops[i];
        const toStop = sortedStops[i + 1];

        const [depH, depM] = fromStop.departure_time.split(':').map(Number);
        const [arrH, arrM] = toStop.arrival_time.split(':').map(Number);

        const departureMin = depH * 60 + depM;
        const arrivalMin = arrH * 60 + arrM;

        // Check if we're on this segment
        if (currentMinutes >= departureMin && currentMinutes <= arrivalMin) {
            // Find the segment
            const segment = segments.find(s =>
                (s.from_station === fromStop.station_id && s.to_station === toStop.station_id) ||
                (s.bidirectional && s.from_station === toStop.station_id && s.to_station === fromStop.station_id)
            );

            if (!segment) continue;

            // Calculate progress along segment
            const travelTime = arrivalMin - departureMin;
            const elapsed = currentMinutes - departureMin;
            const progress = travelTime > 0 ? elapsed / travelTime : 0;

            return {
                segment_id: segment.id,
                progress: Math.min(1, Math.max(0, progress)),
                from_station: fromStop.station_id,
                to_station: toStop.station_id,
                delay: 0  // TODO: Calculate delay if needed
            };
        }

        // Check if at station (dwelling)
        if (currentMinutes >= arrivalMin && currentMinutes < departureMin) {
            return {
                station_id: fromStop.station_id,
                progress: 0,
                delay: 0
            };
        }
    }

    return null;
}

// ============================================================================
// UI UPDATES
// ============================================================================

function updateClockDisplay() {
    const clockEl = document.getElementById('sim-clock');
    if (!clockEl || !simulationState.currentTime) return;

    const hours = simulationState.currentTime.getHours().toString().padStart(2, '0');
    const minutes = simulationState.currentTime.getMinutes().toString().padStart(2, '0');
    clockEl.textContent = `${hours}:${minutes}`;
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
}

function updateSimulationMetrics() {
    // Fetch or calculate current network load based on active trains
    if (window.timeline && window.timeline.updateMetrics) {
        const mockStatus = {
            network_load_pct: Math.min(100, simulationState.activeTrainCount * 5),
            active_trains: simulationState.activeTrainCount || 0,
            weather: { condition: 'clear' }
        };
        window.timeline.updateMetrics(mockStatus);
    }
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
    updateTrains: updateTrainsFromTimetable
};
