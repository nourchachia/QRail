/**
 * Main Application - Orchestrates all components
 * 
 * Initializes:
 * - Network view (D3.js)
 * - Timeline (Chart.js)
 * - Control panel
 * 
 * Loads:
 * - Stations from API
 * - Segments from API
 * - Live status from API
 */

async function initApp() {
    console.log('🚄 Initializing Neural Rail Conductor...');

    try {
        // Initialize components
        window.networkView.init();
        window.timeline.init();
        window.controlPanel.init();

        // Setup demo mode toggle
        setupDemoMode();

        // Setup reset button
        document.getElementById('reset-btn').addEventListener('click', resetApp);

        // Load network data
        await loadNetworkData();

        console.log('✅ Application ready!');

    } catch (error) {
        console.error('❌ Initialization failed:', error);
        showError('Failed to initialize application. Check console for details.');
    }
}

async function loadNetworkData() {
    try {
        console.log('Loading network data...');

        // Try to load from API
        if (!window.appState.demoMode) {
            try {
                const [stations, segments, liveStatus] = await Promise.all([
                    window.api.getStations(),
                    window.api.getSegments(),
                    window.api.getLiveStatus(),
                ]);

                window.appState.setState({
                    stations,
                    segments,
                    liveStatus,
                });

                console.log(`✅ Loaded ${stations.length} stations and ${segments.length} segments from API`);

            } catch (apiError) {
                console.warn('⚠️ API not available, using demo mode:', apiError);
                window.appState.setState({ demoMode: true });
                updateDemoModeUI();
                loadDemoData();
                return;
            }
        } else {
            loadDemoData();
        }

        // Render network
        window.networkView.render(window.appState.stations, window.appState.segments);

        // Update status displays
        window.networkView.updateStatus(window.appState.liveStatus);
        window.timeline.updateMetrics(window.appState.liveStatus);

    } catch (error) {
        console.error('Failed to load network data:', error);
        showError('Failed to load network data. Using demo mode.');
        window.appState.setState({ demoMode: true });
        updateDemoModeUI();
        loadDemoData();
    }
}

async function loadDemoData() {
    console.log('Loading demo data...');

    try {
        // Load from backend API (proper architecture)
        const [stationsResponse, segmentsResponse] = await Promise.all([
            fetch('http://localhost:8000/api/stations'),
            fetch('http://localhost:8000/api/segments')
        ]);

        let stations, segments;

        if (stationsResponse.ok && segmentsResponse.ok) {
            const stationsData = await stationsResponse.json();
            const segmentsData = await segmentsResponse.json();
            stations = stationsData.stations;
            segments = segmentsData.segments;
            console.log('✅ Loaded real network from backend API');
        } else {
            // Fallback to generated data if backend not running
            console.warn('Backend API not available, generating mock data');
            console.warn('To use real data, start backend: python src/api/main.py');
            stations = generateMockStations(50);
            segments = generateMockSegments(stations);
        }

        window.appState.setState({
            stations,
            segments,
            liveStatus: window.mockData.liveStatus,
        });

        window.networkView.render(stations, segments);
        window.networkView.updateStatus(window.mockData.liveStatus);
        window.timeline.updateMetrics(window.mockData.liveStatus);

        // Load timetable and initialize train animation
        loadTimetableAndStartTrains(stations, segments);

        console.log(`✅ Loaded demo network with ${stations.length} stations`);

    } catch (error) {
        console.error('Error loading network data:', error);
        // Fallback to generated data
        const stations = generateMockStations(20);
        const segments = generateMockSegments(stations);

        window.appState.setState({
            stations,
            segments,
            liveStatus: window.mockData.liveStatus,
        });

        window.networkView.render(stations, segments);
        window.networkView.updateStatus(window.mockData.liveStatus);
        window.timeline.updateMetrics(window.mockData.liveStatus);

        loadTimetableAndStartTrains(stations, segments);
        console.log(`✅ Loaded generated network with ${stations.length} stations`);
    }
}

async function loadTimetableAndStartTrains(stations, segments) {
    try {
        // Load timetable.json
        const response = await fetch('/data/network/timetable.json');
        if (!response.ok) {
            console.warn('Could not load timetable.json, using mock trains');
            renderMockTrains(segments, stations);
            return;
        }

        const timetable = await response.json();

        // Also load real segments if available
        let realSegments = segments;
        try {
            const segResponse = await fetch('/data/network/segments.json');
            if (segResponse.ok) {
                realSegments = await segResponse.json();
            }
        } catch (e) {
            console.warn('Using generated segments');
        }

        // Initialize timetable engine
        window.TimetableEngine.init(timetable, realSegments, stations);

        // Detect and log conflicts
        const conflicts = window.TimetableEngine.getConflictSummary();
        if (conflicts.hasConflicts) {
            console.warn('⚠️ Schedule Conflicts Detected:');
            console.warn(`   Platform conflicts: ${conflicts.totalPlatformConflicts}`);
            console.warn(`   Segment conflicts: ${conflicts.totalSegmentConflicts}`);
            conflicts.platformConflicts.forEach(c => {
                console.warn(`   🔴 ${c.trains.join(' & ')} at ${c.station} Platform ${c.platform} @ ${c.time}`);
            });
        } else {
            console.log('✅ No scheduling conflicts detected');
        }

        // Get current active trains
        const activeTrains = window.TimetableEngine.getActiveTrains();
        console.log(`🚂 ${activeTrains.length} trains active now`);

        // Render trains on network
        window.networkView.renderTrains(activeTrains, realSegments, stations);

        // Start animation loop
        startTrainAnimationLoop(realSegments, stations);

    } catch (error) {
        console.error('Error loading timetable:', error);
        renderMockTrains(segments, stations);
    }
}

function renderMockTrains(segments, stations) {
    // Fallback to mock trains from state.js
    const mockTrains = window.mockData.liveStatus.active_trains || [];
    window.networkView.renderTrains(mockTrains, segments, stations);
}

let trainAnimationLoopId = null;

function startTrainAnimationLoop(segments, stations) {
    // Update trains every second
    trainAnimationLoopId = setInterval(() => {
        if (!window.TimetableEngine || !window.TimetableEngine.timetable.length) return;

        const activeTrains = window.TimetableEngine.getActiveTrains();
        window.networkView.updateTrains(activeTrains, segments, stations);

        // Update train count in status bar
        document.getElementById('trains-status').textContent = `🚂 Trains: ${activeTrains.length}`;
    }, 1000);
}

function generateMockStations(count) {
    const stations = [];
    const types = ['major_hub', 'regional', 'local'];

    for (let i = 0; i < count; i++) {
        stations.push({
            id: `STN_${String(i + 1).padStart(3, '0')}`,
            name: `Station ${i + 1}`,
            type: types[Math.floor(Math.random() * types.length)],
            zone: 'core',
            platforms: Math.floor(Math.random() * 8) + 2,
            daily_passengers: Math.floor(Math.random() * 100000),
            coordinates: [
                Math.random() * 100,
                Math.random() * 100,
            ],
            has_switches: Math.random() > 0.5,
            is_junction: Math.random() > 0.7,
        });
    }

    return stations;
}

function generateMockSegments(stations) {
    const segments = [];

    // Connect each station to 2-3 nearest neighbors
    stations.forEach((station, i) => {
        const nearestCount = Math.floor(Math.random() * 2) + 2;

        // Find nearest stations
        const distances = stations
            .map((other, j) => ({
                index: j,
                distance: Math.hypot(
                    station.coordinates[0] - other.coordinates[0],
                    station.coordinates[1] - other.coordinates[1]
                ),
            }))
            .filter(d => d.index !== i)
            .sort((a, b) => a.distance - b.distance)
            .slice(0, nearestCount);

        distances.forEach(({ index }) => {
            // Avoid duplicate segments
            const exists = segments.some(s =>
                (s.from_station === station.id && s.to_station === stations[index].id) ||
                (s.from_station === stations[index].id && s.to_station === station.id)
            );

            if (!exists) {
                segments.push({
                    id: `SEG_${String(segments.length + 1).padStart(3, '0')}`,
                    from_station: station.id,
                    to_station: stations[index].id,
                    length_km: Math.random() * 10 + 1,
                    speed_limit: 120,
                    bidirectional: true,
                    track_type: 'main_line',
                });
            }
        });
    });

    return segments;
}

function setupDemoMode() {
    const toggleBtn = document.getElementById('demo-mode-toggle');

    toggleBtn.addEventListener('click', () => {
        window.appState.setState({
            demoMode: !window.appState.demoMode,
        });

        updateDemoModeUI();

        // Reload network data
        loadNetworkData();
    });
}

function updateDemoModeUI() {
    const toggleBtn = document.getElementById('demo-mode-toggle');
    const status = document.getElementById('demo-mode-status');

    if (window.appState.demoMode) {
        toggleBtn.classList.add('active');
        status.textContent = 'ON';
    } else {
        toggleBtn.classList.remove('active');
        status.textContent = 'OFF';
    }
}

function resetApp() {
    console.log('Resetting application...');

    // Reset state
    window.appState.reset();

    // Clear UI
    document.getElementById('incident-text').value = '';
    document.getElementById('search-status').classList.add('hidden');
    document.getElementById('similar-cases').classList.add('hidden');
    document.getElementById('resolution-options').classList.add('hidden');
    document.getElementById('feedback-form').classList.add('hidden');
    window.timeline.hideComparison();

    // Reset feedback
    document.querySelectorAll('.star').forEach(star => {
        star.classList.remove('active');
        star.textContent = '☆';
    });
    document.getElementById('rating-label').textContent = 'Click to rate';
    document.getElementById('feedback-notes').value = '';
    document.getElementById('submit-feedback-btn').disabled = true;

    // Reload network
    loadNetworkData();

    console.log('✅ Application reset');
}

function showError(message) {
    alert(`❌ Error: ${message}`);
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
