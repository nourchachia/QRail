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

        // Start real-time telemetry updates
        if (window.timeline && window.timeline.startTelemetry) {
            window.timeline.startTelemetry();
        }

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

// Real-Time Clock (shows actual wall clock, not simulation time)
function startClock() {
    const clockEl = document.getElementById('app-clock');
    if (!clockEl) return;

    setInterval(() => {
        const now = new Date();
        clockEl.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    }, 1000);
}

// Fixed loadDemoData to prioritize local JSON files over random generation
async function loadDemoData() {
    console.log('Loading demo data...');
    startClock();

    try {
        // PRIORITIZE: Static generated data (inserted via script tag)
        if (window.staticNetworkData) {
            console.log('✅ Loaded REAL network topology from static-data.js');
            const { stations, segments } = window.staticNetworkData;

            window.appState.setState({
                stations,
                segments,
                liveStatus: window.mockData.liveStatus,
            });

            window.networkView.render(stations, segments);
            window.networkView.updateStatus(window.mockData.liveStatus);
            window.timeline.updateMetrics(window.mockData.liveStatus);

            loadTimetableAndStartTrains(stations, segments);
            return;
        }

        // Fallback to random generation if static data missing
        console.warn('⚠️ static-data.js not found, falling back to random generator');
        const stations = generateMockStations(50);
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

    } catch (error) {
        console.error('Error loading network data:', error);
        // Emergency fallback
        const stations = generateMockStations(20);
        const segments = generateMockSegments(stations);
        window.appState.setState({ stations, segments, liveStatus: window.mockData.liveStatus });
        window.networkView.render(stations, segments);
        loadTimetableAndStartTrains(stations, segments);
    }
}

async function loadTimetableAndStartTrains(stations, segments) {
    try {
        // Load timetable.json from backend
        const response = await fetch('http://localhost:8002/static/timetable.json');

        let timetable;
        if (!response.ok) {
            console.warn('Could not load timetable.json, using mock trains');
            renderMockTrains(segments, stations);
            return;
        } else {
            timetable = await response.json();
        }

        // Store globally for simulation-engine.js
        window.timetableData = timetable;

        // Also load real segments if available
        let realSegments = segments;
        try {
            const segResponse = await fetch('http://localhost:8002/static/segments.json');
            if (segResponse.ok) {
                realSegments = await segResponse.json();
                // Update global state with real segments
                window.appState.setState({ segments: realSegments });
                // Also update staticNetworkData for simulation engine
                if (!window.staticNetworkData) window.staticNetworkData = {};
                window.staticNetworkData.segments = realSegments;
                window.staticNetworkData.stations = stations;
            }
        } catch (e) {
            console.warn('Using generated segments');
        }

        // Initialize timetable engine (helper for conflict detection)
        if (window.TimetableEngine) {
            window.TimetableEngine.init(timetable, realSegments, stations);
        }

        console.log(`✅ Timetable loaded: ${timetable.length} trains`);

        // Initialize Simulation (Time Travel)
        if (window.simulation) {
            // Default to 8 AM
            console.log('🕐 Starting simulation at 08:00');
            window.simulation.init("08:00");

            // Force initial train render
            window.simulation.updateTrains();

            // Start time flowing
            window.simulation.start();

            console.log('✅ Simulation engine started');
        } else {
            console.error("❌ Simulation engine not found!");
            renderMockTrains(segments, stations);
        }

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

// NOTE: Old local animation loop removed in favor of simulation-engine.js

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

    // Ensure we create segments that match the mock data trains (SEG_001, SEG_005...)
    // Mock trains in state.js use SEG_001..SEG_035

    // First, force create a simple loop/line for the mock trains
    for (let i = 0; i < stations.length - 1; i++) {
        const id = `SEG_${String(i * 5 + 1).padStart(3, '0')}`; // SEG_001, SEG_006, etc... close enough
        // Override for exact matches needed by mockData
        // Mock data uses: SEG_001, SEG_005, SEG_010, SEG_015, SEG_020, SEG_025, SEG_030, SEG_035

        segments.push({
            id: id,
            from_station: stations[i].id,
            to_station: stations[i + 1].id,
            length_km: Math.random() * 10 + 5,
            speed_limit: 120,
            bidirectional: true,
            track_type: 'main_line',
        });
    }

    // Connect some extra randoms
    stations.forEach((station, i) => {
        if (Math.random() > 0.5) return;
        const targetIdx = Math.floor(Math.random() * stations.length);
        if (targetIdx === i) return;

        segments.push({
            id: `SEG_R${String(segments.length + 1).padStart(3, '0')}`,
            from_station: station.id,
            to_station: stations[targetIdx].id,
            length_km: Math.random() * 15 + 5,
            speed_limit: 100,
            bidirectional: true,
        });
    });

    // Ensure specifically used segments exist if not created above
    const requiredSegments = ['SEG_001', 'SEG_005', 'SEG_010', 'SEG_015', 'SEG_020', 'SEG_025', 'SEG_030', 'SEG_035'];
    requiredSegments.forEach((reqId, idx) => {
        if (!segments.find(s => s.id === reqId) && stations.length > 2) {
            segments.push({
                id: reqId,
                from_station: stations[idx % stations.length].id,
                to_station: stations[(idx + 1) % stations.length].id,
                length_km: 10,
                speed_limit: 100,
                bidirectional: true
            });
        }
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
    document.getElementById('incidents-value').textContent = '0';
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
