/**
 * Application State Management
 * 
 * State machine: idle → detecting → searching → analyzing → resolved
 */

const appState = {
    // Current state
    status: 'idle', // 'idle' | 'detecting' | 'searching' | 'analyzing' | 'resolved'

    // Demo mode
    demoMode: false,

    // Network data (cached)
    stations: [],
    segments: [],
    liveStatus: null,

    // Current incident
    currentIncident: null,
    analysisResult: null,
    affectedNodes: [],

    // Selected resolution
    selectedResolution: null,

    // Feedback
    feedbackRating: 0,
    feedbackNotes: '',

    /**
     * Update state and notify listeners
     */
    setState(newState) {
        Object.assign(this, newState);
        this.notifyListeners();
    },

    /**
     * Reset to initial state
     */
    reset() {
        this.setState({
            status: 'idle',
            currentIncident: null,
            analysisResult: null,
            affectedNodes: [],
            selectedResolution: null,
            feedbackRating: 0,
            feedbackNotes: '',
        });
    },

    /**
     * State change listeners
     */
    listeners: [],

    subscribe(callback) {
        this.listeners.push(callback);
        return () => {
            this.listeners = this.listeners.filter(cb => cb !== callback);
        };
    },

    notifyListeners() {
        this.listeners.forEach(callback => callback(this));
    },
};

// Predefined scenarios for quick testing
const scenarios = {
    signal_failure: {
        text: 'Signal failure at Central Station during morning peak. Heavy rain. 5 trains affected with cascade delays.',
        location: { station_ids: ['STN_001'], zone: 'core' },
        severity: 'high',
    },

    train_breakdown: {
        text: 'Train breakdown on segment between North Terminal and South Junction. Mechanical failure blocking track. Emergency services dispatched.',
        location: { station_ids: ['STN_002', 'STN_003'], zone: 'core' },
        severity: 'critical',
    },

    power_outage: {
        text: 'Power outage affecting West Hub station. All platforms without power. Backup systems activated but trains halted.',
        location: { station_ids: ['STN_005'], zone: 'core' },
        severity: 'critical',
    },
};

// Mock data for demo mode
const mockData = {
    analysisResult: {
        raw_text: 'Signal failure at Central Station during morning peak.',
        parsed: {
            primary_failure_code: 'SIGNAL_FAIL',
            estimated_delay_minutes: 45,
            confidence: 0.89,
            reasoning: 'Detected signal keywords + peak hour context',
            weather: {
                condition: 'heavy_rain',
                temperature_c: 12,
                wind_speed_kmh: 25,
                visibility_km: 3.5,
            },
            network_load_pct: 85,
        },
        embeddings: {
            semantic: new Array(384).fill(0.5),
            structural: new Array(64).fill(0.3),
            temporal: new Array(64).fill(0.7),
        },
        similar_incidents: [
            {
                incident_id: 'INC_762',
                score: 0.94,
                is_golden: true,
                explanation: {
                    topology_match: 0.92,
                    cascade_pattern: 0.95,
                    semantic_similarity: 0.93,
                },
            },
            {
                incident_id: 'INC_543',
                score: 0.87,
                is_golden: false,
                explanation: {
                    topology_match: 0.85,
                    cascade_pattern: 0.88,
                    semantic_similarity: 0.89,
                },
            },
            {
                incident_id: 'INC_329',
                score: 0.82,
                is_golden: false,
                explanation: {
                    topology_match: 0.80,
                    cascade_pattern: 0.84,
                    semantic_similarity: 0.83,
                },
            },
        ],
        conflicts: {
            headway_violation: 0.82,
            platform_oversubscription: 0.15,
            crew_shortage: 0.05,
            signal_queue: 0.72,
            power_supply: 0.03,
            track_capacity: 0.45,
            weather_safety: 0.91,
            equipment_failure: 0.88,
        },
        recommendations: [
            {
                strategy: 'HOLD_UPSTREAM',
                confidence: 0.92,
                incident_id: 'INC_762',
                type: 'proven',
                description: 'Hold trains upstream to prevent cascade',
                actions: [
                    { action: 'Hold all trains at stations before Central', duration_minutes: 15 },
                    { action: 'Dispatch repair crew to signal box', duration_minutes: 5 },
                    { action: 'Communicate delays to passengers', duration_minutes: 2 },
                ],
                expected_outcome: 0.88,
            },
            {
                strategy: 'REROUTE_ALTERNATE',
                confidence: 0.85,
                incident_id: 'INC_543',
                type: 'historical',
                description: 'Reroute trains via alternate junction',
                actions: [
                    { action: 'Reroute trains through West Hub', duration_minutes: 20 },
                    { action: 'Adjust timetable for affected routes', duration_minutes: 10 },
                ],
                expected_outcome: 0.75,
            },
            {
                strategy: 'EXTEND_DWELL',
                confidence: 0.78,
                type: 'template',
                description: 'Extend dwell times at adjacent stations',
                actions: [
                    { action: 'Increase dwell time by 3 minutes', duration_minutes: 30 },
                    { action: 'Monitor passenger flow', duration_minutes: 45 },
                ],
                expected_outcome: 0.65,
            },
        ],
    },

    liveStatus: {
        timestamp: new Date().toISOString(),
        network_load_pct: 85,
        active_incidents: 0,
        weather: {
            condition: 'heavy_rain',
            temperature_c: 12,
            wind_speed_kmh: 25,
            visibility_km: 3.5,
        },
        active_trains: [],
    },
};

// Export for use in other modules
window.appState = appState;
window.scenarios = scenarios;
window.mockData = mockData;
