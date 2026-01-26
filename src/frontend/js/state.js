/**
 * Application State Management
 * 
 * State machine: idle → detecting → searching → analyzing → resolved
 */

const appState = {
    // Current state
    status: 'idle', // 'idle' | 'detecting' | 'searching' | 'analyzing' | 'resolved'

    // Demo mode
    demoMode: true,

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

    // Active incidents counter
    activeIncidents: 0,

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
            activeIncidents: 0,
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
        active_trains: [
            { id: 'TRN_001', segment_id: 'SEG_001', progress: 0.2, speed: 80, direction: 'forward', status: 'moving' },
            { id: 'TRN_002', segment_id: 'SEG_005', progress: 0.6, speed: 60, direction: 'forward', status: 'moving' },
            { id: 'TRN_003', segment_id: 'SEG_010', progress: 0.4, speed: 0, direction: 'forward', status: 'stopped' },
            { id: 'TRN_004', segment_id: 'SEG_015', progress: 0.8, speed: 45, direction: 'backward', status: 'moving' },
            { id: 'TRN_005', segment_id: 'SEG_020', progress: 0.3, speed: 30, direction: 'forward', status: 'delayed' },
            { id: 'TRN_006', segment_id: 'SEG_025', progress: 0.5, speed: 90, direction: 'forward', status: 'moving' },
            { id: 'TRN_007', segment_id: 'SEG_030', progress: 0.7, speed: 70, direction: 'backward', status: 'moving' },
            { id: 'TRN_008', segment_id: 'SEG_035', progress: 0.1, speed: 55, direction: 'forward', status: 'moving' },
            { id: 'TRN_009', segment_id: 'SEG_001', progress: 0.8, speed: 85, direction: 'backward', status: 'moving' },
            { id: 'TRN_010', segment_id: 'SEG_005', progress: 0.2, speed: 65, direction: 'backward', status: 'moving' },
            { id: 'TRN_011', segment_id: 'SEG_R01', progress: 0.5, speed: 75, direction: 'forward', status: 'moving' },
            { id: 'TRN_012', segment_id: 'SEG_R02', progress: 0.3, speed: 50, direction: 'forward', status: 'moving' },
            { id: 'TRN_013', segment_id: 'SEG_010', progress: 0.9, speed: 0, direction: 'backward', status: 'stopped' },
            { id: 'TRN_014', segment_id: 'SEG_020', progress: 0.7, speed: 35, direction: 'backward', status: 'delayed' },
            { id: 'TRN_015', segment_id: 'SEG_025', progress: 0.2, speed: 95, direction: 'backward', status: 'moving' },
            { id: 'TRN_016', segment_id: 'SEG_030', progress: 0.4, speed: 72, direction: 'forward', status: 'moving' },
            { id: 'TRN_017', segment_id: 'SEG_035', progress: 0.9, speed: 58, direction: 'backward', status: 'moving' },
            { id: 'TRN_018', segment_id: 'SEG_015', progress: 0.1, speed: 48, direction: 'forward', status: 'moving' },
            { id: 'TRN_019', segment_id: 'SEG_040', progress: 0.5, speed: 82, direction: 'forward', status: 'moving' },
            { id: 'TRN_020', segment_id: 'SEG_040', progress: 0.2, speed: 80, direction: 'backward', status: 'moving' }
        ],
    },
};

// Export for use in other modules
window.appState = appState;
window.scenarios = scenarios;
window.mockData = mockData;
