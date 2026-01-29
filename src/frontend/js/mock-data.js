/**
 * Mock Data for fallback and testing
 */
const MOCK_DATA = {
    liveStatus: {
        weather: {
            condition: 'clear',
            temperature: 22,
            wind_speed: 5
        },
        network_load_pct: 45,
        active_trains: [], // Will be populated by simulation
        incidents: []
    },

    // Sample station for fallback
    sampleStation: {
        id: "ST_CENTRAL",
        name: "Central Station",
        coordinates: [50, 50],
        type: "hub"
    }
};

window.mockData = MOCK_DATA;
