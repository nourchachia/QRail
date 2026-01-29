/**
 * Global Configuration
 */
const CONFIG = {
    // API Settings
    API_URL: 'http://localhost:8002',
    POLL_INTERVAL: 30000, // 30s
    ANIMATION_SPEED: 1,

    // Map Settings
    MAP: {
        DEFAULT_ZOOM: 1,
        MIN_ZOOM: 0.5,
        MAX_ZOOM: 4,
        STATION_RADIUS: 4,
        SEGMENT_WIDTH: 2,
    },

    // Colors
    COLORS: {
        PRIMARY: '#3b82f6',
        SECONDARY: '#10b981',
        DANGER: '#ef4444',
        WARNING: '#f59e0b',
        BACKGROUND: '#1e293b',
        TEXT: '#f8fafc',
    },

    // Train Status Colors
    TRAIN_STATUS: {
        ON_TIME: '#10b981',
        DELAYED: '#ef4444',
        MOVING: '#3b82f6',
        DWELLING: '#8b5cf6',
    }
};

window.CONFIG = CONFIG;
