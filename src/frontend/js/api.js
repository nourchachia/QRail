/**
 * API Service - Handles all backend communication
 * 
 * Endpoints:
 * - POST /api/analyze - Main incident analysis
 * - POST /api/search - Search similar incidents
 * - GET /api/stations - Get all stations
 * - GET /api/segments - Get all segments
 * - GET /api/network/status - Get live network status
 * - GET /api/network-data - Get combined network data
 * - POST /api/feedback - Submit feedback
 */

const API_BASE = 'http://localhost:8002';

const api = {
    /**
     * Analyze incident with AI
     * @param {string} text - Incident description
     * @returns {Promise<Object>} Analysis result
     */
    async analyzeIncident(text) {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return response.json();
    },

    /**
     * Search for similar incidents
     * @param {string} query_text - Search query
     * @param {number} limit - Number of results
     * @returns {Promise<Object>} Search results
     */
    async searchSimilar(query_text, limit = 5) {
        const response = await fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query_text, limit }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return response.json();
    },

    /**
     * Get all stations
     * @returns {Promise<Array>} List of stations
     */
    async getStations() {
        const response = await fetch(`${API_BASE}/api/stations`);

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();
        return data.stations;
    },

    /**
     * Get all track segments
     * @returns {Promise<Array>} List of segments
     */
    async getSegments() {
        const response = await fetch(`${API_BASE}/api/segments`);

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();
        return data.segments;
    },

    /**
     * Get combined network data (Stations + Segments + Status)
     * @returns {Promise<Object>} Combined data
     */
    async getCombinedNetworkData() {
        const response = await fetch(`${API_BASE}/api/network-data`);
        if (!response.ok) throw new Error('Failed to fetch combined network data');
        return response.json();
    },

    /**
     * Get live network status
     * @returns {Promise<Object>} Live status data
     */
    async getLiveStatus() {
        const response = await fetch(`${API_BASE}/api/network/status`);

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return response.json();
    },

    /**
     * Submit user feedback
     * @param {Object} feedback - Feedback data
     * @returns {Promise<Object>} Response
     */
    async submitFeedback(feedback) {
        const response = await fetch(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(feedback),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return response.json();
    },
};

// Export base for dynamic use
api.base = API_BASE;
window.api = api;
