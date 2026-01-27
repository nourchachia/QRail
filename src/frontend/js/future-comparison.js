/**
 * Future Comparison - 4-Map Parallel Timeline Simulation
 * 
 * Visualizes the impact of different resolution strategies:
 * - Map 1: Baseline (no action)
 * - Map 2: Resolution A
 * - Map 3: Resolution B  
 * - Map 4: Resolution C
 */

class FutureComparison {
    constructor() {
        this.isActive = false;
        this.simulations = {
            baseline: null,
            resolutionA: null,
            resolutionB: null,
            resolutionC: null
        };
        this.currentTimeOffset = 0; // Minutes into the future
        this.maxFutureTime = 40; // Show 40 minutes into future
        this.updateInterval = null;
    }

    /**
     * Initialize 4-map comparison view
     */
    activate(incident, resolutions) {
        if (!incident || !resolutions || resolutions.length < 3) {
            console.error('Need incident and 3 resolutions to activate comparison');
            return;
        }

        this.isActive = true;
        this.incident = incident;
        this.resolutions = resolutions.slice(0, 3); // Take first 3
        this.currentTimeOffset = 0;

        // Create comparison UI
        this.createComparisonUI();

        // Initialize all 4 simulations
        this.initializeSimulations();

        // Start timeline playback
        this.startPlayback();

        console.log('🔮 Future Comparison activated with 4 parallel timelines');
    }

    /**
     * Create the 4-map grid UI
     */
    createComparisonUI() {
        // Hide normal view, show comparison container
        const mainContainer = document.querySelector('.app-main');
        if (mainContainer) {
            mainContainer.style.display = 'none';
        }

        // Create comparison container
        let comparisonContainer = document.getElementById('comparison-container');
        if (!comparisonContainer) {
            comparisonContainer = document.createElement('div');
            comparisonContainer.id = 'comparison-container';
            comparisonContainer.className = 'comparison-container';
            document.querySelector('.app').appendChild(comparisonContainer);
        }

        comparisonContainer.innerHTML = `
            <div class="comparison-header">
                <h2>🔮 Future Comparison: 3-Way Resolution Analysis</h2>
                <button class="close-comparison-btn" onclick="futureComparison.deactivate()">
                    ✕ Exit Comparison
                </button>
            </div>

            <div class="comparison-grid">
                <!-- Map 1: Baseline -->
                <div class="comparison-map" data-scenario="baseline">
                    <div class="map-header">
                        <h3>📉 Baseline</h3>
                        <span class="map-subtitle">No Action Taken</span>
                    </div>
                    <svg class="comparison-svg" id="svg-baseline"></svg>
                    <div class="map-metrics">
                        <div class="metric">
                            <span class="metric-label">Avg Delay</span>
                            <span class="metric-value" id="delay-baseline">--</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Affected Trains</span>
                            <span class="metric-value" id="trains-baseline">--</span>
                        </div>
                    </div>
                </div>

                <!-- Map 2: Resolution A -->
                <div class="comparison-map" data-scenario="resolutionA">
                    <div class="map-header">
                        <h3>✨ Resolution A</h3>
                        <span class="map-subtitle" id="subtitle-resA">Strategy A</span>
                    </div>
                    <svg class="comparison-svg" id="svg-resolutionA"></svg>
                    <div class="map-metrics">
                        <div class="metric">
                            <span class="metric-label">Avg Delay</span>
                            <span class="metric-value" id="delay-resolutionA">--</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Affected Trains</span>
                            <span class="metric-value" id="trains-resolutionA">--</span>
                        </div>
                    </div>
                </div>

                <!-- Map 3: Resolution B -->
                <div class="comparison-map" data-scenario="resolutionB">
                    <div class="map-header">
                        <h3>✨ Resolution B</h3>
                        <span class="map-subtitle" id="subtitle-resB">Strategy B</span>
                    </div>
                    <svg class="comparison-svg" id="svg-resolutionB"></svg>
                    <div class="map-metrics">
                        <div class="metric">
                            <span class="metric-label">Avg Delay</span>
                            <span class="metric-value" id="delay-resolutionB">--</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Affected Trains</span>
                            <span class="metric-value" id="trains-resolutionB">--</span>
                        </div>
                    </div>
                </div>

                <!-- Map 4: Resolution C -->
                <div class="comparison-map" data-scenario="resolutionC">
                    <div class="map-header">
                        <h3>✨ Resolution C</h3>
                        <span class="map-subtitle" id="subtitle-resC">Strategy C</span>
                    </div>
                    <svg class="comparison-svg" id="svg-resolutionC"></svg>
                    <div class="map-metrics">
                        <div class="metric">
                            <span class="metric-label">Avg Delay</span>
                            <span class="metric-value" id="delay-resolutionC">--</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Affected Trains</span>
                            <span class="metric-value" id="trains-resolutionC">--</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="comparison-controls">
                <div class="timeline-control">
                    <button id="play-pause-comparison" onclick="futureComparison.togglePlayback()">⏸ Pause</button>
                    <div class="timeline-slider">
                        <input type="range" id="timeline-slider" min="0" max="${this.maxFutureTime}" value="0" 
                               oninput="futureComparison.seekTo(this.value)">
                        <div class="timeline-labels">
                            <span>Now</span>
                            <span id="current-time-label">T+0 min</span>
                            <span>T+${this.maxFutureTime} min</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Update subtitles with resolution details
        this.resolutions.forEach((res, idx) => {
            const label = ['resA', 'resB', 'resC'][idx];
            const subtitle = document.getElementById(`subtitle-${label}`);
            if (subtitle) {
                // Use action_type if available, otherwise fall back to strategy
                let strategyText = 'Strategy ' + String.fromCharCode(65 + idx); // Default: "Strategy A", "Strategy B", etc.

                if (res.actions && res.actions.length > 0 && res.actions[0].action_type) {
                    strategyText = res.actions[0].action_type.replace(/_/g, ' ');
                } else if (res.strategy) {
                    strategyText = res.strategy.replace(/_/g, ' ');
                }

                subtitle.textContent = strategyText;
            }
        });
    }

    /**
     * Initialize all 4 parallel simulations
     */
    initializeSimulations() {
        const networkData = window.staticNetworkData;
        if (!networkData) {
            console.error('No network data available');
            return;
        }

        // Initialize D3 maps for each scenario
        ['baseline', 'resolutionA', 'resolutionB', 'resolutionC'].forEach(scenario => {
            this.initializeMap(scenario, networkData);
        });

        console.log('✅ All 4 simulations initialized');
    }

    /**
     * Initialize a single map view
     */
    initializeMap(scenario, networkData) {
        const svgId = `svg-${scenario}`;
        const svg = d3.select(`#${svgId}`);

        if (svg.empty()) {
            console.error(`SVG not found: ${svgId}`);
            return;
        }

        // Set SVG dimensions
        const width = 600;
        const height = 400;
        svg.attr('width', width).attr('height', height);

        // Create scales for positioning
        const lons = networkData.stations.map(s => s.coordinates[0]);
        const lats = networkData.stations.map(s => s.coordinates[1]);

        const xScale = d3.scaleLinear()
            .domain([Math.min(...lons) - 0.05, Math.max(...lons) + 0.05])
            .range([40, width - 40]);

        const yScale = d3.scaleLinear()
            .domain([Math.min(...lats) - 0.05, Math.max(...lats) + 0.05])
            .range([height - 40, 40]);

        // Store scales for later use
        this.simulations[scenario] = {
            svg,
            xScale,
            yScale,
            width,
            height,
            stations: networkData.stations,
            segments: networkData.segments
        };

        // Render static network
        this.renderStaticNetwork(scenario);
    }

    /**
     * Render static network (stations and segments)
     */
    renderStaticNetwork(scenario) {
        const sim = this.simulations[scenario];
        if (!sim) return;

        const { svg, xScale, yScale, stations, segments } = sim;

        // Clear existing
        svg.selectAll('*').remove();

        // Render segments
        const segmentGroup = svg.append('g').attr('class', 'segments');
        const stationMap = new Map(stations.map(s => [s.id, s]));

        segments.forEach(seg => {
            const from = stationMap.get(seg.from_station);
            const to = stationMap.get(seg.to_station);
            if (from && to) {
                segmentGroup.append('line')
                    .attr('x1', xScale(from.coordinates[0]))
                    .attr('y1', yScale(from.coordinates[1]))
                    .attr('x2', xScale(to.coordinates[0]))
                    .attr('y2', yScale(to.coordinates[1]))
                    .attr('stroke', '#374151')
                    .attr('stroke-width', 1.5);
            }
        });

        // Render stations
        const stationGroup = svg.append('g').attr('class', 'stations');

        stations.forEach(station => {
            const colors = {
                major_hub: '#3b82f6',
                regional: '#6366f1',
                local: '#8b5cf6',
                minor_halt: '#a78bfa'
            };

            stationGroup.append('circle')
                .attr('cx', xScale(station.coordinates[0]))
                .attr('cy', yScale(station.coordinates[1]))
                .attr('r', station.type === 'major_hub' ? 5 : 3)
                .attr('fill', colors[station.type] || '#6366f1')
                .attr('stroke', '#fff')
                .attr('stroke-width', 1);
        });

        // Create train group
        svg.append('g').attr('class', 'trains');
    }

    /**
     * Calculate simulation state at a given time offset
     */
    calculateSimulationState(scenario, timeOffsetMinutes) {
        const incident = this.incident;

        // Different delay curves for each scenario
        let delayMultiplier = 1.0;

        if (scenario === 'baseline') {
            // Baseline: delays keep growing
            if (timeOffsetMinutes <= 15) {
                delayMultiplier = 1 + (timeOffsetMinutes / 15) * 1.5;
            } else {
                delayMultiplier = 2.5 + (timeOffsetMinutes - 15) / 25;
            }
        } else {
            // Resolutions: faster recovery
            const resolutionIdx = scenario === 'resolutionA' ? 0 : scenario === 'resolutionB' ? 1 : 2;
            const resolution = this.resolutions[resolutionIdx];

            // Recovery starts immediately, peaks lower, decays faster
            const peakTime = 8;
            const peakDelay = 1.8;
            const decayRate = 0.12;

            if (timeOffsetMinutes <= peakTime) {
                delayMultiplier = 1 + (timeOffsetMinutes / peakTime) * peakDelay;
            } else {
                delayMultiplier = (1 + peakDelay) * Math.exp(-decayRate * (timeOffsetMinutes - peakTime));
            }
        }

        // Calculate affected trains and delays
        const baseDelay = 12; // Base delay in minutes
        const avgDelay = baseDelay * delayMultiplier;
        const affectedTrains = Math.min(6, Math.ceil(delayMultiplier * 4));

        return {
            avgDelay: Math.max(0, avgDelay),
            affectedTrains,
            delayMultiplier
        };
    }

    /**
     * Update all maps to current timeline position
     */
    updateAllMaps() {
        ['baseline', 'resolutionA', 'resolutionB', 'resolutionC'].forEach(scenario => {
            const state = this.calculateSimulationState(scenario, this.currentTimeOffset);
            this.updateMapMetrics(scenario, state);
            this.updateMapVisualization(scenario, state);
        });

        // Update time label
        const label = document.getElementById('current-time-label');
        if (label) {
            label.textContent = `T+${this.currentTimeOffset} min`;
        }

        // Update slider
        const slider = document.getElementById('timeline-slider');
        if (slider) {
            slider.value = this.currentTimeOffset;
        }
    }

    /**
     * Update metrics display for a map
     */
    updateMapMetrics(scenario, state) {
        const delayEl = document.getElementById(`delay-${scenario}`);
        const trainsEl = document.getElementById(`trains-${scenario}`);

        if (delayEl) {
            const delay = Math.round(state.avgDelay * 10) / 10;
            delayEl.textContent = `${delay} min`;
            delayEl.style.color = delay > 15 ? '#dc2626' : delay > 8 ? '#ef4444' : delay > 3 ? '#fb923c' : '#10b981';
        }

        if (trainsEl) {
            trainsEl.textContent = state.affectedTrains;
        }
    }

    /**
     * Update map visualization (show affected areas)
     */
    updateMapVisualization(scenario, state) {
        const sim = this.simulations[scenario];
        if (!sim) return;

        const { svg, xScale, yScale, stations } = sim;

        // Highlight affected stations (from incident location)
        const affectedStationIds = this.incident.location?.station_ids || [];

        svg.selectAll('.stations circle')
            .attr('stroke', d => 'none')
            .attr('stroke-width', d => 0);

        affectedStationIds.forEach(stationId => {
            const station = stations.find(s => s.id === stationId);
            if (station) {
                const color = state.avgDelay > 15 ? '#dc2626' : state.avgDelay > 8 ? '#ef4444' : '#fb923c';

                svg.selectAll('.stations circle')
                    .filter(function () {
                        const cx = parseFloat(d3.select(this).attr('cx'));
                        const cy = parseFloat(d3.select(this).attr('cy'));
                        const expectedCx = xScale(station.coordinates[0]);
                        const expectedCy = yScale(station.coordinates[1]);
                        return Math.abs(cx - expectedCx) < 1 && Math.abs(cy - expectedCy) < 1;
                    })
                    .attr('fill', color)
                    .attr('stroke', color)
                    .attr('stroke-width', 3)
                    .attr('r', 7);
            }
        });
    }

    /**
     * Start automatic timeline playback
     */
    startPlayback() {
        this.isPlaying = true;
        this.updateInterval = setInterval(() => {
            this.currentTimeOffset += 1;

            if (this.currentTimeOffset > this.maxFutureTime) {
                this.currentTimeOffset = 0; // Loop
            }

            this.updateAllMaps();
        }, 500); // Update every 500ms (2x speed)

        const btn = document.getElementById('play-pause-comparison');
        if (btn) btn.textContent = '⏸ Pause';
    }

    /**
     * Toggle playback
     */
    togglePlayback() {
        if (this.isPlaying) {
            clearInterval(this.updateInterval);
            this.isPlaying = false;
            const btn = document.getElementById('play-pause-comparison');
            if (btn) btn.textContent = '▶ Play';
        } else {
            this.startPlayback();
        }
    }

    /**
     * Seek to specific time
     */
    seekTo(timeOffset) {
        this.currentTimeOffset = parseInt(timeOffset);
        this.updateAllMaps();
    }

    /**
     * Deactivate comparison and return to normal view
     */
    deactivate() {
        this.isActive = false;

        // Stop playback
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        // Remove comparison container
        const container = document.getElementById('comparison-container');
        if (container) {
            container.remove();
        }

        // Show normal view
        const mainContainer = document.querySelector('.app-main');
        if (mainContainer) {
            mainContainer.style.display = 'flex';
        }

        console.log('✅ Future Comparison deactivated');
    }
}

// Export singleton instance
window.futureComparison = new FutureComparison();
