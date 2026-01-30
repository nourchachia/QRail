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

        // Initialize charts
        this.initDelayEvolutionChart();

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
     * FIX: Make each resolution different based on effectiveness
     */
    calculateSimulationState(scenario, timeOffsetMinutes) {
        const incident = this.incident;

        // Different delay curves for each scenario
        let delayMultiplier = 1.0;

        if (scenario === 'baseline') {
            // Baseline: delays keep growing (no intervention)
            if (timeOffsetMinutes <= 15) {
                delayMultiplier = 1 + (timeOffsetMinutes / 15) * 1.5;
            } else {
                delayMultiplier = 2.5 + (timeOffsetMinutes - 15) / 25;
            }
        } else {
            // FIX: Resolutions differ based on expected_outcome quality
            const resolutionIdx = scenario === 'resolutionA' ? 0 : scenario === 'resolutionB' ? 1 : 2;
            const resolution = this.resolutions[resolutionIdx];

            // Extract quality from resolution (0.5 to 0.95 range)
            const effectiveness = resolution?.expected_outcome || resolution?.confidence || 0.65;

            // Better resolutions have:
            // - Earlier peak (faster intervention)
            // - Lower peak (less max delay)
            // - Faster decay (quicker recovery)
            const peakTime = 12 - (effectiveness * 4);        // 8-12 minutes (better = earlier)
            const peakDelay = 2.0 - (effectiveness * 0.8);    // 1.2-2.0x (better = lower)
            const decayRate = 0.08 + (effectiveness * 0.08);  // 0.08-0.16 (better = faster)

            if (timeOffsetMinutes <= peakTime) {
                // Growth phase to peak
                delayMultiplier = 1 + (timeOffsetMinutes / peakTime) * peakDelay;
            } else {
                // Decay phase after intervention
                delayMultiplier = (1 + peakDelay) * Math.exp(-decayRate * (timeOffsetMinutes - peakTime));
            }
        }

        // FIX: Use actual incident data for train count
        const baseAffectedTrains = this.incident?.parsed?.station_ids?.length > 0
            ? this.incident.parsed.station_ids.length * 2  // Estimate: 2 trains per affected station
            : 5;  // Fallback

        // FIX: Base delay from incident or simulation
        let baseDelay = 15; // Default
        if (window.simulation && window.simulation.getIncidentProgression) {
            const progression = window.simulation.getIncidentProgression();
            baseDelay = progression.delayMinutes || 15;
        } else if (this.incident?.severity) {
            const severityMap = { low: 8, medium: 15, high: 25, critical: 35 };
            baseDelay = severityMap[this.incident.severity] || 15;
        }

        const avgDelay = baseDelay * delayMultiplier;
        const affectedTrains = Math.max(1, Math.min(baseAffectedTrains, Math.ceil(delayMultiplier * baseAffectedTrains / 2)));

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
            this.updateAllMaps();
        }, 1200); // Slower animation (1.2s per step) for better visibility

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
    /**
     * Initialize the exponential delay evolution chart
     */
    initDelayEvolutionChart() {
        const ctx = document.getElementById('delay-evolution-chart');
        if (!ctx) return;

        // Destroy existing chart if any
        if (this.delayChart) {
            this.delayChart.destroy();
        }

        // Generate time labels (0 to maxFutureTime)
        const labels = Array.from({ length: this.maxFutureTime + 1 }, (_, i) => `T+${i}`);

        this.delayChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Baseline (Do Nothing)',
                        data: [], // Filled dynamically
                        borderColor: '#ef4444', // Red
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        tension: 0.4, // Smooth exponential curve
                        fill: true
                    },
                    {
                        label: 'Resolution A',
                        data: [],
                        borderColor: '#3b82f6', // Blue
                        borderWidth: 2,
                        tension: 0.4,
                        borderDash: [5, 5]
                    },
                    {
                        label: 'Resolution B',
                        data: [],
                        borderColor: '#10b981', // Green
                        borderWidth: 2,
                        tension: 0.4,
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false, // Disable chart animation for performance during update
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8', maxTicksLimit: 10 }
                    },
                    y: {
                        title: { display: true, text: 'Avg Delay (min)', color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' },
                        beginAtZero: true,
                        suggestedMax: 20 // Keep scale reasonable for visualization
                    }
                },
                plugins: {
                    legend: { labels: { color: '#e2e8f0' } },
                    tooltip: { mode: 'index', intersect: false }
                },
                elements: {
                    point: { radius: 0, hoverRadius: 6 } // Hide points for clean line
                }
            }
        });

        // Fix container height to be compact (user request)
        ctx.parentNode.style.height = '200px';

        // Pre-calculate data for the whole timeline
        this.updateDelayEvolutionChart();
    }

    /**
     * Update the delay chart with calculated data
     */
    updateDelayEvolutionChart() {
        if (!this.delayChart) return;

        const timeSteps = this.maxFutureTime + 1;

        // Initialize time labels if not set
        if (!this.delayChart.data.labels || this.delayChart.data.labels.length === 0) {
            this.delayChart.data.labels = Array.from({ length: timeSteps }, (_, i) => `T+${i}`);
        }

        // Add 4th dataset if missing
        if (this.delayChart.data.datasets.length < 4) {
            this.delayChart.data.datasets.push({
                label: 'Resolution C',
                data: [],
                borderColor: '#f59e0b', // Amber/Orange
                borderWidth: 2,
                tension: 0.4,
                borderDash: [5, 5]
            });
        }

        // Calculate series for each scenario
        const baselineData = [];
        const resAData = [];
        const resBData = [];
        const resCData = [];

        for (let t = 0; t < timeSteps; t++) {
            baselineData.push(this.calculateSimulationState('baseline', t).avgDelay);
            resAData.push(this.calculateSimulationState('resolutionA', t).avgDelay);
            resBData.push(this.calculateSimulationState('resolutionB', t).avgDelay);
            resCData.push(this.calculateSimulationState('resolutionC', t).avgDelay);
        }

        this.delayChart.data.datasets[0].data = baselineData;
        this.delayChart.data.datasets[1].data = resAData;
        this.delayChart.data.datasets[2].data = resBData;
        this.delayChart.data.datasets[3].data = resCData;

        this.delayChart.update();
    }

    /**
     * Render trains on the map (FIX: Ensure trains don't disappear)
     */
    renderTrains(sim, activeTrains) {
        const { svg, xScale, yScale } = sim;

        // Join data
        const trains = svg.select('.trains')
            .selectAll('.train-marker')
            .data(activeTrains, d => d.train_id);

        // EXIT
        trains.exit().remove();

        // UPDATE
        trains.transition().duration(200)
            .attr('transform', d => `translate(${xScale(d.coords[0])}, ${yScale(d.coords[1])})`);

        // ENTER
        const enter = trains.enter()
            .append('g')
            .attr('class', 'train-marker')
            .attr('transform', d => `translate(${xScale(d.coords[0])}, ${yScale(d.coords[1])})`);

        // Train body
        enter.append('rect')
            .attr('x', -6).attr('y', -3)
            .attr('width', 12).attr('height', 6)
            .attr('rx', 2)
            .attr('fill', d => d.color || '#fbbf24')
            .attr('stroke', '#000')
            .attr('stroke-width', 1);

        // Train label (optional, can be toggled)
        enter.append('text')
            .attr('y', -6)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', '8px')
            .text(d => d.train_id);
    }
}

// Export singleton instance
window.futureComparison = new FutureComparison();
