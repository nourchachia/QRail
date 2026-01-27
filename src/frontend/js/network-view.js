/**
 * Network View Component - D3.js visualization
 * 
 * Renders:
 * - 50 stations (colored by type)
 * - 70 track segments
 * - Live trains (animated)
 * - Affected nodes (pulse animation)
 */

let networkSvg, xScale, yScale;
let stationNodes, segmentLines, trainMarkers;
let trainAnimationInterval = null;
let trainMarkerGroup = null;

// Global state for train routes (to fix looping issue)
let trainRouteState = new Map(); // trainId -> { routeSegmentIds: [], currentSegmentIndex: 0, progress: 0, dwellTimeRemaining: 0 }


/**
 * Build complete route from timetable data
 * Returns array of segment IDs representing the journey
 */
function buildTrainRoute(trainId, timetableData, segments, stationMap) {
    if (!timetableData || !Array.isArray(timetableData)) {
        console.warn('No timetable data available for route building');
        return null;
    }

    const trainData = timetableData.find(t => t.train_id === trainId);
    if (!trainData || !trainData.stops) return null;

    // Get current day type from live status (default to weekday)
    const dayType = window.appState?.liveStatus?.day_type || 'weekday';

    // Filter stops by day type and sort by arrival time
    const relevantStops = trainData.stops
        .filter(s => s.daytype === dayType)
        .sort((a, b) => {
            const timeA = a.arrival_time.split(':').map(Number);
            const timeB = b.arrival_time.split(':').map(Number);
            return (timeA[0] * 60 + timeA[1]) - (timeB[0] * 60 + timeB[1]);
        });

    if (relevantStops.length < 2) return null;

    // Extract station sequence
    const stationSequence = [...new Set(relevantStops.map(s => s.station_id))]; // Remove duplicates

    // Convert to segment sequence
    const routeSegmentIds = [];
    for (let i = 0; i < stationSequence.length - 1; i++) {
        const from = stationSequence[i];
        const to = stationSequence[i + 1];

        // Find segment between these stations
        const segment = segments.find(s =>
            (s.from_station === from && s.to_station === to) ||
            (s.bidirectional && s.from_station === to && s.to_station === from)
        );

        if (segment) {
            routeSegmentIds.push(segment.id);
        }
    }

    return routeSegmentIds.length > 0 ? routeSegmentIds : null;
}



function initNetworkView() {
    const container = document.getElementById('network-svg');
    if (!container) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    networkSvg = d3.select('#network-svg')
        .attr('width', width)
        .attr('height', height);

    // Create zoom container group
    let zoomContainer = networkSvg.select('.zoom-container');
    if (zoomContainer.empty()) {
        zoomContainer = networkSvg.append('g')
            .attr('class', 'zoom-container');
    }

    // Create D3 zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.5, 4])  // Allow zoom from 0.5x to 4x
        .on('zoom', (event) => {
            zoomContainer.attr('transform', event.transform);
        });

    // Apply zoom behavior to SVG
    networkSvg.call(zoom);

    // Store zoom behavior for external control
    networkSvg.zoomBehavior = zoom;
    networkSvg.zoomContainer = zoomContainer;

    // Create scales for positioning (ZOOMED OUT for better spacing)
    // Reduced domain to show less area = stations appear more spread out
    xScale = d3.scaleLinear()
        .domain([-5, 125])  // Moderately expanded to show peripheral stations without over-zooming out
        .range([100, width - 100]);

    yScale = d3.scaleLinear()
        .domain([-5, 125])  // Moderately expanded to show peripheral stations without over-zooming out
        .range([height - 100, 100]);
}

function renderNetwork(stations, segments) {
    if (!networkSvg) initNetworkView();

    // Get or create zoom container
    let zoomContainer = networkSvg.select('.zoom-container');
    if (zoomContainer.empty()) {
        zoomContainer = networkSvg.append('g').attr('class', 'zoom-container');
        networkSvg.zoomContainer = zoomContainer;
    }

    // Clear only the zoom container content, not the whole SVG
    zoomContainer.selectAll('*').remove();

    // Render segments first (so they're behind stations)
    renderSegments(segments, stations);

    // Render stations
    renderStations(stations);
}

function renderSegments(segments, stations) {
    const stationMap = new Map(stations.map(s => [s.id, s]));
    const zoomContainer = networkSvg.zoomContainer || networkSvg.select('.zoom-container');

    const lines = zoomContainer.selectAll('.segment-line')
        .data(segments)
        .enter()
        .append('line')
        .attr('class', 'segment-line')
        .attr('x1', d => {
            const from = stationMap.get(d.from_station);
            return from ? xScale(from.coordinates[0]) : 0;
        })
        .attr('y1', d => {
            const from = stationMap.get(d.from_station);
            return from ? yScale(from.coordinates[1]) : 0;
        })
        .attr('x2', d => {
            const to = stationMap.get(d.to_station);
            return to ? xScale(to.coordinates[0]) : 0;
        })
        .attr('y2', d => {
            const to = stationMap.get(d.to_station);
            return to ? yScale(to.coordinates[1]) : 0;
        });

    segmentLines = lines;
}

function renderStations(stations) {
    const zoomContainer = networkSvg.zoomContainer || networkSvg.select('.zoom-container');

    const groups = zoomContainer.selectAll('.station-node')
        .data(stations)
        .enter()
        .append('g')
        .attr('class', d => `station-node station-${d.type}`)
        .attr('data-station-id', d => d.id)
        .attr('transform', d => `translate(${xScale(d.coordinates[0])}, ${yScale(d.coordinates[1])})`);

    // Station circles
    groups.append('circle')
        .attr('class', 'station-circle')
        .attr('r', d => {
            // Size by type
            if (d.type === 'major_hub') return 10;
            if (d.type === 'regional') return 7;
            if (d.type === 'local') return 5;
            return 3;
        })
        .attr('stroke-width', 2);

    // Station labels (for all stations)
    groups.append('text')
        .attr('class', 'station-label')
        .attr('x', 0)
        .attr('y', -14) // Slightly above the circle
        .attr('text-anchor', 'middle')
        .text(d => d.name)
        .attr('font-size', d => {
            // Reduce font sizes for less clutter
            if (d.type === 'major_hub') return '11px';
            if (d.type === 'regional') return '9px';
            return '0px';  // Hide minor station labels entirely
        })
        .attr('font-weight', d => d.type === 'major_hub' ? 'bold' : 'normal')
        .attr('fill', '#e2e8f0') // Light text for dark theme
        .attr('pointer-events', 'none')
        .style('text-shadow', '2px 2px 4px rgba(0,0,0,0.8)'); // Shadow for readability

    stationNodes = groups;
}

function highlightAffectedNodes(nodeIds) {
    if (!stationNodes) return;

    // Reset all stations first
    stationNodes.classed('station-affected', false);

    // Highlight affected stations
    nodeIds.forEach(nodeId => {
        stationNodes.filter(d => d.id === nodeId)
            .classed('station-affected', true);
    });

    console.log(`Highlighted ${nodeIds.length} affected stations`);
}

function animateCascadeWave(affectedNodes, stations) {
    const stationMap = new Map(stations.map(s => [s.id, s]));
    const epicenter = stationMap.get(affectedNodes[0]);
    if (!epicenter) return;

    // Sort nodes by distance from epicenter
    const sorted = affectedNodes
        .map(id => stationMap.get(id))
        .filter(Boolean)
        .sort((a, b) => {
            const distA = Math.hypot(
                a.coordinates[0] - epicenter.coordinates[0],
                a.coordinates[1] - epicenter.coordinates[1]
            );
            const distB = Math.hypot(
                b.coordinates[0] - epicenter.coordinates[0],
                b.coordinates[1] - epicenter.coordinates[1]
            );
            return distA - distB;
        });

    // Animate with staggered delays
    sorted.forEach((station, i) => {
        setTimeout(() => {
            const x = xScale(station.coordinates[0]);
            const y = yScale(station.coordinates[1]);

            // Add pulse circle
            networkSvg.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', 10)
                .attr('fill', 'none')
                .attr('stroke', '#ef4444')
                .attr('stroke-width', 3)
                .attr('opacity', 0.8)
                .transition()
                .duration(800)
                .attr('r', 40)
                .attr('opacity', 0)
                .remove();
        }, i * 250);
    });
}

function animateRecovery(affectedNodes, stations) {
    const stationMap = new Map(stations.map(s => [s.id, s]));

    affectedNodes.forEach((nodeId, i) => {
        setTimeout(() => {
            const station = stationMap.get(nodeId);
            if (!station) return;

            const x = xScale(station.coordinates[0]);
            const y = yScale(station.coordinates[1]);

            // Green recovery pulse
            networkSvg.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', 10)
                .attr('fill', 'none')
                .attr('stroke', '#10b981')
                .attr('stroke-width', 3)
                .transition()
                .duration(1000)
                .attr('r', 35)
                .attr('opacity', 0)
                .remove();

            // Change station color to green
            stationNodes.filter(d => d.id === nodeId)
                .classed('station-affected', false)
                .classed('station-recovering', true);
        }, i * 200);
    });
}

/**
 * Format weather condition from snake_case to Title Case
 * Example: "heavy_rain" → "Heavy Rain"
 */
function formatWeatherCondition(condition) {
    if (!condition) return 'Clear';

    return condition
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function updateNetworkStatus(liveStatus) {
    if (!liveStatus) return;

    // Update status bar
    const weather = liveStatus.weather || {};
    const weatherEl = document.getElementById('weather-status');
    if (weatherEl) {
        weatherEl.textContent = `🌦️ Weather: ${formatWeatherCondition(weather.condition)}`;
    }

    const loadEl = document.getElementById('load-status');
    if (loadEl) {
        loadEl.textContent = `📊 Load: ${liveStatus.network_load_pct || 0}%`;
    }

    const trainCount = liveStatus.active_trains ? liveStatus.active_trains.length : 0;
    const trainsEl = document.getElementById('trains-status');
    if (trainsEl) {
        trainsEl.textContent = `🚂 Trains: ${trainCount}`;
    }
}

// ============================================================================
// Train Animation Functions
// ============================================================================

/**
 * Get the position (x, y) on a segment given progress (0 to 1)
 */
function getPositionOnSegment(segment, progress, stationMap) {
    const from = stationMap.get(segment.from_station);
    const to = stationMap.get(segment.to_station);

    if (!from || !to) return null;

    // Linear interpolation between stations
    const x = from.coordinates[0] + (to.coordinates[0] - from.coordinates[0]) * progress;
    const y = from.coordinates[1] + (to.coordinates[1] - from.coordinates[1]) * progress;

    return { x: xScale(x), y: yScale(y) };
}

/**
 * Calculate rotation angle for train direction indicator
 */
function getTrainRotation(fromStationId, toStationId, stationMap) {
    const from = stationMap.get(fromStationId);
    const to = stationMap.get(toStationId);

    if (!from || !to) return 0;

    // Use pixel coordinates to calculate screen rotation correctly
    const x1 = xScale(from.coordinates[0]);
    const y1 = yScale(from.coordinates[1]);
    const x2 = xScale(to.coordinates[0]);
    const y2 = yScale(to.coordinates[1]);

    const dx = x2 - x1;
    const dy = y2 - y1;
    let angle = Math.atan2(dy, dx) * (180 / Math.PI);

    return angle;
}


/**
 * Render train markers on the network
 */
function renderTrains(trains, segments, stations) {
    if (!networkSvg) return;

    const stationMap = new Map(stations.map(s => [s.id, s]));
    const segmentMap = new Map(segments.map(s => [s.id, s]));

    // Get zoom container (trains must be inside it to zoom/pan with map)
    const zoomContainer = networkSvg.zoomContainer || networkSvg.select('.zoom-container');

    // Use existing group or create new one INSIDE zoom container
    trainMarkerGroup = zoomContainer.select('.train-group');
    if (trainMarkerGroup.empty()) {
        trainMarkerGroup = zoomContainer.append('g').attr('class', 'train-group');
    }

    // Filter valid trains
    const validTrains = trains.filter(train => {
        if (train.segment_id && segmentMap.has(train.segment_id)) return true;
        if (train.station_id && stationMap.has(train.station_id)) return true;
        return false;
    });

    // Helper to calculate transform
    const getTransform = (d) => {
        if (d.segment_id && segmentMap.has(d.segment_id)) {
            const segment = segmentMap.get(d.segment_id);
            const pos = getPositionOnSegment(segment, d.progress !== undefined ? d.progress : 0.5, stationMap);
            // Use the absolute travel endpoints (from/to) for rotation - no more direction flags needed!
            const rotation = getTrainRotation(d.from_station, d.to_station, stationMap);
            return pos ? `translate(${pos.x}, ${pos.y}) rotate(${rotation})` : 'translate(0, 0)';
        } else if (d.station_id && stationMap.has(d.station_id)) {
            const station = stationMap.get(d.station_id);
            const x = xScale(station.coordinates[0]);
            const y = yScale(station.coordinates[1]);
            // Still rotate even if dwelling
            const rotation = getTrainRotation(d.from_station, d.to_station, stationMap);
            return `translate(${x}, ${y}) rotate(${rotation})`;
        }
        return 'translate(0, 0)';
    };

    // DATA JOIN
    const trainGroups = trainMarkerGroup.selectAll('.train-marker')
        .data(validTrains, d => d.id);

    // EXIT
    trainGroups.exit().remove();

    // UPDATE - Smooth transition for existing trains
    // FIX: At high speeds (>2x), disable transition to prevent interruption locking
    const transitionDuration = (window.simulationState && window.simulationState.speed > 2) ? 0 : 100;

    trainGroups.transition()
        .duration(transitionDuration)
        .ease(d3.easeLinear)
        .attr('transform', d => getTransform(d));

    // Update status/color with delay severity
    trainGroups.select('.train-body')
        .attr('fill', d => getTrainColor(d.status, d.delay))
        .attr('class', d => `train-body ${d.status === 'delayed' ? 'train-delayed' : ''}`);

    // Update or add pulsing animation for delayed trains
    trainGroups.each(function (d) {
        const group = d3.select(this);

        if (d.status === 'delayed') {
            // Add or update pulse circle
            let pulse = group.select('.train-pulse-delay');
            if (pulse.empty()) {
                pulse = group.append('circle')
                    .attr('class', 'train-pulse-delay')
                    .attr('r', 18)
                    .attr('fill', 'none')
                    .attr('stroke-width', 2);
            }
            pulse.attr('stroke', getTrainColor(d.status, d.delay))
                .attr('opacity', 0.8);
        } else {
            // Remove pulse if not delayed
            group.select('.train-pulse-delay').remove();
        }

        // Update or add delay badge
        if (d.delay > 0 && d.status === 'delayed') {
            let delayBadge = group.select('.delay-badge');
            if (delayBadge.empty()) {
                delayBadge = group.append('text')
                    .attr('class', 'delay-badge')
                    .attr('x', 20)
                    .attr('y', 5)
                    .attr('text-anchor', 'start')
                    .attr('fill', '#fff')
                    .attr('font-size', '10px')
                    .attr('font-weight', 'bold')
                    .attr('pointer-events', 'none');
            }
            delayBadge.text(`+${Math.round(d.delay)}'`);
        } else {
            group.select('.delay-badge').remove();
        }
    });

    // ENTER - Create new trains
    const enterGroups = trainGroups.enter()
        .append('g')
        .attr('class', d => `train-marker train-${d.status}`)
        .attr('data-train-id', d => d.id)
        .attr('transform', d => getTransform(d));

    enterGroups.append('polygon')
        .attr('class', 'train-body')
        .attr('points', '12,0 -8,6 -8,-6')
        .attr('fill', d => getTrainColor(d.status, d.delay))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    // Add pulse for delayed trains
    enterGroups.filter(d => d.status === 'delayed')
        .append('circle')
        .attr('class', 'train-pulse-delay')
        .attr('r', 18)
        .attr('fill', 'none')
        .attr('stroke', d => getTrainColor(d.status, d.delay))
        .attr('stroke-width', 2)
        .attr('opacity', 0.8);

    enterGroups.append('text')
        .attr('class', 'train-label')
        .attr('x', 0)
        .attr('y', -18)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .attr('pointer-events', 'none')
        .text(d => d.id);

    // Add delay badge for new delayed trains
    enterGroups.filter(d => d.delay > 0 && d.status === 'delayed')
        .append('text')
        .attr('class', 'delay-badge')
        .attr('x', 20)
        .attr('y', 5)
        .attr('text-anchor', 'start')
        .attr('fill', '#fff')
        .attr('font-size', '10px')
        .attr('font-weight', 'bold')
        .attr('pointer-events', 'none')
        .text(d => `+${Math.round(d.delay)}'`);
}

/**
 * Update train positions with smooth animation
 */
function updateTrainPositions(trains, segments, stations) {
    if (!trainMarkerGroup) return;

    const stationMap = new Map(stations.map(s => [s.id, s]));
    const segmentMap = new Map(segments.map(s => [s.id, s]));

    trains.forEach(train => {
        let transformStr = null;

        // Calculate position based on segment or station
        if (train.segment_id && segmentMap.has(train.segment_id)) {
            const segment = segmentMap.get(train.segment_id);
            const pos = getPositionOnSegment(segment, train.progress !== undefined ? train.progress : 0.5, stationMap);
            const rotation = getTrainRotation(segment, train.direction || 'forward', stationMap);
            if (pos) {
                transformStr = `translate(${pos.x}, ${pos.y}) rotate(${rotation})`;
            }
        } else if (train.station_id && stationMap.has(train.station_id)) {
            const station = stationMap.get(train.station_id);
            const x = xScale(station.coordinates[0]);
            const y = yScale(station.coordinates[1]) - 20;
            transformStr = `translate(${x}, ${y})`;
        }

        if (!transformStr) return;

        // Animate to new position
        const trainEl = trainMarkerGroup.select(`[data-train-id="${train.id}"]`);

        if (!trainEl.empty()) {
            trainEl
                .transition()
                .duration(900)
                .ease(d3.easeLinear)
                .attr('transform', transformStr);

            trainEl.select('.train-body')
                .attr('fill', train.color || getTrainColor(train.status));
        }
    });
}

/**
 * Simulate train movement with complete route following and station dwell time
 */
function simulateTrainMovement(trains, timetableData, segments, stations) {
    if (!trains || !segments) return trains || [];

    const stationMap = new Map(stations.map(s => [s.id, s]));

    return trains.map(train => {
        // Only move moving trains
        if (train.status !== 'moving') return train;

        // Get or initialize route state
        if (!trainRouteState.has(train.id)) {
            const route = buildTrainRoute(train.id, timetableData, segments, stationMap);

            if (!route || route.length === 0) {
                // Fallback: if no route found, just use current segment
                trainRouteState.set(train.id, {
                    routeSegmentIds: [train.segment_id],
                    currentSegmentIndex: 0,
                    progress: train.progress || 0,
                    dwellTimeRemaining: 0
                });
            } else {
                // Find current position in route
                const currentIdx = route.indexOf(train.segment_id);
                trainRouteState.set(train.id, {
                    routeSegmentIds: route,
                    currentSegmentIndex: currentIdx >= 0 ? currentIdx : 0,
                    progress: train.progress || 0,
                    dwellTimeRemaining: 0
                });
            }
        }

        const routeState = trainRouteState.get(train.id);

        // Check if train is dwelling at station
        if (routeState.dwellTimeRemaining > 0) {
            routeState.dwellTimeRemaining--;
            return {
                ...train,
                status: 'dwelling', // Show train is at station
                progress: routeState.progress
            };
        }

        // Calculate movement
        const speedFactor = (train.speed || 80) / 100;
        const increment = 0.01 * speedFactor; // Slower, more realistic
        let newProgress = routeState.progress + increment;

        // ✅ FIX: Instead of looping, advance to next segment
        if (newProgress >= 1.0) {
            // Reached end of current segment - advance to next station
            routeState.currentSegmentIndex++;

            // If at end of route, loop back to start (for circular routes like Main Line Ring)
            if (routeState.currentSegmentIndex >= routeState.routeSegmentIds.length) {
                routeState.currentSegmentIndex = 0;
            }

            // Start station dwell (3-5 animation frames = ~1.5-2.5 seconds)
            routeState.dwellTimeRemaining = Math.floor(Math.random() * 3) + 3;
            routeState.progress = 0;
            newProgress = 0;
        } else {
            routeState.progress = newProgress;
        }

        return {
            ...train,
            segment_id: routeState.routeSegmentIds[routeState.currentSegmentIndex],
            progress: newProgress,
            status: 'moving',
            direction: 'forward' // Could be enhanced to detect reversals
        };
    });
}

/**
 * Start train animation loop (INTERNAL loop, optional usage)
 */
function startTrainAnimation(getTrainsCallback, segments, stations) {
    stopTrainAnimation(); // Clear any existing interval

    trainAnimationInterval = setInterval(() => {
        let trains = getTrainsCallback();
        if (!trains || trains.length === 0) return;

        // Simulate movement
        trains = simulateTrainMovement(trains);

        // Update state with new positions
        if (window.mockData && window.mockData.liveStatus) {
            window.mockData.liveStatus.active_trains = trains;
        }

        // Animate to new positions
        updateTrainPositions(trains, segments, stations);
    }, 500);

    console.log('Train animation started');
}

/**
 * Stop train animation loop
 */
function stopTrainAnimation() {
    if (trainAnimationInterval) {
        clearInterval(trainAnimationInterval);
        trainAnimationInterval = null;
        console.log('Train animation stopped');
    }
}

/**
 * Get color based on train status and delay severity
 */
function getTrainColor(status, delay = 0) {
    // Delay severity colors
    if (status === 'delayed') {
        if (delay > 15) return '#dc2626';  // Critical - darker red
        if (delay > 8) return '#ef4444';   // Major - red
        if (delay > 3) return '#fb923c';   // Minor - orange
        return '#ef4444';  // Default delayed
    }

    const colors = {
        moving: '#3b82f6',
        stopped: '#f59e0b',
        delayed: '#ef4444',
        recovering: '#10b981',
        dwelling: '#3b82f6'
    };
    return colors[status] || colors.moving;
}

// ============================================================================
// ZOOM CONTROLS
// ============================================================================

/**
 * Zoom in programmatically
 */
function zoomIn() {
    if (!networkSvg || !networkSvg.zoomBehavior) return;

    networkSvg.transition()
        .duration(350)
        .call(networkSvg.zoomBehavior.scaleBy, 1.3);
}

/**
 * Zoom out programmatically
 */
function zoomOut() {
    if (!networkSvg || !networkSvg.zoomBehavior) return;

    networkSvg.transition()
        .duration(350)
        .call(networkSvg.zoomBehavior.scaleBy, 0.77);
}

/**
 * Reset zoom to default view
 */
function zoomReset() {
    if (!networkSvg || !networkSvg.zoomBehavior) return;

    networkSvg.transition()
        .duration(500)
        .call(networkSvg.zoomBehavior.transform, d3.zoomIdentity);
}

/**
 * Zoom to incident epicenter
 */
function zoomToIncident() {
    if (!networkSvg || !networkSvg.zoomBehavior) return;

    // Get current incident from global state
    const incident = window.appState && window.appState.analysisResult;
    if (!incident || !incident.parsed || !incident.parsed.station_ids || incident.parsed.station_ids.length === 0) {
        console.log('No incident location to zoom to');
        return;
    }

    const affectedStationIds = incident.parsed.station_ids;
    const stations = window.appState.stations || window.staticNetworkData?.stations || [];

    // Find affected stations
    const affectedStations = stations.filter(s => affectedStationIds.includes(s.id));

    if (affectedStations.length === 0) {
        console.log('No valid stations found for incident');
        return;
    }

    // Calculate center of affected area
    const avgX = affectedStations.reduce((sum, s) => sum + s.coordinates[0], 0) / affectedStations.length;
    const avgY = affectedStations.reduce((sum, s) => sum + s.coordinates[1], 0) / affectedStations.length;

    // Convert to SVG coordinates
    const svgX = xScale(avgX);
    const svgY = yScale(avgY);

    // Calculate zoom transform to center on this point
    const width = networkSvg.attr('width') || 800;
    const height = networkSvg.attr('height') || 600;
    const scale = 2.5; // Zoom in 2.5x

    const transform = d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(scale)
        .translate(-svgX, -svgY);

    // Smoothly transition to the incident location
    networkSvg.transition()
        .duration(750)
        .call(networkSvg.zoomBehavior.transform, transform);
}


//Export functions
window.networkView = {
    init: initNetworkView,
    render: renderNetwork,
    highlightNodes: highlightAffectedNodes,
    animateCascade: animateCascadeWave,
    animateRecovery: animateRecovery,
    updateStatus: updateNetworkStatus,
    renderTrains: renderTrains,
    updateTrains: updateTrainPositions,
    simulateMovement: simulateTrainMovement, // Exposed for app.js
    startTrainAnimation: startTrainAnimation,
    stopTrainAnimation: stopTrainAnimation,
    zoomIn: zoomIn,
    zoomOut: zoomOut,
    zoomReset: zoomReset,
    zoomToIncident: zoomToIncident,
    clearHighlights: () => {
        if (stationNodes) stationNodes.classed('station-affected station-recovering', false);
    }
};
