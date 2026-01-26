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

function initNetworkView() {
    const container = document.getElementById('network-svg');
    const width = container.clientWidth;
    const height = container.clientHeight;

    networkSvg = d3.select('#network-svg')
        .attr('width', width)
        .attr('height', height);

    // Create scales for positioning
    xScale = d3.scaleLinear()
        .domain([-50, 150])
        .range([50, width - 50]);

    yScale = d3.scaleLinear()
        .domain([-50, 150])
        .range([50, height - 50]);

    console.log('Network view initialized');
}

function renderNetwork(stations, segments) {
    if (!networkSvg) initNetworkView();

    networkSvg.selectAll('*').remove();

    // Render segments first (so they're behind stations)
    renderSegments(segments, stations);

    // Render stations
    renderStations(stations);

    console.log(`Rendered ${stations.length} stations and ${segments.length} segments`);
}

function renderSegments(segments, stations) {
    const stationMap = new Map(stations.map(s => [s.id, s]));

    const lines = networkSvg.selectAll('.segment-line')
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
    const groups = networkSvg.selectAll('.station-node')
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

    // Station labels (only for major hubs)
    groups.filter(d => d.type === 'major_hub')
        .append('text')
        .attr('class', 'station-label')
        .attr('x', 0)
        .attr('y', -15)
        .attr('text-anchor', 'middle')
        .text(d => d.name);

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

function updateNetworkStatus(liveStatus) {
    if (!liveStatus) return;

    // Update status bar
    const weather = liveStatus.weather || {};
    document.getElementById('weather-status').textContent =
        `🌦️ Weather: ${weather.condition || 'Clear'}`;

    document.getElementById('load-status').textContent =
        `📊 Load: ${liveStatus.network_load_pct || 0}%`;

    const trainCount = liveStatus.active_trains ? liveStatus.active_trains.length : 0;
    document.getElementById('trains-status').textContent =
        `🚂 Trains: ${trainCount}`;
}

// ============================================================================
// Train Animation Functions
// ============================================================================

let trainMarkerGroup = null;
let trainAnimationInterval = null;

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
function getTrainRotation(segment, direction, stationMap) {
    const from = stationMap.get(segment.from_station);
    const to = stationMap.get(segment.to_station);

    if (!from || !to) return 0;

    const dx = to.coordinates[0] - from.coordinates[0];
    const dy = to.coordinates[1] - from.coordinates[1];
    let angle = Math.atan2(dy, dx) * (180 / Math.PI);

    // Reverse angle if going backward
    if (direction === 'backward') {
        angle += 180;
    }

    return angle;
}

/**
 * Get train color based on status
 */
function getTrainColor(status) {
    switch (status) {
        case 'moving': return '#3b82f6'; // Blue
        case 'stopped': return '#f59e0b'; // Orange
        case 'delayed': return '#ef4444'; // Red
        default: return '#3b82f6';
    }
}

/**
 * Render train markers on the network
 */
function renderTrains(trains, segments, stations) {
    if (!networkSvg) return;

    const stationMap = new Map(stations.map(s => [s.id, s]));
    const segmentMap = new Map(segments.map(s => [s.id, s]));

    // Remove existing train group
    networkSvg.selectAll('.train-group').remove();

    // Create train group layer (on top of everything)
    trainMarkerGroup = networkSvg.append('g').attr('class', 'train-group');

    // Filter trains - handle both segment-based and station-based positioning
    const validTrains = trains.filter(train => {
        if (train.segment_id && segmentMap.has(train.segment_id)) return true;
        if (train.station_id && stationMap.has(train.station_id)) return true;
        return false;
    });

    // Create train markers
    const trainGroups = trainMarkerGroup.selectAll('.train-marker')
        .data(validTrains, d => d.id)
        .enter()
        .append('g')
        .attr('class', d => `train-marker train-${d.status}`)
        .attr('data-train-id', d => d.id)
        .attr('transform', d => {
            // Position based on segment or station
            if (d.segment_id && segmentMap.has(d.segment_id)) {
                const segment = segmentMap.get(d.segment_id);
                const pos = getPositionOnSegment(segment, d.progress || 0.5, stationMap);
                const rotation = getTrainRotation(segment, d.direction || 'forward', stationMap);
                return pos ? `translate(${pos.x}, ${pos.y}) rotate(${rotation})` : 'translate(0, 0)';
            } else if (d.station_id && stationMap.has(d.station_id)) {
                // Train is at a station
                const station = stationMap.get(d.station_id);
                const x = xScale(station.coordinates[0]);
                const y = yScale(station.coordinates[1]) - 20; // Offset above station
                return `translate(${x}, ${y})`;
            }
            return 'translate(0, 0)';
        });

    // Train body (arrow/triangle shape pointing in direction)
    trainGroups.append('polygon')
        .attr('class', 'train-body')
        .attr('points', '8,0 -6,5 -6,-5')
        .attr('fill', d => d.color || getTrainColor(d.status))
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);

    // Pulsing effect for moving trains
    trainGroups.filter(d => d.status === 'moving')
        .append('circle')
        .attr('class', 'train-pulse')
        .attr('r', 12)
        .attr('fill', 'none')
        .attr('stroke', d => d.color || getTrainColor(d.status))
        .attr('stroke-width', 2)
        .attr('opacity', 0.5);

    // Add train ID label (hidden by default, shown on hover via CSS)
    trainGroups.append('text')
        .attr('class', 'train-label')
        .attr('x', 0)
        .attr('y', -15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '10px')
        .attr('font-weight', 'bold')
        .attr('pointer-events', 'none')
        .text(d => d.id);

    trainMarkers = trainGroups;
    console.log(`Rendered ${validTrains.length} trains`);
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
            const pos = getPositionOnSegment(segment, train.progress || 0.5, stationMap);
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

        trainEl
            .transition()
            .duration(900)
            .ease(d3.easeLinear)
            .attr('transform', transformStr);

        trainEl.select('.train-body')
            .attr('fill', train.color || getTrainColor(train.status));
    });
}

/**
 * Simulate train movement (for demo mode)
 */
function simulateTrainMovement(trains) {
    return trains.map(train => {
        if (train.status !== 'moving') return train;

        // Calculate progress increment based on speed
        const speedFactor = train.speed / 100; // Normalize speed
        const increment = 0.02 * speedFactor;

        let newProgress = train.progress + (train.direction === 'forward' ? increment : -increment);

        // Wrap around or reverse at segment ends
        if (newProgress >= 1) {
            newProgress = 0.99;
            train.direction = 'backward';
        } else if (newProgress <= 0) {
            newProgress = 0.01;
            train.direction = 'forward';
        }

        return { ...train, progress: newProgress };
    });
}

/**
 * Start train animation loop
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
    startTrainAnimation: startTrainAnimation,
    stopTrainAnimation: stopTrainAnimation,
};

