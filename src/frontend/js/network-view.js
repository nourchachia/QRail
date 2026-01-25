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

//Export functions
window.networkView = {
    init: initNetworkView,
    render: renderNetwork,
    highlightNodes: highlightAffectedNodes,
    animateCascade: animateCascadeWave,
    animateRecovery: animateRecovery,
    updateStatus: updateNetworkStatus,
};
