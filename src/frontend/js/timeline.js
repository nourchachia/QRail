/**
 * Timeline Component - Charts and comparison view
 * 
 * Displays:
 * - Telemetry history (30 minutes)
 * - Metrics summary (weather, load, trains)
 * - Before/After comparison (time savings)
 */

let telemetryChart, comparisonChart;

function initTimeline() {
    initTelemetryChart();
    initComparisonChart();
    console.log('Timeline initialized');
}

function initTelemetryChart() {
    const ctx = document.getElementById('telemetry-chart');
    if (!ctx) return;

    // Generate mock telemetry data (30 minutes, 10 data points)
    const telemetryData = generateMockTelemetry(10);

    telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: telemetryData.map(d => d.t),
            datasets: [{
                label: 'Network Load (%)',
                data: telemetryData.map(d => d.load),
                borderColor: '#3b82f6', // Blue for load
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: { color: '#94a3b8', boxWidth: 10 }
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#e2e8f0',
                    bodyColor: '#cbd5e1',
                    borderColor: '#334155',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    min: 0,
                    max: 100, // Fixed percentage scale
                    title: { display: true, text: 'Load (%)', color: '#64748b' },
                    ticks: { color: '#64748b', stepSize: 20 },
                    grid: { color: '#334155' }
                },
                x: {
                    ticks: { color: '#64748b', maxTicksLimit: 6 },
                    grid: { display: false }
                }
            }
        }
    });
    // Set fixed height for container
    ctx.parentNode.style.height = '180px';
}

function initComparisonChart() {
    const ctx = document.getElementById('comparison-chart');
    if (!ctx) return;

    comparisonChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Without AI',
                    data: [],
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false,
                },
                {
                    label: 'With AI Resolution',
                    data: [],
                    borderColor: '#10b981',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Total Delay (min)',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#64748b' },
                    grid: { color: '#334155' }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time (minutes)',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#64748b' },
                    grid: { color: '#334155' }
                }
            }
        }
    });
}

function updateMetricsSummary(liveStatus) {
    if (!liveStatus) return;

    const weather = liveStatus.weather || {};
    const condition = weather.condition || 'Clear';
    document.getElementById('weather-value').textContent =
        condition.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    const loadVal = liveStatus.network_load_pct !== undefined ? liveStatus.network_load_pct : 0;
    document.getElementById('load-value').textContent = `${loadVal}%`;

    // Update Chart if it exists
    if (window.timeline && window.timeline.updateLoadChart) {
        window.timeline.updateLoadChart(loadVal);
    }

    // Handle both array (from API) and number (from simulation) formats
    const trainCount = Array.isArray(liveStatus.active_trains)
        ? liveStatus.active_trains.length
        : (typeof liveStatus.active_trains === 'number' ? liveStatus.active_trains : 0);
    document.getElementById('trains-value').textContent = trainCount;

    document.getElementById('incidents-value').textContent =
        (liveStatus.active_incidents !== undefined) ? liveStatus.active_incidents : 0;
}

/**
 * Helper: Count how many trains are actually affected by an incident
 * Uses real simulation data instead of hardcoded values
 */
function countAffectedTrains(incident) {
    if (!incident || !incident.parsed) return 5; // Fallback

    const affectedStations = incident.parsed.station_ids || [];
    const affectedSegments = incident.parsed.segment_ids || [];

    // Get current train positions from network view
    if (window.networkView && window.networkView.getTrains) {
        const trains = window.networkView.getTrains();

        const affected = trains.filter(train => {
            const isStationAffected = affectedStations.includes(train.from_station) ||
                affectedStations.includes(train.to_station) ||
                affectedStations.includes(train.station_id);
            const isSegmentAffected = affectedSegments.includes(train.segment_id);
            return isStationAffected || isSegmentAffected;
        });

        return Math.max(1, affected.length); // At least 1
    }

    // Fallback: estimate based on network size
    const stationCount = affectedStations.length;
    return Math.max(3, Math.min(10, stationCount * 2));
}

function showComparisonView(incident, resolution) {
    const container = document.getElementById('comparison-view');
    if (!container) return;

    container.classList.remove('hidden');

    // Calculate scenarios using REAL simulation data
    const withoutAI = calculateCascadeScenario(incident, null);
    const withAI = calculateCascadeScenario(incident, resolution);

    const timeSaved = withoutAI.total_delay - withAI.total_delay;
    const improvement = withoutAI.total_delay > 0 ? (timeSaved / withoutAI.total_delay) * 100 : 0;
    const passengersSaved = withoutAI.passengers_affected - withAI.passengers_affected;

    // Update metrics
    document.getElementById('delay-without').textContent = `${withoutAI.total_delay} min`;
    document.getElementById('delay-with').textContent = `${withAI.total_delay} min`;

    document.getElementById('trains-without').textContent = `${withoutAI.trains_affected} trains affected`;
    document.getElementById('trains-with').textContent = `${withAI.trains_affected} trains affected`;

    document.getElementById('passengers-without').textContent =
        `${withoutAI.passengers_affected.toLocaleString()} passengers`;
    document.getElementById('passengers-with').textContent =
        `${withAI.passengers_affected.toLocaleString()} passengers`;

    // Update savings banner
    document.getElementById('time-saved').textContent = `${timeSaved} min`;
    document.getElementById('improvement-pct').textContent = `${improvement.toFixed(1)}%`;
    document.getElementById('passengers-saved').textContent = passengersSaved.toLocaleString();

    // Update comparison chart
    updateComparisonChart(withoutAI, withAI);
}

function hideComparisonView() {
    const container = document.getElementById('comparison-view');
    if (container) {
        container.classList.add('hidden');
    }
}

function calculateCascadeScenario(incident, resolution) {
    // FIX: Use actual train count instead of hardcoded 8
    const actualTrainsAffected = countAffectedTrains(incident);
    const passengersPerTrain = 300; // Average passengers per train

    // FIX: Try to get delay from actual simulation engine first
    let baseDelayMinutes;

    if (window.simulation && window.simulation.getIncidentProgression) {
        const progression = window.simulation.getIncidentProgression();
        baseDelayMinutes = progression.delayMinutes || 15;
    } else {
        // Fallback: use severity-based estimate
        const severityDelays = {
            low: 8,
            medium: 15,
            high: 25,
            critical: 35,
        };
        baseDelayMinutes = severityDelays[incident?.severity || 'medium'] || 15;
    }

    // Apply multipliers for context
    const cascadeMultiplier = incident?.location?.is_junction ? 1.5 : 1.0;
    const weatherPenalty = incident?.weather === 'heavy_rain' ? 1.3 : 1.0;

    const delayPerTrain = baseDelayMinutes * cascadeMultiplier * weatherPenalty;
    const totalDelay = delayPerTrain * actualTrainsAffected;

    if (!resolution) {
        // Without AI: full delay impact
        return {
            total_delay: Math.round(totalDelay),
            trains_affected: actualTrainsAffected,
            passengers_affected: actualTrainsAffected * passengersPerTrain,
            recovery_time: Math.round(delayPerTrain * 3), // Recovery takes 3x the delay
        };
    }

    // FIX: With AI - effectiveness varies the reduction significantly
    const effectiveness = resolution.expected_outcome || 0.65;

    // Better resolutions reduce MORE delay and affect FEWER trains
    const delayReduction = effectiveness; // 0.5 to 0.95 range
    const trainReduction = effectiveness * 0.7; // Slightly less reduction in train count

    const reducedTrains = Math.max(1, Math.ceil(actualTrainsAffected * (1 - trainReduction)));
    const reducedDelay = totalDelay * (1 - delayReduction);

    return {
        total_delay: Math.round(reducedDelay),
        trains_affected: reducedTrains,
        passengers_affected: reducedTrains * passengersPerTrain,
        recovery_time: Math.round(delayPerTrain * (1 - effectiveness) * 2),
    };
}

function updateComparisonChart(withoutAI, withAI) {
    if (!comparisonChart) return;

    // Generate chart data (60 minutes, every 5 minutes)
    const data = [];
    for (let minute = 0; minute <= 60; minute += 5) {
        const t = minute;

        // Without AI: peaks at 20 min, slow recovery
        let delayWithout;
        if (t <= 20) {
            delayWithout = (t / 20) * withoutAI.total_delay;
        } else {
            delayWithout = withoutAI.total_delay * Math.exp(-0.03 * (t - 20));
        }

        // With AI: intervention at 15 min, fast recovery
        let delayWith;
        if (t <= 15) {
            delayWith = (t / 15) * withAI.total_delay;
        } else {
            delayWith = withAI.total_delay * Math.exp(-0.08 * (t - 15));
        }

        data.push({
            minute: t,
            delay_without_ai: Math.round(delayWithout),
            delay_with_ai: Math.round(delayWith),
        });
    }

    comparisonChart.data.labels = data.map(d => d.minute);
    comparisonChart.data.datasets[0].data = data.map(d => d.delay_without_ai);
    comparisonChart.data.datasets[1].data = data.map(d => d.delay_with_ai);
    comparisonChart.update();
}

function generateMockTelemetry(points) {
    const data = [];
    for (let i = 0; i < points; i++) {
        data.push({
            t: `-${(points - i) * 3}m`,
            load: Math.floor(Math.random() * 20) + 70, // Random load 70-90%
        });
    }
    return data;
}

// Export functions
window.timeline = {
    init: initTimeline,
    updateMetrics: updateMetricsSummary,
    showComparison: showComparisonView,
    hideComparison: hideComparisonView,
};

// ============================================================================
// Real-Time Telemetry Updates
// ============================================================================

let networkLoadHistory = Array(30).fill(0);  // Rolling 30-minute window
let telemetryUpdateInterval = null;

/**
 * Start polling network status every 30 seconds
 */
function startTelemetryPolling() {
    if (telemetryUpdateInterval) {
        clearInterval(telemetryUpdateInterval);
    }

    fetchNetworkTelemetry();

    telemetryUpdateInterval = setInterval(async () => {
        await fetchNetworkTelemetry();
    }, 30000);  // 30 seconds

    console.log('✅ Telemetry polling started (30s interval)');
}

async function fetchNetworkTelemetry() {
    try {
        const status = await window.api.getLiveStatus();
        if (!status) return;

        // Update chart
        if (status.network_load_pct !== undefined) {
            updateNetworkLoadChart(status.network_load_pct);
        }

        // Update summary metrics (the dashboard boxes)
        updateMetricsSummary(status);

        // Also sync with global appState if needed
        window.appState.setState({ liveStatus: status });

        // Update top bar in network view
        if (window.networkView && window.networkView.updateStatus) {
            window.networkView.updateStatus(status);
        }
    } catch (error) {
        console.warn('Telemetry fetch failed:', error);
    }
}

function updateNetworkLoadChart(newLoad) {
    networkLoadHistory.shift();
    networkLoadHistory.push(newLoad);

    if (telemetryChart) {
        telemetryChart.data.labels = networkLoadHistory.map((_, i) => `-${30 - i}m`);
        telemetryChart.data.datasets[0].data = networkLoadHistory;
        telemetryChart.update('none');
    }
}

function stopTelemetryPolling() {
    if (telemetryUpdateInterval) {
        clearInterval(telemetryUpdateInterval);
        telemetryUpdateInterval = null;
        console.log('Telemetry polling stopped');
    }
}

window.timeline.startTelemetry = startTelemetryPolling;
window.timeline.stopTelemetry = stopTelemetryPolling;
