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
                    labels: {
                        color: '#94a3b8'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMin: 0,
                    suggestedMax: 100, // Load percentage

                    ticks: { color: '#64748b' },
                    grid: { color: '#334155' }
                },
                x: {
                    ticks: { color: '#64748b' },
                    grid: { color: '#334155' }
                }
            }
        }
    });
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

    document.getElementById('load-value').textContent =
        `${liveStatus.network_load_pct || 0}%`;

    // Handle both array (from API) and number (from simulation) formats
    const trainCount = Array.isArray(liveStatus.active_trains) 
        ? liveStatus.active_trains.length 
        : (typeof liveStatus.active_trains === 'number' ? liveStatus.active_trains : 0);
    document.getElementById('trains-value').textContent = trainCount;

    document.getElementById('incidents-value').textContent =
        (liveStatus.active_incidents !== undefined) ? liveStatus.active_incidents : 0;
}

function showComparisonView(incident, resolution) {
    const container = document.getElementById('comparison-view');
    if (!container) return;

    container.classList.remove('hidden');

    // Calculate scenarios
    const withoutAI = calculateCascadeScenario(incident, null);
    const withAI = calculateCascadeScenario(incident, resolution);

    const timeSaved = withoutAI.total_delay - withAI.total_delay;
    const improvement = (timeSaved / withoutAI.total_delay) * 100;
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
    const severityDelays = {
        low: 60,
        medium: 120,
        high: 180,
        critical: 240,
    };

    const baseDelay = severityDelays[incident?.severity || 'medium'] || 120;
    const cascadeMultiplier = incident?.location?.is_junction ? 1.5 : 1.0;
    const weatherPenalty = incident?.weather === 'heavy_rain' ? 1.2 : 1.0;

    const naturalDelay = baseDelay * cascadeMultiplier * weatherPenalty;

    if (!resolution) {
        // Without AI: slower recovery
        return {
            total_delay: Math.round(naturalDelay),
            trains_affected: 8,
            passengers_affected: 8 * 300,
            recovery_time: 90,
        };
    }

    // With AI: faster recovery based on resolution effectiveness
    const effectiveness = resolution.expected_outcome || 0.65;
    const reduction = effectiveness;

    return {
        total_delay: Math.round(naturalDelay * (1 - reduction)),
        trains_affected: Math.ceil(8 * (1 - reduction)),
        passengers_affected: Math.ceil(8 * 300 * (1 - reduction)),
        recovery_time: Math.round(90 * (1 - reduction)),
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
