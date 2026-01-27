/**
 * Control Panel Component - Incident input, AI results, feedback
 * 
 * Handles:
 * - Free-text incident input
 * - Quick scenario buttons
 * - Search status animation
 * - Similar cases display
 * - Resolution options
 * - Feedback submission
 */

function initControlPanel() {
    // Time Slider (replaces preset buttons)
    const timeSlider = document.getElementById('time-slider');
    const simClock = document.getElementById('sim-clock');

    if (timeSlider && simClock) {
        // Update map immediately as user drags the slider
        timeSlider.addEventListener('input', (e) => {
            const totalMinutes = parseInt(e.target.value);
            const hours = Math.floor(totalMinutes / 60);
            const minutes = totalMinutes % 60;
            const timeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;

            // Update clock display
            simClock.textContent = timeStr;

            if (window.simulation) {
                // Stop any running simulation
                window.simulation.stop();

                // Set time to selected value
                window.simulation.init(timeStr);

                // Show trains at this exact time (snapshot)
                window.simulation.updateTrains();
            }
        });
    }

    // Play/Pause Control - starts time flowing from current slider positions
    const playBtn = document.getElementById('play-pause-btn');
    if (playBtn) {
        playBtn.addEventListener('click', () => {
            if (window.simulation) {
                const state = window.simulation.getState();
                if (state.isRunning && !state.isPaused) {
                    window.simulation.pause();
                } else if (state.isPaused) {
                    window.simulation.resume();
                } else {
                    window.simulation.start();
                }
            }
        });
    }

    // Speed Controls
    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const speed = parseInt(e.currentTarget.dataset.speed);
            if (window.simulation) {
                window.simulation.setSpeed(speed);
            }
        });
    });

    // Day Type Selector
    document.querySelectorAll('.day-type-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const dayType = e.currentTarget.dataset.day;
            if (window.simulation) {
                window.simulation.setDayType(dayType);
            }
        });
    });

    // Time Preset Buttons (8 AM, 12 PM, 5 PM, 8 PM)
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const timeStr = e.currentTarget.dataset.time;
            if (window.simulation) {
                // Stop current simulation
                window.simulation.stop();

                // Jump to selected time
                window.simulation.init(timeStr);

                // Update slider position
                const timeSlider = document.getElementById('time-slider');
                if (timeSlider) {
                    const [hours, minutes] = timeStr.split(':').map(Number);
                    timeSlider.value = hours * 60 + minutes;
                }

                // Update clock display
                const simClock = document.getElementById('sim-clock');
                if (simClock) {
                    simClock.textContent = timeStr;
                }

                // Render trains at this time
                window.simulation.updateTrains();

                // Highlight active button
                document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');

                showToast(`Jumped to ${timeStr}`, 'info');
            }
        });
    });

    // Analyze button
    document.getElementById('analyze-btn').addEventListener('click', handleAnalyzeClick);

    // Quick scenario buttons
    document.querySelectorAll('.scenario-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const scenarioId = e.currentTarget.dataset.scenario;
            handleQuickScenario(scenarioId);
        });
    });

    // Feedback rating stars
    document.querySelectorAll('.star').forEach(star => {
        star.addEventListener('click', (e) => {
            const rating = parseInt(e.currentTarget.dataset.rating);
            setFeedbackRating(rating);
        });
    });

    // Submit feedback button
    document.getElementById('submit-feedback-btn').addEventListener('click', handleFeedbackSubmit);

    console.log('Control panel initialized');
}

async function handleAnalyzeClick() {
    const text = document.getElementById('incident-text').value.trim();
    if (!text) return;

    await analyzeIncident(text);
}

async function handleQuickScenario(scenarioId) {
    const scenario = window.scenarios[scenarioId];
    if (!scenario) return;

    document.getElementById('incident-text').value = scenario.text;
    await analyzeIncident(scenario.text, scenario);
}

async function analyzeIncident(text, scenario = null) {
    try {
        // Update state to searching
        window.appState.setState({
            status: 'detecting',
            currentIncident: { text },
        });

        // Show search status
        showSearchStatus('detecting');
        hideResults();

        // Simulate detection delay
        await sleep(1500);

        window.appState.setState({ status: 'searching' });
        showSearchStatus('searching');

        // Call API or use mock data
        let result;
        if (window.appState.demoMode) {
            await sleep(2000);
            result = JSON.parse(JSON.stringify(window.mockData.analysisResult));

            // If we have a scenario, use its metadata to override mock result
            if (scenario) {
                result.parsed.station_ids = scenario.location.station_ids || [];
                result.parsed.segment_ids = scenario.location.segment_ids || [];
            }
        } else {
            result = await window.api.analyzeIncident(text);
        }

        // Update state to analyzing
        window.appState.setState({
            status: 'analyzing',
            analysisResult: result,
            activeIncidents: 1,  // Increment incident counter
        });

        showSearchStatus('analyzing');
        await sleep(1000);

        // Update incidents display
        const incidentsCountEl = document.getElementById('incidents-value');
        if (incidentsCountEl) {
            incidentsCountEl.textContent = '1';
            incidentsCountEl.classList.add('pulse');
            setTimeout(() => incidentsCountEl.classList.remove('pulse'), 2000);
        }

        // Display results
        displaySimilarCases(result.similar_incidents || []);
        displayResolutionOptions(result.recommendations || []);

        // Highlight affected nodes if available
        if (result.parsed && result.parsed.station_ids) {
            window.networkView.highlightNodes(result.parsed.station_ids);
            window.networkView.animateCascade(result.parsed.station_ids, window.appState.stations);
        }

        hideSearchStatus();

    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Failed to analyze incident. Check that the backend is running on http://localhost:8002');
        window.appState.setState({ status: 'idle' });
        hideSearchStatus();
    }
}

function showSearchStatus(stage) {
    const container = document.getElementById('search-status');
    container.classList.remove('hidden');

    const steps = container.querySelectorAll('.step-item');
    const progressFill = container.querySelector('.progress-fill');
    const message = document.getElementById('status-message');

    // Update step states
    steps.forEach(step => {
        const stepName = step.dataset.step;
        step.classList.remove('active', 'done');

        if (stage === 'detecting' && stepName === 'topology') {
            step.classList.add('active');
            step.querySelector('.step-icon').textContent = '⏳';
        } else if (stage === 'searching' && stepName === 'cascade') {
            steps[0].classList.add('done');
            steps[0].querySelector('.step-icon').textContent = '✓';
            step.classList.add('active');
            step.querySelector('.step-icon').textContent = '⏳';
        } else if (stage === 'analyzing' && stepName === 'context') {
            steps[0].classList.add('done');
            steps[0].querySelector('.step-icon').textContent = '✓';
            steps[1].classList.add('done');
            steps[1].querySelector('.step-icon').textContent = '✓';
            step.classList.add('active');
            step.querySelector('.step-icon').textContent = '⏳';
        }
    });

    // Update progress bar
    const progress = {
        detecting: 33,
        searching: 66,
        analyzing: 100,
    }[stage] || 0;
    progressFill.style.width = `${progress}%`;

    // Update message
    const messages = {
        detecting: 'Encoding current situation...',
        searching: 'Querying Qdrant for similar cases...',
        analyzing: 'Generating recommendations...',
    };
    message.textContent = messages[stage] || '';
}

function hideSearchStatus() {
    const container = document.getElementById('search-status');
    container.classList.add('hidden');
}

function displaySimilarCases(cases) {
    const container = document.getElementById('similar-cases');
    container.innerHTML = '<h3>📚 Similar Historical Incidents</h3>';

    cases.forEach((incident, index) => {
        const card = createCaseCard(incident, index + 1);
        container.appendChild(card);
    });

    container.classList.remove('hidden');
}

function createCaseCard(incident, matchNumber) {
    const card = document.createElement('div');
    card.className = `case-card ${incident.is_golden ? 'golden' : ''}`;

    const score = Math.round(incident.score * 100);

    card.innerHTML = `
    <div class="case-header">
      <span class="match-number">Match #${matchNumber}</span>
      <span class="match-score">${score}%</span>
    </div>
    <div class="case-details">
      <p class="incident-type">Incident ${incident.incident_id}</p>
      <p class="case-meta">High similarity match</p>
    </div>
    ${incident.explanation ? createSimilarityBreakdown(incident.explanation) : ''}
    ${incident.is_golden ? '<div class="golden-badge">⭐ Golden Run - Verified Best Practice</div>' : ''}
  `;

    return card;
}

function createSimilarityBreakdown(explanation) {
    // Handle both nested and flat explanation structures
    const breakdown = explanation.similarity_breakdown || explanation;

    const labels = {
        topology: 'Network Topology',
        topology_match: 'Network Topology',
        cascade: 'Cascade Pattern',
        cascade_pattern: 'Cascade Pattern',
        context: 'Context Match',
        semantic_similarity: 'Context Match',
    };

    let html = '<div class="similarity-breakdown">';
    html += '<div class="breakdown-label">Match Breakdown:</div>';

    for (const [key, value] of Object.entries(breakdown)) {
        const label = labels[key] || key;
        const numValue = parseFloat(value);

        // Skip if not a number
        if (isNaN(numValue)) continue;

        const percent = Math.round(numValue * 100);
        const color = numValue > 0.8 ? '#10b981' : numValue > 0.6 ? '#3b82f6' : '#f59e0b';

        html += `
      <div class="sim-bar">
        <span class="sim-label">${label}</span>
        <div class="bar-container">
          <div class="bar-fill" style="width: ${percent}%; background-color: ${color};"></div>
        </div>
        <span class="sim-value">${percent}%</span>
      </div>
    `;
    }

    html += '</div>';
    return html;
}

function displayResolutionOptions(recommendations) {
    const container = document.getElementById('resolution-options');
    container.innerHTML = '<h3>💡 AI-Recommended Resolutions</h3>';

    recommendations.forEach(option => {
        const card = createResolutionCard(option);
        container.appendChild(card);
    });

    container.classList.remove('hidden');
}

function createResolutionCard(option) {
    const card = document.createElement('div');
    card.className = 'resolution-card';

    const confidence = Math.round(option.confidence * 100);
    const actions = option.actions || [];

    // Enhanced golden run data
    const hasEnhancedData = option.why_golden || option.actual_outcomes || option.lessons_learned;

    card.innerHTML = `
    <div class="resolution-header">
      <span class="resolution-type">${option.type}</span>
      <span class="confidence-badge">${confidence}% confident</span>
    </div>
    <h4 class="resolution-title">${option.strategy.replace(/_/g, ' ')}</h4>
    ${option.description ? `<p style="color: #94a3b8; margin-bottom: 1rem;">${option.description}</p>` : ''}
    
    ${option.why_golden ? `
      <div style="background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), transparent); 
                  border-left: 3px solid #f59e0b; padding: 12px; border-radius: 6px; margin-bottom: 16px;">
        <div style="color: #f59e0b; font-weight: 600; font-size: 0.75rem; margin-bottom: 4px;">⭐ WHY THIS IS GOLDEN</div>
        <div style="color: #e2e8f0; font-size: 0.875rem;">${option.why_golden}</div>
      </div>
    ` : ''}
    
    ${option.actual_outcomes ? `
      <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; 
                  padding: 12px; border-radius: 6px; margin-bottom: 16px;">
        <div style="color: #10b981; font-weight: 600; font-size: 0.75rem; margin-bottom: 8px;">📊 PROVEN RESULTS</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.875rem;">
          ${option.actual_outcomes.delay_reduction_pct ?
                `<div>
              <span style="color: #64748b;">Delay Reduction:</span>
              <span style="color: #10b981; font-weight: 600; margin-left: 4px;">${option.actual_outcomes.delay_reduction_pct}%</span>
            </div>` : ''}
          ${option.actual_outcomes.safety_score_improvement ?
                `<div>
              <span style="color: #64748b;">Safety Improvement:</span>
              <span style="color: #10b981; font-weight: 600; margin-left: 4px;">+${Math.round(option.actual_outcomes.safety_score_improvement * 100)}%</span>
            </div>` : ''}
          ${option.actual_outcomes.passenger_satisfaction ?
                `<div>
              <span style="color: #64748b;">Satisfaction:</span>
              <span style="color: #10b981; font-weight: 600; margin-left: 4px; text-transform: capitalize;">${option.actual_outcomes.passenger_satisfaction}</span>
            </div>` : ''}
          ${option.actual_outcomes.network_reliability_gain_pct ?
                `<div>
              <span style="color: #64748b;">Reliability Gain:</span>
              <span style="color: #10b981; font-weight: 600; margin-left: 4px;">+${option.actual_outcomes.network_reliability_gain_pct}%</span>
            </div>` : ''}
        </div>
      </div>
    ` : ''}
    
    ${option.lessons_learned && option.lessons_learned.length > 0 ? `
      <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; 
                  padding: 12px; border-radius: 6px; margin-bottom: 16px;">
        <div style="color: #3b82f6; font-weight: 600; font-size: 0.75rem; margin-bottom: 8px;">📝 OPERATOR LESSONS</div>
        ${option.lessons_learned.slice(0, 2).map(lesson => `
          <div style="color: #cbd5e1; font-size: 0.813rem; margin-bottom: 6px; padding-left: 12px; position: relative;">
            <span style="position: absolute; left: 0; color: #3b82f6;">▸</span>
            ${lesson}
          </div>
        `).join('')}
      </div>
    ` : ''}
    
    <div class="action-list">
      <div class="action-label">Actions:</div>
      ${actions.slice(0, 3).map(action => `
        <div class="action-item">
          <span class="action-bullet">•</span>
          <span>${action.action}${action.duration_minutes ? ` - ${action.duration_minutes} min` : ''}</span>
        </div>
      `).join('')}
      ${actions.length > 3 ? `<div class="action-item">+${actions.length - 3} more steps</div>` : ''}
    </div>
    <button class="apply-button" data-option-id="${option.strategy}">
      ▶ Apply This Resolution
    </button>
  `;

    // Add click handler for apply button
    card.querySelector('.apply-button').addEventListener('click', () => {
        handleApplyResolution(option);
    });

    return card;
}

function handleApplyResolution(option) {
    window.appState.setState({
        status: 'resolved',
        selectedResolution: option,
    });

    // Show comparison view
    window.timeline.showComparison(window.appState.currentIncident, option);

    // Animate recovery on network
    if (window.appState.analysisResult && window.appState.analysisResult.parsed.station_ids) {
        window.networkView.animateRecovery(
            window.appState.analysisResult.parsed.station_ids,
            window.appState.stations
        );
    }

    // Disable all apply buttons
    document.querySelectorAll('.apply-button').forEach(btn => {
        btn.classList.add('applied');
        btn.textContent = '✓ Applied';
        btn.disabled = true;
    });

    // Show feedback form
    showFeedbackForm();
}

function showFeedbackForm() {
    const container = document.getElementById('feedback-form');
    container.classList.remove('hidden');
}

function setFeedbackRating(rating) {
    window.appState.setState({ feedbackRating: rating });

    const stars = document.querySelectorAll('.star');
    stars.forEach((star, index) => {
        if (index < rating) {
            star.classList.add('active');
            star.textContent = '★';
        } else {
            star.classList.remove('active');
            star.textContent = '☆';
        }
    });

    const labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'];
    document.getElementById('rating-label').textContent = labels[rating - 1] || 'Click to rate';

    document.getElementById('submit-feedback-btn').disabled = false;
}

async function handleFeedbackSubmit() {
    const rating = window.appState.feedbackRating;
    const notes = document.getElementById('feedback-notes').value;

    const feedback = {
        incident_id: (window.appState.currentIncident?.text || "").substring(0, 50),
        resolution_id: window.appState.selectedResolution?.strategy,
        operator_rating: rating,
        execution_success: rating >= 4,
        notes: notes || undefined,
    };

    try {
        if (!window.appState.demoMode) {
            await window.api.submitFeedback(feedback);
        }

        showToast('✅ Feedback submitted! The AI will learn from this resolution.', 'success');

    } catch (error) {
        console.error('Feedback submission failed:', error);
        alert('⚠️ Feedback submission failed (endpoint not implemented yet)');
    }
}

function hideResults() {
    document.getElementById('similar-cases').classList.add('hidden');
    document.getElementById('resolution-options').classList.add('hidden');
    document.getElementById('feedback-form').classList.add('hidden');
    window.timeline.hideComparison();
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// TOAST NOTIFICATION SYSTEM
// ============================================================================

/**
 * Show a beautiful toast notification
 * @param {string} message - Message to display
 * @param {string} type - 'success', 'error', 'warning', 'info'
 */
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
        `;
        document.body.appendChild(container);
    }

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    // Icon based on type
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };

    // Colors based on type
    const colors = {
        success: { bg: 'rgba(16, 185, 129, 0.95)', border: '#10b981' },
        error: { bg: 'rgba(239, 68, 68, 0.95)', border: '#ef4444' },
        warning: { bg: 'rgba(245, 158, 11, 0.95)', border: '#f59e0b' },
        info: { bg: 'rgba(59, 130, 246, 0.95)', border: '#3b82f6' }
    };

    const color = colors[type] || colors.info;

    toast.style.cssText = `
        background: ${color.bg};
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        border-left: 4px solid ${color.border};
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), 0 0 20px ${color.border}40;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 14px;
        font-weight: 500;
        max-width: 400px;
        pointer-events: all;
        animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    `;

    toast.innerHTML = `
        <span style="font-size: 20px;">${icons[type] || icons.info}</span>
        <span>${message}</span>
    `;

    // Add animation keyframes if not already present
    if (!document.getElementById('toast-animations')) {
        const style = document.createElement('style');
        style.id = 'toast-animations';
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(400px);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(400px);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    container.appendChild(toast);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, 4000);
}

// Export functions
window.controlPanel = {
    init: initControlPanel,
};

window.showToast = showToast;
