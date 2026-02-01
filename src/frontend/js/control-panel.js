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
        btn.addEventListener('click', () => {
            const speed = parseInt(btn.dataset.speed);
            if (window.simulation) {
                window.simulation.setSpeed(speed);
            }
        });
    });

    // Reset Button
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            // Reset state
            window.appState.reset();

            // Clear search result view
            hideResults();

            // Clear incident input
            const input = document.getElementById('incident-text');
            if (input) input.value = '';

            // Reset simulation incident state
            if (window.simulation) {
                window.simulation.clearIncident();
                // Ensure simulation keeps running
                if (!window.simulation.getState().isRunning) {
                    window.simulation.start();
                }
            }

            // Zoom reset
            if (window.networkView) {
                window.networkView.highlightNodes([], []); // Clear highlights
                // We don't necessarily want to zoom out fully, just clear the "incident focus"
                // window.networkView.zoomReset(); 
            }

            console.log('🔄 Application reset');
        });
    }

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

    // Add event listeners for Apply buttons
    // NOTE: These buttons are dynamically created, so listeners should ideally be attached
    // after they are created (e.g., in displayResolutionOptions).
    // For now, this will only attach to buttons present on initial load, which is none.
    // A more robust solution would be event delegation or attaching listeners when buttons are created.
    document.querySelectorAll('.apply-button').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const strategy = e.target.dataset.optionId;
            selectResolution(strategy);
        });
    });

    // Add event listeners for Compare buttons
    // NOTE: These buttons are dynamically created, so listeners should ideally be attached
    // after they are created (e.g., in displayResolutionOptions).
    // For now, this will only attach to buttons present on initial load, which is none.
    // A more robust solution would be event delegation or attaching listeners when buttons are created.
    document.querySelectorAll('.compare-button').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const strategy = e.target.dataset.optionId;  // Currently unused by activate, but good for context
            console.log('🔮 Activating comparison view...');

            // Get current analysis result
            const result = window.appState.analysisResult;
            if (window.futureComparison && result && result.recommendations) {
                window.futureComparison.activate(result, result.recommendations);
            } else {
                console.error('Compare failed: Missing data or module', {
                    mod: !!window.futureComparison,
                    res: !!result
                });
                alert('⚠️ Comparison view unavailable (check console)');
            }
        });
    });

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
        let result = await window.api.analyzeIncident(text);

        window.appState.setState({
            status: 'analyzing',
            analysisResult: result,
            activeIncidents: 1,  // Increment incident counter
        });

        // 🛡️ AUTHENTICITY CHECK: Prove to user we heavily rely on real data
        // 🛡️ AUTHENTICITY CHECK: Prove to user we heavily rely on real data
        // NOTE: Moved to backend terminal logs per user request (integration.py)
        if (result.truth_attribution) {
            // Logs silenced for cleaner UI
        }

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

        // Display Model 4 conflict predictions
        if (result.conflicts && Object.keys(result.conflicts).length > 0) {
            if (window.displayConflictTypes) {
                window.displayConflictTypes(result.conflicts);
            } else {
                console.warn('⚠️ displayConflictTypes not loaded yet');
            }
        }

        displayResolutionOptions(result.recommendations || []);

        // Show Anomaly Warning if detected
        console.log("🦢 Anomaly Check:", result.anomaly);
        if (result.anomaly && result.anomaly.is_anomaly) {
            console.log("⚠️ SHOWING ANOMALY WARNING");
            showAnomalyWarning(result.anomaly);
        } else {
            console.log("✓ Hiding anomaly warning");
            hideAnomalyWarning();
        }

        // Highlight affected nodes if available
        if (window.networkView && result.parsed) {
            window.networkView.highlightNodes(result.parsed.station_ids, result.parsed.segment_ids);
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

    // Score is already 0-1, convert to percentage
    const score = Math.min(100, Math.max(0, Math.round(incident.score * 100)));

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
    // Handle both nested and flat explanation structures, and safe parsing
    let breakdown = {};

    if (explanation && typeof explanation === 'object') {
        breakdown = explanation.similarity_breakdown || explanation;
    } else if (typeof explanation === 'string') {
        // Fallback for string explanations (should be rare with backend fix)
        return `<div class="similarity-explanation">${explanation}</div>`;
    }

    const labels = {
        topology: 'Network Topology',
        topology_match: 'Network Topology',
        cascade: 'Cascade Pattern',
        cascade_pattern: 'Cascade Pattern',
        context: 'Context Match',
        semantic_similarity: 'Context Match',
        semantic: 'Semantic Meaning',
        structural: 'Structural Layout',
        temporal: 'Temporal/Speed'
    };

    let html = '<div class="similarity-breakdown">';
    html += '<div class="breakdown-label">Match Breakdown:</div>';

    let hasData = false;
    for (const [key, value] of Object.entries(breakdown)) {
        // Filter out non-score keys just in case
        if (typeof value !== 'number') continue;

        hasData = true;
        const label = labels[key] || key.replace(/_/g, ' ');
        // value is 0-1, convert and clamp to 0-100
        const percent = Math.min(100, Math.max(0, Math.round(value * 100)));
        const color = value > 0.8 ? '#10b981' : value > 0.6 ? '#3b82f6' : '#f59e0b';

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

    if (!hasData) {
        html += '<div class="no-breakdown">Detailed metrics unavailable</div>';
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
    <div class="button-group" style="display: flex; gap: 8px; margin-top: 16px;">
      <button class="apply-button" data-option-id="${option.strategy}" style="flex: 1;">
        ▶ Apply This Resolution
      </button>
      <button class="compare-button" data-option-id="${option.strategy}" 
              style="flex: 0 0 auto; background: linear-gradient(135deg, #6366f1, #8b5cf6); 
                     color: white; border: none; padding: 12px 20px; border-radius: 8px; 
                     cursor: pointer; font-weight: 600; font-size: 0.875rem; transition: all 0.3s;
                     white-space: nowrap;">
        🔮 Compare
      </button>
    </div>
  `;

    // Add click handler for apply button
    card.querySelector('.apply-button').addEventListener('click', () => {
        handleApplyResolution(option);
    });

    // Add click handler for compare button
    card.querySelector('.compare-button').addEventListener('click', () => {
        // Trigger comparison view with this resolution and 2 alternatives
        if (window.appState.analysisResult && window.appState.analysisResult.recommendations) {
            const allResolutions = window.appState.analysisResult.recommendations;

            // Trigger simulation incident if not already triggered
            if (window.simulation && window.appState.currentIncident) {
                window.simulation.triggerIncident(window.appState.currentIncident);
            }

            // Activate future comparison
            window.futureComparison.activate(
                window.appState.currentIncident,
                allResolutions
            );
        }
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
        } else {
            // Mock success in demo mode
            await sleep(500);
        }

        showToast('✅ Feedback submitted! The AI will learn from this resolution.', 'success');

        // Disable form after success
        document.getElementById('submit-feedback-btn').disabled = true;
        document.getElementById('submit-feedback-btn').textContent = 'Submitted';

    } catch (error) {
        console.error('Feedback submission failed:', error);
        alert('⚠️ Feedback submission failed: ' + error.message);
    }
}

function hideResults() {
    document.getElementById('similar-cases').classList.add('hidden');
    document.getElementById('resolution-options').classList.add('hidden');
    document.getElementById('feedback-form').classList.add('hidden');

    // Hide conflicts section if it exists
    const conflictsSection = document.getElementById('conflicts-section');
    if (conflictsSection) {
        conflictsSection.classList.add('hidden');
    }

    hideAnomalyWarning();
    window.timeline.hideComparison();
}

/**
 * Show Black Swan Anomaly Warning
 * @param {Object} anomaly - Anomaly data from backend
 */
function showAnomalyWarning(anomaly) {
    const container = document.getElementById('anomaly-warning');
    if (!container) return;

    container.className = 'anomaly-warning modal-show'; // Modal style
    container.innerHTML = `
        <button class="anomaly-close-btn" onclick="hideAnomalyWarning()">✕</button>
        <div class="anomaly-icon">⚠️</div>
        <div class="anomaly-content">
            <h3>
                ⚡ UNPRECEDENTED INCIDENT DETECTED
                <span class="anomaly-score">Anomaly Score: ${Math.abs(anomaly.anomaly_score).toFixed(3)}</span>
            </h3>
            <p>
                This incident pattern deviates significantly from historical data. 
                The isolation forest algorithm has flagged this as anomalous.
            </p>
            <div class="anomaly-advice">
                🔍 This may represent a novel scenario not seen before in the training data. 
                Standard resolution strategies may not apply.
            </div>
            <div class="anomaly-actions">
                <button onclick="submitFeedback('valid', document.getElementById('incident-text').value)">
                    ✓ Confirm Anomaly
                </button>
                <button onclick="submitFeedback('invalid', document.getElementById('incident-text').value)">
                    ✗ Mark Invalid
                </button>
            </div>
        </div>
    `;
    container.classList.remove('hidden');
}

function hideAnomalyWarning() {
    const container = document.getElementById('anomaly-warning');
    if (container) {
        container.classList.add('hidden');
        container.innerHTML = ''; // Clean up
    }
}

/**
 * Submit feedback on anomaly detection accuracy
 * Called from anomaly warning buttons
 */
async function submitFeedback(feedbackType, incidentText) {
    console.log('📤 Submitting feedback:', feedbackType, 'for:', incidentText?.substring(0, 50));

    const feedback = {
        incident_id: (incidentText || "").substring(0, 100),
        resolution_id: "ANOMALY_" + feedbackType.toUpperCase(),
        operator_rating: feedbackType === 'valid' ? 5 : 1,
        execution_success: feedbackType === 'valid',
        notes: `Anomaly detection feedback: ${feedbackType === 'valid' ? 'Valid anomaly detection' : 'Invalid anomaly flag'}`,
    };

    console.log('📋 Feedback object:', JSON.stringify(feedback, null, 2));
    console.log('🌐 API Base URL:', window.api.base);

    try {
        const response = await window.api.submitFeedback(feedback);
        console.log('✅ Feedback response:', response);
        showToast('Anomaly feedback submitted! The AI will learn from this.', 'success');
        hideAnomalyWarning();
    } catch (error) {
        console.error('❌ Anomaly feedback failed:', error);
        console.error('Error details:', error.message, error.stack);
        // Show specific error type
        if (error.message.includes('API error')) {
            showToast(`API Error: ${error.message}. Is the backend running on ${window.api.base}?`, 'error');
        } else if (error.message.includes('Failed to fetch')) {
            showToast(`Connection Error: Cannot reach ${window.api.base}. Is the backend running?`, 'error');
        } else {
            showToast(`Failed to submit feedback: ${error.message}`, 'error');
        }
    }
}

// Make globally available for anomaly-feedback.js
window.hideAnomalyWarning = hideAnomalyWarning;

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
