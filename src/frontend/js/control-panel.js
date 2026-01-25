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
    await analyzeIncident(scenario.text);
}

async function analyzeIncident(text) {
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
            result = window.mockData.analysisResult;
        } else {
            result = await window.api.analyzeIncident(text);
        }

        // Update state to analyzing
        window.appState.setState({
            status: 'analyzing',
            analysisResult: result,
        });

        showSearchStatus('analyzing');
        await sleep(1000);

        // Display results
        displaySimilarCases(result.similar_incidents);
        displayResolutionOptions(result.recommendations);

        // Highlight affected nodes if available
        if (result.parsed && result.parsed.station_ids) {
            window.networkView.highlightNodes(result.parsed.station_ids);
            window.networkView.animateCascade(result.parsed.station_ids, window.appState.stations);
        }

        hideSearchStatus();

    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Failed to analyze incident. Check that the backend is running on http://localhost:8000');
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
    const labels = {
        topology_match: 'Network Topology',
        cascade_pattern: 'Cascade Pattern',
        semantic_similarity: 'Context Match',
    };

    let html = '<div class="similarity-breakdown">';
    html += '<div class="breakdown-label">Match Breakdown:</div>';

    for (const [key, value] of Object.entries(explanation)) {
        const label = labels[key] || key;
        const percent = Math.round(value * 100);
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

    card.innerHTML = `
    <div class="resolution-header">
      <span class="resolution-type">${option.type}</span>
      <span class="confidence-badge">${confidence}% confident</span>
    </div>
    <h4 class="resolution-title">${option.strategy.replace(/_/g, ' ')}</h4>
    ${option.description ? `<p style="color: #94a3b8; margin-bottom: 1rem;">${option.description}</p>` : ''}
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
        incident_id: window.appState.currentIncident?.text.substring(0, 50),
        resolution_id: window.appState.selectedResolution?.strategy,
        operator_rating: rating,
        execution_success: rating >= 4,
        notes: notes || undefined,
    };

    try {
        if (!window.appState.demoMode) {
            await window.api.submitFeedback(feedback);
        }

        alert('✅ Feedback submitted! The AI will learn from this resolution.');

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

// Export functions
window.controlPanel = {
    init: initControlPanel,
};
