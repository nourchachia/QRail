// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/'/g, '&#39;');
}

// Submit feedback for anomaly review
async function submitFeedback(action, incidentText) {
    try {
        const response = await fetch('/api/v1/incidents/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                incident_text: incidentText,
                action: action,
                timestamp: new Date().toISOString()
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Show confirmation
        const warning = document.getElementById('anomaly-warning');
        if (warning) {
            let message = '';
            if (action === 'valid') {
                message = '✓ Feedback recorded. Please create a solution for this incident.';
                warning.className = 'anomaly-warning-banner feedback-success';
            } else if (action === 'invalid') {
                message = '✓ Marked as invalid input. This pattern will be filtered in future.';
                warning.className = 'anomal y-warning-banner feedback-invalid';
            } else {
                message = '✓ Dismissed. No action taken.';
                warning.className = 'anomaly-warning-banner feedback-dismissed';
            }

            warning.innerHTML = `
                <div class="anomaly-header">
                    <div class="anomaly-icon">✓</div>
                    <div class="anomaly-content">
                        <h3>Feedback Submitted</h3>
                        <p>${message}</p>
                    </div>
                </div>
            `;

            // Auto-hide after 3 seconds
            setTimeout(() => {
                hideAnomalyWarning();
            }, 3000);
        }

    } catch (error) {
        console.error('Error submitting feedback:', error);
        alert('Failed to submit feedback. Please try again.');
    }
}
