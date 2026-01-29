/**
 * Display detected conflict types with probabilities
 */
function displayConflictTypes(conflicts) {
    let conflictsSection = document.getElementById('conflicts-section');
    if (!conflictsSection) {
        const similarSection = document.getElementById('similar-cases');
        if (similarSection) {
            conflictsSection = document.createElement('div');
            conflictsSection.id = 'conflicts-section';
            conflictsSection.className = 'panel-section';
            conflictsSection.innerHTML = '<h3>⚠️ Detected Conflicts</h3><div id=\"conflicts-list\"></div>';
            similarSection.parentNode.insertBefore(conflictsSection, similarSection.nextSibling);
        }
    }

    const conflictsList = document.getElementById('conflicts-list');
    if (!conflictsList) return;

    const sortedConflicts = Object.entries(conflicts).sort(([, a], [, b]) => b - a).slice(0, 3);

    if (sortedConflicts.every(([, prob]) => prob < 0.3)) {
        conflictsList.innerHTML = '<div class=\"no-results\">✓ No significant conflicts</div>';
        return;
    }

    const conflictNames = {
        'headway_violation': 'Headway Violation',
        'platform_oversubscription': 'Platform Overload',
        'crew_timeout': 'Crew Timeout',
        'signal_blockage': 'Signal Blockage',
        'track_capacity': 'Track Capacity',
        'power_demand': 'Power Demand',
        'safety_margin': 'Safety Margin',
        'passenger_overflow': 'Passenger Overflow'
    };

    conflictsList.innerHTML = sortedConflicts.map(([type, prob]) => {
        const pct = (prob * 100).toFixed(0);
        const risk = prob > 0.7 ? 'high' : prob > 0.4 ? 'medium' : 'low';
        const color = risk === 'high' ? '239,68,68' : risk === 'medium' ? '245,158,11' : '34,197,94';
        return `
            <div class=\"conflict-item\" style=\"margin: 8px 0; padding: 10px; background: rgba(${color}, 0.1); border-left: 3px solid rgb(${color}); border-radius: 4px;\">
                <div style=\"display: flex; justify-content: space-between; margin-bottom: 6px;\">
                    <span style=\"font-weight: 500;\">${conflictNames[type] || type}</span>
                    <span style=\"font-weight: bold; color: rgb(${color});\">${pct}%</span>
                </div>
                <div style=\"height: 6px; background: rgba(100,116,139,0.2); border-radius: 3px; overflow: hidden;\">
                    <div style=\"width: ${pct}%; height: 100%; background: rgb(${color}); transition: width 0.5s ease;\"></div>
                </div>
            </div>
        `;
    }).join('');

    console.log(`✅ Displayed ${sortedConflicts.length} conflict types`);
}

// Add to global scope
window.displayConflictTypes = displayConflictTypes;
