"""
Multi-scenario backend test - testing variance across different incident types
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.integration import IncidentPipeline

# Test scenarios
test_scenarios = [
    {
        "name": "Signal Failure",
        "text": "Multiple signal failures on main line segment at South Junction blocking trains"
    },
    {
        "name": "Derailment",
        "text": "Express train EXP_001 derailed on main line due to track defect in rainy conditions"
    },
    {
        "name": "Collision",
        "text": "Regional train collision with stopped express train between stations"
    },
    {
        "name": "Track Obstruction",
        "text": "Debris blocking track service between Westgate Station and Regional Hub"
    },
    {
        "name": "Power Failure",
        "text": "Overhead power line failure affecting electrified corridor near junction"
    }
]

print("="*70)
print("MULTI-SCENARIO VARIANCE TEST")
print("="*70)
print("\nInitializing pipeline...")

# Initialize pipeline once
pipeline = IncidentPipeline()

print("✓ Pipeline ready!\n")

# Test each scenario
results_summary = []

for scenario in test_scenarios:
    print("\n" + "="*70)
    print(f"SCENARIO: {scenario['name']}")
    print("="*70)
    print(f"Incident: {scenario['text']}\n")
    
    # Process incident
    result = pipeline.process(scenario['text'])
    
    if 'recommendations' in result and result['recommendations']:
        recommendations = result['recommendations']
        confidences = [r.get('confidence', 0) for r in recommendations]
        
        min_conf = min(confidences)
        max_conf = max(confidences)
        variance = max_conf - min_conf
        
        # Count golden runs
        golden_count = sum(1 for r in recommendations if r.get('type') == 'proven')
        
        # Calculate delay difference
        base_delay = 35
        best_delay = base_delay * (1 - max_conf * 0.5)
        worst_delay = base_delay * (1 - min_conf * 0.5)
        delay_diff = worst_delay - best_delay
        
        print(f"Recommendations: {len(recommendations)}")
        print(f"Golden Runs: {golden_count}")
        print(f"\nConfidence Range:")
        print(f"  Highest: {max_conf:.1%}")
        print(f"  Lowest:  {min_conf:.1%}")
        print(f"  Variance: {variance:.1%}")
        print(f"\nDelay Impact:")
        print(f"  Best:  {best_delay:.1f} min")
        print(f"  Worst: {worst_delay:.1f} min")
        print(f"  Diff:  {delay_diff:.1f} min")
        
        # Verdict
        if variance >= 0.30:
            status = "✅ EXCELLENT"
        elif variance >= 0.15:
            status = "✅ GOOD"
        elif variance >= 0.05:
            status = "⚠️  MODERATE"
        else:
            status = "❌ LOW"
        
        print(f"\nStatus: {status}")
        
        # Save for summary
        results_summary.append({
            'name': scenario['name'],
            'variance': variance,
            'delay_diff': delay_diff,
            'golden': golden_count,
            'status': status
        })
    else:
        print("❌ No recommendations!")
        results_summary.append({
            'name': scenario['name'],
            'variance': 0,
            'delay_diff': 0,
            'golden': 0,
            'status': "❌ FAILED"
        })

# Overall summary
print("\n\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)
print()
print(f"{'Scenario':<20} {'Variance':<12} {'Delay Diff':<12} {'Status'}")
print("-"*70)

for result in results_summary:
    print(f"{result['name']:<20} {result['variance']:>6.1%}      {result['delay_diff']:>6.1f} min    {result['status']}")

print()
print("="*70)

# Final verdict
good_scenarios = sum(1 for r in results_summary if r['variance'] >= 0.15)
total_scenarios = len(results_summary)

print(f"\n✅ {good_scenarios}/{total_scenarios} scenarios show GOOD or EXCELLENT variance")
print(f"⭐ Golden runs detected in {sum(r['golden'] for r in results_summary)} scenarios")

if good_scenarios >= 3:
    print("\n🎉 BACKEND FIX CONFIRMED WORKING!")
else:
    print("\n⚠️  Some scenarios need improvement")

print()
