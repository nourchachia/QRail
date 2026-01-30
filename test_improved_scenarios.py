"""
Improved multi-scenario test WITH specific station names
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.integration import IncidentPipeline

# IMPROVED test scenarios with SPECIFIC station names
test_scenarios = [
    {
        "name": "Signal Failure (STN_003)",
        "text": "Multiple signal failures at South Junction (STN_003) blocking express trains on main line"
    },
    {
        "name": "Derailment (STN_001-STN_002)",
        "text": "Express train EXP_001 derailed between Central Station (STN_001) and North Terminal (STN_002) due to track defect"
    },
    {
        "name": "Collision (STN_002-STN_003)",
        "text": "Regional train collision between North Terminal (STN_002) and South Junction (STN_003)"
    },
    {
        "name": "Track Obstruction (STN_004-STN_005)",
        "text": "Debris blocking track between Westgate Station (STN_004) and Regional Hub (STN_005)"
    },
    {
        "name": "Generic (no stations)",
        "text": "Power failure affecting electrified corridor"
    }
]

print("="*70)
print("IMPROVED MULTI-SCENARIO TEST (With Station Names)")
print("="*70)

pipeline = IncidentPipeline()
results_summary = []

for scenario in test_scenarios:
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"{'='*70}")
    print(f"{scenario['text']}\n")
    
    result = pipeline.process(scenario['text'])
    
    if 'recommendations' in result and result['recommendations']:
        recs = result['recommendations']
        confs = [r.get('confidence', 0) for r in recs]
        
        variance = max(confs) - min(confs)
        golden_count = sum(1 for r in recs if r.get('type') == 'proven')
        
        delay_diff = 35 * 0.5 * variance
        
        status = "✅ GOOD" if variance >= 0.15 else "❌ LOW"
        
        print(f"Confidence Range: {min(confs):.1%} - {max(confs):.1%}")
        print(f"Variance: {variance:.1%}")
        print(f"Delay Difference: {delay_diff:.1f} min")
        print(f"Golden Runs: {golden_count}")
        print(f"Status: {status}")
        
        results_summary.append({
            'name': scenario['name'],
            'variance': variance,
            'delay_diff': delay_diff,
            'golden': golden_count,
            'status': status
        })

print(f"\n\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}\n")
print(f"{'Scenario':<35} {'Variance':<12} {'Status'}")
print("-"*70)

for r in results_summary:
    print(f"{r['name']:<35} {r['variance']:>6.1%}      {r['status']}")

good_count = sum(1 for r in results_summary if r['variance'] >= 0.15)
print(f"\n✅ {good_count}/{len(results_summary)} scenarios with GOOD variance")
print(f"⭐ {sum(r['golden'] for r in results_summary)} golden runs detected total")

if good_count >= 3:
    print("\n🎉 FIX WORKING - Incidents with station names show variance!")
else:
    print("\n⚠️  Only incidents with specific station names show variance")
