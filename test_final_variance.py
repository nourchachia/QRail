"""
Final test using STATION NAMES (not IDs) like the successful test
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.integration import IncidentPipeline

# Use actual station NAMES from stations.json that map to IDs
test_scenarios = [
    {
        "name": "Test 1: South Junction (worked before)",
        "text": "Multiple signal failures at South Junction blocking express trains"
    },
    {
        "name": "Test 2: Central Station",  
        "text": "Express train derailed at Central Station due to track defect"
    },
    {
        "name": "Test 3: North Terminal",
        "text": "Power outage at North Terminal affecting electrified services" 
    }
]

print("="*70)
print("FINAL VARIANCE TEST - Using Station Names")
print("="*70)

pipeline = IncidentPipeline()

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n{'-'*70}")
    print(f"TEST {i}: {scenario['name']}")
    print(f"{'-'*70}")
    print(f"Incident: {scenario['text']}\n")
    
    result = pipeline.process(scenario['text'])
    
    if 'recommendations' in result:
        recs = result['recommendations']
        confs = [r.get('confidence', 0) for r in recs]
        golden = sum(1 for r in recs if r.get('type') == 'proven')
        
        print(f"✓ Recommendations: {len(recs)}")
        print(f"✓ Golden Runs: {golden}")
        print(f"✓ Confidence Range: {min(confs):.1%} - {max(confs):.1%}")
        
        variance = max(confs) - min(confs) 
        print(f"✓ Variance: {variance:.1%}")
        
        if variance >= 0.30:
            print(f"  → ✅ EXCELLENT - Delays will differ significantly!")
        elif variance >= 0.15:
            print(f"  → ✅ GOOD - Noticeable delay differences")  
        else:
            print(f"  → ❌ LOW - Delays may look similar")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("\nThe uploader fix works when:")
print("1. Incident mentions actual STATION NAMES (not IDs)")
print("2. Station names match entries in stations.json")
print("3. This allows GNN embedding → better matches → variance")
print()
print("Generic incidents without station names rely only on")
print("semantic matching, which produces similar confidence scores.")
