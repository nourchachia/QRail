"""
Testing with golden run description INCLUDING location
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.integration import IncidentPipeline

# INC_001 with location info added (derailment at Central Station)
test_with_location = "Express train EXP_001 derailed on main line at Central Station in rainy conditions due to track defect"

print("="*70)
print("GOLDEN RUN TEST WITH LOCATION")
print("="*70)
print("\nBased on INC_001 from golden_runs_accidents_enhanced.json")
print(f"\nEnhanced incident text:")
print(f'"{test_with_location}"')
print("\nExpected: Should match INC_001 derailment with HIGH variance")
print("="*70 + "\n")

pipeline = IncidentPipeline()
result = pipeline.process(test_with_location)

if 'recommendations' in result:
    recs = result['recommendations']
    confs = [r.get('confidence', 0) for r in recs]
    golden_count = sum(1 for r in recs if r.get('type') == 'proven')
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70 + "\n")
    
    for i, rec in enumerate(recs, 1):
        icon = "⭐" if rec.get('type') == 'proven' else "  "
        print(f"{icon} {i}. {rec.get('strategy', 'Unknown')}")
        print(f"     Confidence: {rec.get('confidence', 0):.1%}")
        print(f"     Type: {rec.get('type', 'unknown')}")
        print()
    
    variance = max(confs) - min(confs)
    
    print("="*70)
    print(f"Golden Runs: {golden_count}")
    print(f"Confidence Range: {min(confs):.1%} - {max(confs):.1%}")
    print(f"Variance: {variance:.1%}")
    
    if variance >= 0.30:
        print("\n🎉 SUCCESS! EXCELLENT variance - fix is working!")
    elif variance >= 0.15:
        print("\n✅ GOOD variance - fix is working!")
    else:
        print("\n❌ LOW variance - station not extracted or no match")
    
print()
