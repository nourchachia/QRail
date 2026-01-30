"""
Comprehensive backend test showing resolution variance and delay impact
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.integration import IncidentPipeline

print("="*70)
print("BACKEND RESOLUTION VARIANCE TEST")
print("="*70)

# Initialize pipeline
pipeline = IncidentPipeline()

# Test incident that should match golden runs
test_incident = "Multiple signal failures on main line segment at South Junction blocking trains"

print(f"\nTest Incident: {test_incident}\n")
print("Analyzing...")

# Process incident
result = pipeline.process(test_incident)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

if 'recommendations' in result:
    recommendations = result['recommendations']
    
    print(f"\nTotal Recommendations: {len(recommendations)}")
    print("\nDetailed Breakdown:\n")
    
    confidences = []
    
    for i, rec in enumerate(recommendations, 1):
        confidence = rec.get('confidence', 0)
        confidences.append(confidence)
        rec_type = rec.get('type', 'unknown')
        strategy = rec.get('strategy', 'Unknown')
        score = rec.get('score', 0)
        
        print(f"{i}. {strategy}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Type: {rec_type}")
        print(f"   Similarity Score: {score:.3f}")
        
        # Calculate estimated delay based on confidence
        # Higher confidence = lower delay (better resolution)
        base_delay = 35  # minutes
        delay_reduction = confidence * 0.5  # Up to 50% reduction
        estimated_delay = base_delay * (1 - delay_reduction)
        
        print(f"   Estimated Delay: {estimated_delay:.1f} minutes")
        print()
    
    # Calculate variance
    if confidences:
        min_conf = min(confidences)
        max_conf = max(confidences)
        variance = max_conf - min_conf
        
        print("="*70)
        print("VARIANCE ANALYSIS")
        print("="*70)
        print(f"Highest Confidence: {max_conf:.1%}")
        print(f"Lowest Confidence:  {min_conf:.1%}")
        print(f"Confidence Range:   {variance:.1%}")
        print()
        
        # Delay variance
        best_delay = 35 * (1 - max_conf * 0.5)
        worst_delay = 35 * (1 - min_conf * 0.5)
        delay_variance = worst_delay - best_delay
        
        print(f"Best Resolution Delay:  {best_delay:.1f} min")
        print(f"Worst Resolution Delay: {worst_delay:.1f} min")
        print(f"Delay Difference:       {delay_variance:.1f} min")
        print()
        
        # Verdict
        if variance >= 0.30:
            print("✅ EXCELLENT variance - delays will be VERY different!")
        elif variance >= 0.15:
            print("✅ GOOD variance - delays will be noticeably different")
        elif variance >= 0.05:
            print("⚠️  MODERATE variance - delays somewhat different")
        else:
            print("❌ LOW variance - delays will look similar")
        
        print()
        print("="*70)
        
        # Golden run detection
        golden_count = sum(1 for r in recommendations if r.get('type') == 'proven')
        if golden_count > 0:
            print(f"⭐ {golden_count} Golden Run(s) detected!")
        else:
            print("ℹ️  No golden runs in top results")
            
else:
    print("❌ No recommendations returned!")

print("\n" + "="*70)
