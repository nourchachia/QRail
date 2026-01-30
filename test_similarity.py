"""
Test if similarity scores improved after re-uploading embeddings
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.backend.integration import IncidentPipeline

# Initialize pipeline
pipeline = IncidentPipeline()

# Test with incident that should match INC_003
test_text = "Signal failure at South Junction affecting 3 trains"

print("="*70)
print("TESTING SIMILARITY SCORES AFTER UPLOADER RE-RUN")
print("="*70)
print(f"\nTest incident: {test_text}")
print("\nExpected match: INC_003 (signal_failure at South Junction)")
print("Expected similarity: 75-90%")
print("\n" + "="*70 + "\n")

# Analyze
result = pipeline.process(test_text)

# Extract key metrics
print("\nRESULTS:")
print("-"*70)

if result.get('similar_incidents'):
    top_3 = result['similar_incidents'][:3]
    
    print("\nTop 3 Similar Incidents:")
    for i, inc in enumerate(top_3, 1):
        print(f"\n  {i}. {inc.get('incident_id', 'Unknown')}")
        print(f"     Similarity: {inc.get('score', 0):.1%}")
        print(f"     Golden Run: {'YES' if inc.get('is_golden') else 'No'}")
        
    # Check if top match is good
    top_score = top_3[0].get('score', 0)
    
    print("\n" + "="*70)
    if top_score > 0.7:
        print("SUCCESS: Similarity > 70% - Embeddings are fresh!")
        print("Resolutions should now show DIFFERENT delays")
    elif top_score > 0.5:
        print("PARTIAL: Similarity 50-70% - Better but could be improved")
    else:
        print(f"ISSUE: Similarity only {top_score:.1%} - Still too low")
        print("Embeddings may still be stale or incident doesn't match well")
    print("="*70)

if result.get('recommendations'):
    print("\n\nRecommended Resolutions:")
    print("-"*70)
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"\n  {i}. {rec.get('strategy', 'Unknown')}")
        print(f"     Confidence: {rec.get('confidence', 0):.1%}")
        
    # Check variance
    confidences = [r.get('confidence', 0) for r in result['recommendations'][:3]]
    if confidences:
        conf_range = max(confidences) - min(confidences)
        print(f"\n  Confidence Range: {conf_range:.1%}")
        if conf_range > 0.15:
            print("  Status: GOOD variance - delays should differ significantly")
        else:
            print("  Status: LOW variance - delays may still look similar")

print("\n" + "="*70)
