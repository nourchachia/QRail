"""
Test the actual API endpoint to see what's being returned
"""
import requests
import json

# Test with the same incident we used before
incident_text = "Signal failure at South Junction (STN_003) causing delays on the main line. Multiple trains affected including express services."

print("="*70)
print("TESTING ACTUAL API ENDPOINT")
print("="*70)
print(f"\nIncident: {incident_text}\n")

# Call the actual API
response = requests.post(
    "http://localhost:8002/api/analyze",
    json={"text": incident_text}
)

if response.status_code == 200:
    result = response.json()
    
    print("RECOMMENDATIONS RETURNED:")
    print("-"*70)
    
    if 'recommendations' in result:
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"\n{i}. {rec.get('strategy', 'Unknown')}")
            print(f"   Confidence: {rec.get('confidence', 0):.1%}")
            print(f"   Type: {rec.get('type', 'unknown')}")
            print(f"   Score: {rec.get('score', 0):.3f}")
    else:
        print("NO RECOMMENDATIONS in response!")
        
    print("\n" + "="*70)
    print(f"Total recommendations: {len(result.get('recommendations', []))}")
    
    # Check confidence variance
    confidences = [r.get('confidence', 0) for r in result.get('recommendations', [])]
    if confidences:
        conf_range = max(confidences) - min(confidences)
        print(f"Confidence range: {conf_range:.1%}")
        if conf_range > 0.15:
            print("✅ GOOD variance - should show different delays")
        else:
            print("❌ LOW variance - delays will look similar")
else:
    print(f"ERROR: API returned {response.status_code}")
    print(response.text)
