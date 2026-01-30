"""
Diagnostic: Check what the API is ACTUALLY returning for the exact UI text
"""
import requests
import json

incident_text = "Express train EXP_001 derailed at Central Station in rainy conditions"

print("="*70)
print("API DIAGNOSTIC - EXACT RESPONSE CHECK")
print("="*70)
print(f"\nTesting: {incident_text}")
print("\nCalling http://localhost:8002/api/analyze...")

try:
    response = requests.post(
        "http://localhost:8002/api/analyze",
        json={"text": incident_text},
        timeout=30
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print("\n" + "="*70)
        print("FULL API RESPONSE")
        print("="*70)
        print(json.dumps(data, indent=2))
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS SUMMARY")
        print("="*70)
        
        if 'recommendations' in data:
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"\n{i}. {rec.get('strategy', 'Unknown')}")
                print(f"   Confidence: {rec.get('confidence', 0):.1%}")
                print(f"   Type: {rec.get('type', 'unknown')}")
                print(f"   Similarity: {rec.get('score', 0):.1%}")
        
        # Check similar_incidents
        print("\n" + "="*70)
        print("SIMILAR INCIDENTS")
        print("="*70)
        
        if 'similar_incidents' in data:
            for i, inc in enumerate(data['similar_incidents'][:5], 1):
                print(f"\n{i}. {inc.get('incident_id', 'Unknown')}")
                print(f"   Similarity: {inc.get('score', 0):.1%}")
                print(f"   Type: {inc.get('accident_type', 'unknown')}")
                print(f"   Golden: {'YES' if inc.get('is_golden') else 'NO'}")
        
        # Summary
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        
        top_similarity = data.get('similar_incidents', [{}])[0].get('score', 0) if data.get('similar_incidents') else 0
        top_confidence = data.get('recommendations', [{}])[0].get('confidence', 0) if data.get('recommendations') else 0
        
        print(f"\nTop Similarity Score: {top_similarity:.1%}")
        print(f"Top Confidence Score: {top_confidence:.1%}")
        print(f"\nIf UI shows DIFFERENT values, there's a problem!")
        print("Expected in UI:")
        print(f"  - Top match: ~{top_similarity:.0%}")
        print(f"  - Top confidence: ~{top_confidence:.0%}")
        
    else:
        print(f"\nERROR: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Could not connect to API!")
    print("Make sure the server is running: python src/api/main.py")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
