"""
Test Anomaly Detection System
------------------------------
Quick test script to run anomaly detection on various incidents.

Usage:
    python scripts/test_anomaly.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.backend.integration import IncidentPipeline

def test_anomaly_detection():
    """Test the anomaly detector with various incidents."""
    
    print("="*70)
    print("ANOMALY DETECTION TEST")
    print("="*70)
    
    # Initialize pipeline
    print("\n[1/3] Initializing pipeline...")
    pipeline = IncidentPipeline()
    
    if not pipeline.anomaly_detector:
        print("[ERROR] Anomaly detector not loaded!")
        print("Run: python src/models/train_anomaly_simple.py")
        return
    
    print("[SUCCESS] Anomaly detector ready")
    
    # Test cases
    test_incidents = [
        {
            "name": "Normal: Signal Failure",
            "text": "Signal failure at Central Station affecting express train EXP_001 during morning rush hour",
            "expected": "NORMAL"
        },
        {
            "name": "Normal: Track Maintenance",
            "text": "Scheduled track maintenance between STN_005 and STN_006, expect 15 minute delays",
            "expected": "NORMAL"
        },
        {
            "name": "Normal: Power Outage",
            "text": "Temporary power outage at West Hub, backup systems activated, minimal disruption",
            "expected": "NORMAL"
        },
        {
            "name": "Black Swan: Alien Invasion",
            "text": "bees attacking train",
            "expected": "ANOMALY"
        },
        {
            "name": "Black Swan: Time Travel",
            "text": "Train from the future has materialized on platform 9 3/4, causing temporal paradoxes",
            "expected": "ANOMALY"
        },
        {
            "name": "Black Swan: Dragon Attack",
            "text": "Fire-breathing dragon has made nest in main tunnel, engineers unable to proceed",
            "expected": "ANOMALY"
        }
    ]
    
    # Run tests
    print(f"\n[2/3] Testing {len(test_incidents)} incidents...\n")
    print("-"*70)
    
    results = []
    for i, incident in enumerate(test_incidents, 1):
        print(f"\nTest {i}/{len(test_incidents)}: {incident['name']}")
        print(f"Input: {incident['text'][:60]}...")
        
        try:
            result = pipeline.process(incident['text'])
            anomaly = result.get('anomaly', {})
            
            is_anomaly = anomaly.get('is_anomaly', False)
            score = anomaly.get('anomaly_score', anomaly.get('score', 0))
            
            status = "ANOMALY" if is_anomaly else "NORMAL"
            icon = "[!]" if is_anomaly else "[✓]"
            
            print(f"{icon} Result: {status} (score: {score:.4f})")
            print(f"Expected: {incident['expected']}")
            
            # Check if prediction matches expectation
            match = (status == incident['expected'])
            match_icon = "✓" if match else "✗"
            print(f"{match_icon} {'PASS' if match else 'FAIL'}")
            
            results.append({
                "test": incident['name'],
                "expected": incident['expected'],
                "got": status,
                "score": score,
                "match": match
            })
            
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
            results.append({
                "test": incident['name'],
                "expected": incident['expected'],
                "got": "ERROR",
                "score": 0,
                "match": False
            })
    
    # Summary
    print("\n" + "="*70)
    print("[3/3] TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['match'])
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    print("\nDetailed Results:")
    for r in results:
        status = "PASS" if r['match'] else "FAIL"
        print(f"  [{status}] {r['test']}: Expected {r['expected']}, Got {r['got']} (score: {r['score']:.4f})")
    
    print("\n" + "="*70)
    
    # Check for logged black swans
    if pipeline.storage and pipeline.storage.client:
        print("\nChecking Qdrant for logged black swans...")
        print("Run: python src/models/query_black_swans.py")
    
    print("\n[SUCCESS] Test complete!")

if __name__ == "__main__":
    test_anomaly_detection()
