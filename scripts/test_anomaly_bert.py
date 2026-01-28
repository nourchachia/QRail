"""
Test Anomaly Detection with BERT Embeddings
--------------------------------------------
Tests the retrained model using real BERT embeddings.

Usage:
    python scripts/test_anomaly_bert.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.models.semantic_encoder import SemanticEncoder

def test_with_bert():
    """Test anomaly detector with BERT embeddings."""
    
    print("="*70)
    print("ANOMALY DETECTION TEST (BERT Embeddings)")
    print("="*70)
    
    # Load model
    model_path = Path("checkpoints/anomaly_detector/model.pkl")
    if not model_path.exists():
        print("\n[ERROR] Model not found!")
        print("Run: python src/models/train_anomaly_bert.py")
        return
    
    print(f"\n[1/4] Loading model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("[SUCCESS] Model loaded")
    
    # Load BERT encoder
    print("\n[2/4] Loading BERT encoder...")
    encoder = SemanticEncoder()
    print("[SUCCESS] Encoder ready")
    
    # Test incidents
    test_cases = [
        {
            "name": "Normal: Signal Failure",
            "text": "Signal failure at Central Station affecting express train during morning rush hour",
            "expected": "NORMAL"
        },
        {
            "name": "Normal: Track Maintenance",
            "text": "Scheduled track maintenance between stations causing minor delays",
            "expected": "NORMAL"
        },
        {
            "name": "Normal: Power Outage",
            "text": "Temporary power outage at hub station, backup systems activated",
            "expected": "NORMAL"
        },
        {
            "name": "Black Swan: Bees",
            "text": "bees attacking train",
            "expected": "ANOMALY"
        },
        {
            "name": "Black Swan: Aliens",
            "text": "Unidentified flying objects hovering over railway tracks suspending operations",
            "expected": "ANOMALY"
        },
        {
            "name": "Black Swan: Time Travel",
            "text": "Train from the future materialized causing temporal paradoxes",
            "expected": "ANOMALY"
        },
        {
            "name": "Black Swan: Dragon",
            "text": "Fire-breathing dragon has made nest in main tunnel blocking engineers",
            "expected": "ANOMALY"
        },
        {
            "name": "Black Swan: Zombies",
            "text": "Zombie outbreak at platform 5 evacuating passengers immediately",
            "expected": "ANOMALY"
        }
    ]
    
    # Run tests
    print(f"\n[3/4] Testing {len(test_cases)} incidents...")
    print("-"*70)
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{len(test_cases)}: {case['name']}")
        print(f"Input: {case['text'][:60]}...")
        
        # Generate BERT embedding
        semantic_vec = encoder.encode(case['text']).tolist()
        structural_vec = [0.0] * 64
        temporal_vec = [0.0] * 64
        combined_vec = semantic_vec + structural_vec + temporal_vec
        
        # Predict
        prediction = model.predict([combined_vec])[0]
        score = model.decision_function([combined_vec])[0]
        
        is_anomaly = (prediction == -1)
        status = "ANOMALY" if is_anomaly else "NORMAL"
        icon = "[!]" if is_anomaly else "[OK]"
        
        print(f"{icon} Result: {status} (score: {score:.4f})")
        print(f"Expected: {case['expected']}")
        
        match = (status == case['expected'])
        match_icon = "[PASS]" if match else "[FAIL]"
        print(f"{match_icon}")
        
        results.append({
            "test": case['name'],
            "expected": case['expected'],
            "got": status,
            "score": score,
            "match": match
        })
    
    # Summary
    print("\n" + "="*70)
    print("[4/4] TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['match'])
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Score analysis
    normal_scores = [r['score'] for r in results if r['expected'] == 'NORMAL']
    anomaly_scores = [r['score'] for r in results if r['expected'] == 'ANOMALY']
    
    print(f"\nScore Analysis:")
    print(f"  Normal incidents:  mean={np.mean(normal_scores):.4f}, range=[{min(normal_scores):.4f}, {max(normal_scores):.4f}]")
    print(f"  Anomaly incidents: mean={np.mean(anomaly_scores):.4f}, range=[{min(anomaly_scores):.4f}, {max(anomaly_scores):.4f}]")
    
    print(f"\nDetailed Results:")
    for r in results:
        status = "PASS" if r['match'] else "FAIL"
        print(f"  [{status}] {r['test']}: Expected {r['expected']}, Got {r['got']} (score: {r['score']:.4f})")
    
    print("\n" + "="*70)
    
    if passed == total:
        print("[SUCCESS] All tests passed!")
    elif passed >= total * 0.7:
        print("[PARTIAL] Most tests passed, model needs tuning")
    else:
        print("[FAIL] Model needs retraining with different parameters")
    
    print("\nNext steps:")
    if passed < total:
        print("  - Adjust contamination parameter in train_anomaly_bert.py")
        print("  - Add more diverse golden runs for better baseline")
        print("  - Consider using full pipeline (GNN + LSTM) for better accuracy")

if __name__ == "__main__":
    test_with_bert()
