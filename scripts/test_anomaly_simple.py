"""
Simple Anomaly Detection Test (No Qdrant Required)
---------------------------------------------------
Tests ONLY the Isolation Forest model without full pipeline.

Usage:
    python scripts/test_anomaly_simple.py
"""

import pickle
import numpy as np
from pathlib import Path

def test_anomaly_simple():
    """Test the anomaly detector directly."""
    
    print("="*70)
    print("SIMPLE ANOMALY DETECTION TEST")
    print("="*70)
    
    # Load model
    model_path = Path("checkpoints/anomaly_detector/model.pkl")
    
    if not model_path.exists():
        print("\n[ERROR] Model not found!")
        print("Run: python src/models/train_anomaly_simple.py")
        return
    
    print(f"\n[1/3] Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("[SUCCESS] Model loaded")
    print(f"Model type: {type(model).__name__}")
    
    # Create test vectors
    print("\n[2/3] Creating test vectors...")
    
    # Normal vectors (similar to training data mean)
    normal_vectors = [
        np.random.randn(512) * 0.1,  # Small variance
        np.random.randn(512) * 0.15,
        np.random.randn(512) * 0.12,
    ]
    
    # Anomaly vectors (far from training data)
    anomaly_vectors = [
        np.random.randn(512) * 3.0 + 5.0,  # Large variance + offset
        np.random.randn(512) * 2.5 - 4.0,
        np.ones(512) * 10.0,  # Extreme values
    ]
    
    # Run predictions
    print("\n[3/3] Testing predictions...\n")
    print("-"*70)
    
    results = []
    
    print("\nNORMAL VECTORS (should NOT be flagged):")
    for i, vec in enumerate(normal_vectors, 1):
        pred = model.predict([vec])[0]
        score = model.decision_function([vec])[0]
        
        is_anomaly = (pred == -1)
        status = "ANOMALY" if is_anomaly else "NORMAL"
        icon = "[!]" if is_anomaly else "[OK]"
        
        print(f"  {icon} Test {i}: {status} (score: {score:.4f})")
        results.append({"expected": "NORMAL", "got": status, "score": score})
    
    print("\nANOMALY VECTORS (SHOULD be flagged):")
    for i, vec in enumerate(anomaly_vectors, 1):
        pred = model.predict([vec])[0]
        score = model.decision_function([vec])[0]
        
        is_anomaly = (pred == -1)
        status = "ANOMALY" if is_anomaly else "NORMAL"
        icon = "[!]" if is_anomaly else "[OK]"
        
        print(f"  {icon} Test {i}: {status} (score: {score:.4f})")
        results.append({"expected": "ANOMALY", "got": status, "score": score})
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    correct = sum(1 for r in results if r["expected"] == r["got"])
    total = len(results)
    
    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    print("\nScore Distribution:")
    scores = [r["score"] for r in results]
    print(f"  Min:  {min(scores):.4f}")
    print(f"  Max:  {max(scores):.4f}")
    print(f"  Mean: {np.mean(scores):.4f}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Test complete!")
    print("\nNext steps:")
    print("  1. Run with real incidents: python scripts/test_anomaly.py")
    print("  2. Start API: python src/api/main.py")
    print("  3. Test frontend: Open src/frontend/index.html")

if __name__ == "__main__":
    test_anomaly_simple()
