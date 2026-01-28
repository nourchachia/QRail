"""
Train Anomaly Detector with REAL BERT Embeddings
-------------------------------------------------
Uses actual semantic encoder to create training data.

Usage:
    python src/models/train_anomaly_bert.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pickle
import numpy as np
import json
from sklearn.ensemble import IsolationForest

# Import ONLY the semantic encoder (avoids Qdrant unicode issues)
from src.models.semantic_encoder import SemanticEncoder

def train_with_bert():
    """Train using real BERT embeddings from golden runs."""
    
    print("[INFO] Loading golden runs...")
    
    # Load data
    golden_path = Path("data/processed/golden_runs_accidents_enhanced.json")
    with open(golden_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'golden_runs' in data:
        golden_runs = data['golden_runs']
    else:
        print("[ERROR] Unexpected JSON structure!")
        return
    
    print(f"[INFO] Found {len(golden_runs)} golden runs")
    
    # Initialize BERT encoder
    print("[INFO] Loading BERT semantic encoder...")
    encoder = SemanticEncoder()
    print("[SUCCESS] Encoder ready")
    
    # Generate REAL embeddings
    print("[INFO] Creating embeddings from incident descriptions...")
    X_train = []
   
    
    for i, run in enumerate(golden_runs, 1):
        # Get incident description
        text = run.get('description', '')
        
        if not text:
            print(f"  [WARN] Run {i} has no description, skipping...")
            continue
        
        # Generate REAL semantic embedding (384-dim)
        semantic_vec = encoder.encode(text).tolist()
        
        # Use ONLY semantic embedding (384-dim)
        # We ignore structural/temporal because they may be missing in production
        # and we want to detect "unprecedented incident TYPES" primarily.
        
        X_train.append(semantic_vec)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(golden_runs)}...")
    
    X_train = np.array(X_train)
    print(f"[INFO] Training data shape: {X_train.shape}")
    
    # Train Isolation Forest
    print("[INFO] Training Isolation Forest...")
    clf = IsolationForest(
        contamination=0.15,  # Increased from 0.05 - expect more anomalies
        random_state=42,
        n_jobs=-1,
        n_estimators=200,  # Increased from default 100
        max_samples=256   # Limit sample size for better generalization
    )
    clf.fit(X_train)
    
    # Test on training data
    print("\n[INFO] Testing on training data...")
    predictions = clf.predict(X_train)
    scores = clf.decision_function(X_train)
    
    anomalies = (predictions == -1).sum()
    print(f"  Detected {anomalies}/{len(X_train)} as anomalies ({anomalies/len(X_train)*100:.1f}%)")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Score mean: {scores.mean():.4f}")
    
    # Save model
    output_dir = Path("checkpoints/anomaly_detector")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"\n[SUCCESS] Model saved to: {output_path}")
    print("\n[INFO] Model trained with REAL BERT embeddings!")
    print("[INFO] Now test with: python scripts/test_anomaly.py")

if __name__ == "__main__":
    train_with_bert()
