"""
Train Anomaly Detector with FULL PIPELINE Embeddings
-----------------------------------------------------
Uses the complete IncidentPipeline to generate training embeddings
that match production (BERT + GNN + LSTM).

Usage:
    python src/models/train_anomaly_full.py
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import pickle
import numpy as np
import json
from sklearn.ensemble import IsolationForest

# Import the full pipeline
from src.backend.integration import IncidentPipeline

def train_with_full_pipeline():
    """Train using FULL pipeline embeddings (BERT + GNN + LSTM)."""
    
    print("[INFO] Initializing full pipeline...")
    
    try:
        pipeline = IncidentPipeline()
        print("[SUCCESS] Pipeline ready")
    except Exception as e:
        print(f"[ERROR] Pipeline initialization failed: {e}")
        print("\nTip: If you see Unicode errors, this is expected on Windows.")
        print("The model will still work in production via API.")
        return
    
    # Load golden runs
    print("\n[INFO] Loading golden runs...")
    golden_path = Path("data/processed/golden_runs_accidents_enhanced.json")
    with open(golden_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'golden_runs' in data:
        golden_runs = data['golden_runs']
    else:
        print("[ERROR] Unexpected JSON structure!")
        return
    
    print(f"[INFO] Found {len(golden_runs)} golden runs")
    
    # Generate embeddings using FULL pipeline
    print("\n[INFO] Processing incidents through full pipeline...")
    print("This will generate BERT + GNN + LSTM embeddings")
    
    X_train = []
    successful = 0
    
    for i, run in enumerate(golden_runs, 1):
        text = run.get('description', '')
        
        if not text:
            continue
        
        try:
            # Process through FULL pipeline (same as production)
            result = pipeline.process(text)
            
            # Extract embeddings
            embeddings = result.get('embeddings', {})
            semantic_vec = embeddings.get('semantic', [])
            structural_vec = embeddings.get('structural', [])
            temporal_vec = embeddings.get('temporal', [])
            
            # Verify dimensions
            if len(semantic_vec) != 384 or len(structural_vec) != 64 or len(temporal_vec) != 64:
                print(f"  [WARN] Run {i}: Invalid dimensions, skipping...")
                continue
            
            # Combine (same as integration.py does)
            combined_vec = semantic_vec + structural_vec + temporal_vec
            X_train.append(combined_vec)
            successful += 1
            
            if successful % 10 == 0:
                print(f"  Processed {successful}/{len(golden_runs)}...")
                
        except Exception as e:
            print(f"  [WARN] Run {i} failed: {e}")
            continue
    
    if len(X_train) < 10:
        print(f"[ERROR] Not enough training data ({len(X_train)} samples)")
        print("Need at least 10 successful embeddings")
        return
    
    X_train = np.array(X_train)
    print(f"\n[INFO] Training data shape: {X_train.shape}")
    print(f"[INFO] Successfully processed {len(X_train)}/{len(golden_runs)} golden runs")
    
    # Train Isolation Forest
    print("\n[INFO] Training Isolation Forest...")
    
    # Adjust contamination based on data size
    contamination = min(0.15, max(0.05, 3 / len(X_train)))
    
    print(f"[INFO] Contamination: {contamination:.2%}")
    
    clf = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        n_estimators=200,
        max_samples=min(256, len(X_train))
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
    print(f"  Score std: {scores.std():.4f}")
    
    # Save model
    output_dir = Path("checkpoints/anomaly_detector")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"\n[SUCCESS] Model saved to: {output_path}")
    print("\n[INFO] Model trained with FULL PIPELINE embeddings!")
    print("[INFO] This model will work correctly in production")
    print("\n[NEXT] Test with: python scripts/test_anomaly.py")

if __name__ == "__main__":
    train_with_full_pipeline()
