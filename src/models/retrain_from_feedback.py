"""
Retrain Anomaly Detector Using Labeled Black Swans
---------------------------------------------------
Improves the anomaly detector by learning from past black swan incidents.

This creates a feedback loop:
1. Incidents detected as anomalies are stored in Qdrant
2. This script retrieves them
3. Model is retrained to better recognize similar patterns

Usage:
    python src/models/retrain_from_feedback.py
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.backend.database import StorageManager

def retrain_with_feedback():
    """Retrain anomaly detector including black swan feedback."""
    
    print("[INFO] Starting feedback-enhanced retraining...")
    
    # Load storage
    storage = StorageManager()
    if not storage.client:
        print("[ERROR] Qdrant not available!")
        return
    
    # 1. Get normal incidents (golden runs)
    print("[INFO] Loading golden runs (normal patterns)...")
    golden_runs = storage.get_golden_runs()
    print(f"[INFO] Found {len(golden_runs)} golden runs")
    
    # For now, create dummy embeddings (in production, process through pipeline)
    normal_embeddings = [np.random.randn(512) for _ in golden_runs]
    
    # 2. Get black swan incidents from Qdrant
    print("[INFO] Querying black swan incidents from Qdrant...")
    try:
        results = storage.client.scroll(
            collection_name="operational_memory",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="detected_as_black_swan",
                        match=MatchValue(value=True)
                    )
                ]
            ),
            limit=100
        )
        
        black_swans = results[0]
        print(f"[INFO] Found {len(black_swans)} black swan incidents")
        
        # Extract embeddings
        black_swan_embeddings = []
        for point in black_swans:
            # Reconstruct 512-dim vector from Qdrant's multi-vector format
            semantic = point.vector.get('semantic', [])
            structural = point.vector.get('structural', [])
            temporal = point.vector.get('temporal', [])
            
            if semantic and structural and temporal:
                combined = np.array(semantic + structural + temporal)
                black_swan_embeddings.append(combined)
        
        print(f"[INFO] Extracted {len(black_swan_embeddings)} embeddings")
        
    except Exception as e:
        print(f"[WARN] Could not retrieve black swans: {e}")
        black_swan_embeddings = []
    
    # 3. Combine data
    if black_swan_embeddings:
        # Include black swans in training to better define boundaries
        all_data = normal_embeddings + black_swan_embeddings
        
        # Adjust contamination based on observed ratio
        contamination = len(black_swan_embeddings) / len(all_data)
        contamination = max(0.01, min(0.3, contamination))  # Clamp between 1% and 30%
        
        print(f"[INFO] Training with {len(all_data)} total samples")
        print(f"[INFO] Contamination rate: {contamination:.2%}")
    else:
        all_data = normal_embeddings
        contamination = 0.1
        print(f"[INFO] No black swans found, using default contamination: {contamination}")
    
    # 4. Train model
    X_train = np.array(all_data)
    print(f"[INFO] Training Isolation Forest...")
    
    clf = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        n_estimators=150  # Increased from default 100
    )
    clf.fit(X_train)
    
    # 5. Save updated model
    output_path = Path("checkpoints/anomaly_detector/model.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"[SUCCESS] Model retrained and saved: {output_path}")
    print(f"[INFO] Training data: {len(normal_embeddings)} normal + {len(black_swan_embeddings)} anomalies")

if __name__ == "__main__":
    retrain_with_feedback()
