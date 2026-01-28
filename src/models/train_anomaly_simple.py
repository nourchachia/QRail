"""
Simple Anomaly Detector Trainer
Train Isolation Forest without full pipeline initialization
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
import json

def train_simple():
    print("[INFO] Loading golden runs...")
    
    # Load golden runs directly
    golden_path = Path("data/processed/golden_runs_accidents_enhanced.json")
    if not golden_path.exists():
        print("[ERROR] Golden runs file not found!")
        return
    
    with open(golden_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract golden_runs array from nested structure
    if isinstance(data, dict) and 'golden_runs' in data:
        golden_runs = data['golden_runs']
    elif isinstance(data, list):
        golden_runs = data
    else:
        print("[ERROR] Unexpected JSON structure!")
        return
    
    print(f"[INFO] Found {len(golden_runs)} examples")
    
    # Create simple 384-dim semantic vectors from text
    # (Using random for now - in production, you'd use BERT)
    print("[INFO] Creating embeddings...")
    X_train = []
    
    for run in golden_runs:
        # Simple approach: create 512-dim random vector
        # In production, this would be the BERT + GNN + LSTM embeddings
        vec = np.random.randn(512)
        X_train.append(vec)
    
    X_train = np.array(X_train)
    print(f"[INFO] Training data shape: {X_train.shape}")
    
    # Train Isolation Forest
    print("[INFO] Training Isolation Forest...")
    clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    clf.fit(X_train)
    
    # Save model
    output_dir = Path("checkpoints/anomaly_detector")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"[SUCCESS] Model saved to: {output_path}")

if __name__ == "__main__":
    train_simple()
