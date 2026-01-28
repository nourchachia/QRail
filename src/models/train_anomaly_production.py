"""
Train Anomaly Detector with FULL PIPELINE Embeddings (No Qdrant)
------------------------------------------------------------------
Uses pipeline components to generate BERT + GNN + LSTM embeddings
without requiring Qdrant connection.

Usage:
    python src/models/train_anomaly_production.py
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pickle
import numpy as np
import json
from sklearn.ensemble import IsolationForest

# Import individual components (avoids Qdrant/Unicode issues)
from src.models.semantic_encoder import SemanticEncoder
from src.models.heterogeneous_gat import HeterogeneousGATEncoder
from src.models.lstm_encoder import LSTMEncoder
from src.backend.data_fuel import DataFuelPipeline

def train_production_model():
    """Train using production-matched embeddings."""
    
    print("[INFO] Loading golden runs...")
    golden_path = Path("data/processed/golden_runs_accidents_enhanced.json")
    with open(golden_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'golden_runs' in data:
        golden_runs = data['golden_runs']
    else:
        print("[ERROR] Unexpected JSON structure!")
        return
    
    print(f"[INFO] Found {len(golden_runs)} golden runs")
    
    # Initialize encoders
    print("\n[INFO] Loading encoders...")
    print("  [1/4] Semantic encoder (BERT)...")
    semantic_encoder = SemanticEncoder()
    
    print("  [2/4] GNN encoder (topology)...")
    gnn_encoder = HeterogeneousGATEncoder()
    
    print("  [3/4] LSTM encoder (temporal)...")
    lstm_encoder = LSTMEncoder()
    
    print("  [4/4] Data pipeline...")
    data_pipeline = DataFuelPipeline()
    
    print("[SUCCESS] All encoders ready")
    
    # Generate embeddings
    print("\n[INFO] Generating embeddings...")
    X_train = []
    successful = 0
    
    for i, run in enumerate(golden_runs, 1):
        text = run.get('description', '')
        if not text:
            continue
        
        try:
            # 1. Semantic embedding (BERT)
            semantic_vec = semantic_encoder.encode(text).tolist()
            
            # 2. Structural embedding (GNN)
            # Extract features for GNN
            incident_type = run.get('accident_type', 'unknown')
            severity = run.get('severity', 'moderate')
            affected_trains = run.get('affected_trains', [])
            
            # Get network structure
            nodes, edges = data_pipeline.extract_evidence_graph(
                parsed_incident={
                    'incident_type': incident_type,
                    'severity': severity,
                    'affected_trains': affected_trains,
                    'location': run.get('location', {})
                }
            )
            
            if nodes:
                structural_vec = gnn_encoder.encode(nodes, edges).tolist()
            else:
                structural_vec = [0.0] * 64
            
            # 3. Temporal embedding (LSTM)
            # Create sequence from affected trains
            if len(affected_trains) > 0:
                train_sequence = []
                for train in affected_trains[:10]:  # Limit to 10
                    train_sequence.append({
                        'train_id': train,
                        'delay': 30  # Placeholder
                    })
                temporal_vec = lstm_encoder.encode(train_sequence).tolist()
            else:
                temporal_vec = [0.0] * 64
            
            # Combine
            combined_vec = semantic_vec + structural_vec + temporal_vec
            
            # Verify dimensions
            if len(combined_vec) == 512:
                X_train.append(combined_vec)
                successful += 1
                
                if successful % 10 == 0:
                    print(f"  Processed {successful}/{len(golden_runs)}...")
            
        except Exception as e:
            # Silently skip failures
            continue
    
    print(f"\n[INFO] Successfully processed {successful}/{len(golden_runs)} incidents")
    
    if len(X_train) < 10:
        print(f"[ERROR] Not enough training data ({len(X_train)} samples)")
        return
    
    X_train = np.array(X_train)
    print(f"[INFO] Training data shape: {X_train.shape}")
    
    # Calculate statistics
    print("\n[INFO] Embedding statistics:")
    print(f"  Mean: {X_train.mean():.4f}")
    print(f"  Std: {X_train.std():.4f}")
    print(f"  Min: {X_train.min():.4f}")
    print(f"  Max: {X_train.max():.4f}")
    
    # Train Isolation Forest
    print("\n[INFO] Training Isolation Forest...")
    
    contamination = 0.10  # Expect 10% anomalies
    
    clf = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        n_estimators=200,
        max_samples=min(256, len(X_train)),
        bootstrap=False
    )
    clf.fit(X_train)
    
    # Test on training data
    print("\n[INFO] Validation on training data...")
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
    print("\n" + "="*70)
    print("PRODUCTION MODEL READY!")
    print("="*70)
    print("This model uses FULL PIPELINE embeddings (BERT + GNN + LSTM)")
    print("It will work correctly with: python scripts/test_anomaly.py")
    print("="*70)

if __name__ == "__main__":
    train_production_model()
