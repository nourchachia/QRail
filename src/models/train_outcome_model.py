
"""
Train Outcome Predictor (Model 5) - XGBoost
File: src/models/train_outcome_model.py

Purpose:
    Trains the XGBoost model to predict resolution success (outcome_score)
    based on incident context and proposed actions.

Input:
    - data/processed/incidents.json

Output:
    - checkpoints/outcome_predictor/model.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.outcome_predictor_xgb import OutcomePredictor
from src.backend.feature_extractor import DataFuelPipeline

# Import AI models for embedding generation
import torch
from src.models.gnn_encoder import HeterogeneousGATEncoder
from src.models.cascade.lstm_encoder import LSTMEncoder
from src.models.semantic_encoder import SemanticEncoder
from torch_geometric.data import Data

def load_data(data_dir="data"):
    """Load incidents and extract features using REAL EMBEDDINGS for training"""
    print("⏳ Loading incidents and generating REAL embeddings...")
    
    # Initialize pipeline
    pipeline = DataFuelPipeline(data_dir=data_dir)
    
    # Initialize AI models (same as integration.py)
    print("   Loading AI encoders...")
    gnn_encoder = HeterogeneousGATEncoder()
    lstm_encoder = LSTMEncoder()
    semantic_encoder = SemanticEncoder()
    
    # Load trained checkpoints
    gnn_ckpt = Path("checkpoints/gnn/best_model.pt")
    lstm_ckpt = Path("checkpoints/lstm/best_model.pt")
    
    if gnn_ckpt.exists():
        checkpoint = torch.load(gnn_ckpt, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            gnn_encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            gnn_encoder.load_state_dict(checkpoint)
        print("   ✓ GNN loaded from checkpoint")
    else:
        print("   ⚠ GNN using random weights (no checkpoint)")
        
    if lstm_ckpt.exists():
        checkpoint = torch.load(lstm_ckpt, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            lstm_encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            lstm_encoder.load_state_dict(checkpoint)
        print("   ✓ LSTM loaded from checkpoint")
    else:
        print("   ⚠ LSTM using random weights (no checkpoint)")
    
    gnn_encoder.eval()
    lstm_encoder.eval()
    
    # Load incidents
    with open(Path(data_dir) / "processed" / "incidents.json", "r") as f:
        data = json.load(f)
    
    incidents = data.get("train", [])
    if not incidents:
        print("❌ No training data found!")
        return None, None
        
    X_list = []
    y_list = []
    
    print(f"   Processing {len(incidents)} incidents with REAL embeddings...")
    
    for i, inc in enumerate(incidents):
        # Skip if no outcome score
        if "outcome_score" not in inc:
            continue
        
        try:
            # === GENERATE REAL EMBEDDINGS (same as integration.py) ===
            
            # 1. Semantic embedding (384-dim)
            text = pipeline.extract_semantic_text(inc)
            semantic_vec = semantic_encoder.encode(text).tolist()
            
            # 2. Structural embedding (64-dim)
            gnn_feat = pipeline.extract_gnn_features(inc)
            n_nodes = len(gnn_feat.get('nodes', []))
            
            if n_nodes > 0:
                # Build real graph
                nodes_list = gnn_feat.get('nodes', [])
                id_to_idx = {node['id']: i for i, node in enumerate(nodes_list)}
                
                # Process edges
                raw_edges = gnn_feat.get('edges', [])
                processed_edges = []
                for edge in raw_edges:
                    if isinstance(edge, dict):
                        f_idx = id_to_idx.get(edge['from'])
                        t_idx = id_to_idx.get(edge['to'])
                        if f_idx is not None and t_idx is not None:
                            processed_edges.append([f_idx, t_idx])
                
                # Create edge_index
                if processed_edges:
                    edge_index = torch.tensor(processed_edges, dtype=torch.long).t().contiguous()
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                
                # Pad node features to 14-dim
                padded_x = []
                for node in nodes_list:
                    feats = node.get('features', [0.0] * 10)
                    padded_x.append(feats + [0.0] * (14 - len(feats)))
                
                graph_data = Data(
                    x=torch.tensor(padded_x, dtype=torch.float),
                    edge_index=edge_index,
                    edge_attr=torch.zeros((edge_index.size(1), 8), dtype=torch.float),
                    node_type=torch.zeros(n_nodes, dtype=torch.long),
                    batch=torch.zeros(n_nodes, dtype=torch.long)
                )
                
                with torch.no_grad():
                    structural_vec = gnn_encoder(graph_data, return_embedding=True).numpy()[0].tolist()
            else:
                structural_vec = [0.0] * 64
            
            # 3. Temporal embedding (64-dim)
            lstm_seq = pipeline.extract_lstm_sequence(inc.get('train_id', ''))
            if lstm_seq:
                lstm_feat = torch.tensor(lstm_seq, dtype=torch.float)
                if lstm_feat.dim() == 1:
                    lstm_feat = lstm_feat.unsqueeze(0).unsqueeze(0)
                elif lstm_feat.dim() == 2:
                    lstm_feat = lstm_feat.unsqueeze(0)
                
                with torch.no_grad():
                    temporal_vec = lstm_encoder(lstm_feat).numpy()[0].tolist()
            else:
                temporal_vec = [0.0] * 64
            
            # === MANUAL CONTEXT FEATURES (8-dim) ===
            f_context = [
                float(inc.get("severity_level", 3)),
                float(inc.get("network_load_pct", 50)) / 100.0,
                1.0 if inc.get("is_peak") else 0.0,
                float(inc.get("trains_affected_count", 1)),
                1.0 if inc.get("weather_condition") in ["snow", "storm", "heavy_rain"] else 0.0
            ]
            
            # Resolution features
            res_code = inc.get("resolution_strategy", "UNKNOWN")
            res_hash = hash(res_code) % 100 / 100.0
            
            f_action = [
                res_hash,
                float(inc.get("estimated_delay_minutes", 0)) / 1000.0,
                1.0 if list(inc.get("actions_taken", [])) else 0.0
            ]
            
            # === COMBINE: 512 (embeddings) + 8 (manual) = 520 total ===
            combined_features = np.concatenate([
                structural_vec,   # 64
                temporal_vec,     # 64
                semantic_vec,     # 384
                f_context,        # 5
                f_action          # 3
            ])
            
            X_list.append(combined_features)
            y_list.append(float(inc.get("outcome_score", 0.5)))
            
        except Exception as e:
            print(f"   ⚠ Skipped incident {i}: {e}")
            continue
        
        if (i+1) % 100 == 0:
            print(f"   ... processed {i+1}/{len(incidents)}")

    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ Generated dataset with REAL embeddings: X={X.shape}, y={y.shape}")
    print(f"   Expected shape: (N, 520) where 520 = 64+64+384+5+3")
    return X, y

def main():
    print("="*60)
    print("🚀 Training Model 5 (Outcome Predictor - REAL DATA)")
    print("="*60)
    
    # 1. Data Preparation
    X, y = load_data()
    if X is None:
        return
        
    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train
    predictor = OutcomePredictor()
    history = predictor.train(X_train, y_train, X_val, y_val)
    
    # 4. Save (Manual fix to avoid changing outcome_predictor_xgb.py)
    # The class .save() method fails on some XGBoost versions due to sklearn wrapper issues
    # So we save manually here using the native get_booster() method
    save_path = "checkpoints/outcome_predictor/model"
    model_json_path = Path(save_path).with_suffix('.json')
    meta_pkl_path = Path(save_path).with_suffix('.pkl')
    
    # Ensure directory exists
    model_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model binary (native XGBoost format)
    predictor.model.get_booster().save_model(str(model_json_path))
    
    # Save metadata (mimicking what the class would have done)
    import pickle
    metadata = {
        'is_trained': predictor.is_trained,
        'n_estimators': predictor.model.n_estimators,
        'max_depth': predictor.model.max_depth,
        'learning_rate': predictor.model.learning_rate
    }
    with open(meta_pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n✅ Model 5 trained and saved to:", str(model_json_path))
    print(f"   Validation MSE: {history['val_mse']:.4f}")

if __name__ == "__main__":
    main()
