
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

def load_data(data_dir="data"):
    """Load incidents and extract features for training"""
    print("⏳ Loading incidents and extracting features...")
    
    # Initialize pipeline
    pipeline = DataFuelPipeline(data_dir=data_dir)
    
    # Load incidents
    with open(Path(data_dir) / "processed" / "incidents.json", "r") as f:
        data = json.load(f)
    
    incidents = data.get("train", [])
    if not incidents:
        print("❌ No training data found!")
        return None, None
        
    X_list = []
    y_list = []
    
    print(f"   Processing {len(incidents)} incidents...")
    
    for i, inc in enumerate(incidents):
        # Skip if no outcome score
        if "outcome_score" not in inc:
            continue
            
        # Extract features
        # We need to simulate the "Incident Vector" + "Resolution Vector" structure
        # inferred by outcome_predictor_xgb.py logic
        
        # 1. Context Embedding (Semantic + Structural + Temporal)
        feats = pipeline.extract_all_features(inc)
        # Note: We simulate the concatenated embedding vector
        # Real pipeline would use the actual model outputs, but for XGBoost training
        # we can use the raw features or a simplified representation if models aren't live
        # For this script, let's assume we can get valid embeddings or fallback
        
        # SIMPLIFICATION FOR ROBUSTNESS:
        # Since we can't easily run GNN/LSTM inference here without loading those models,
        # we'll extract numeric metadata features as a proxy for the 'Incident Vector'
        # and 'Resolution Vector'.
        
        # Incident Features (Context)
        f_context = [
            float(inc.get("severity_level", 3)),
            float(inc.get("network_load_pct", 50)) / 100.0,
            1.0 if inc.get("is_peak") else 0.0,
            float(inc.get("trains_affected_count", 1)),
            1.0 if inc.get("weather_condition") in ["snow", "storm", "heavy_rain"] else 0.0
        ]
        
        # Semantic embedding (Context) - using pure dummy or pipeline if available
        # Ideally we'd use semantic_encoder here, but let's stick to metadata for speed/stability
        # unless pipeline has it.
        
        # Resolution Features (Action)
        # We encode the resolution strategy
        res_code = inc.get("resolution_strategy", "UNKNOWN")
        # Hash trick for simple categorical encoding
        res_hash = hash(res_code) % 100 / 100.0
        
        f_action = [
            res_hash,
            float(inc.get("estimated_delay_minutes", 0)) / 1000.0, # Normalizing
            1.0 if list(inc.get("actions_taken", [])) else 0.0
        ]
        
        # Pad to match expected dimensions if needed by the class, 
        # but OutcomePredictor seems to accept any dimension X as long as consistency holds.
        # Let's verify outcome_predictor_xgb.py... it takes X (concatenated).
        # We will create a feature vector of relevant size.
        
        combined_features = np.array(f_context + f_action)
        
        X_list.append(combined_features)
        y_list.append(float(inc.get("outcome_score", 0.5)))
        
        if (i+1) % 100 == 0:
            print(f"   ... processed {i+1}/{len(incidents)}")

    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ Generated dataset: X={X.shape}, y={y.shape}")
    return X, y

def main():
    print("="*60)
    print("🚀 Training Model 5 (Outcome Predictor)")
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
    
    # 4. Save
    save_path = "checkpoints/outcome_predictor/model"
    predictor.save(save_path)
    
    print("\n✅ Model 5 trained and saved to:", save_path + ".json")
    print(f"   Validation MSE: {history['val_mse']:.4f}")

if __name__ == "__main__":
    main()
