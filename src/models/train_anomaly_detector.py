"""
Train Anomaly Detector (Isolation Forest)
-----------------------------------------
Trains a model to identify "Black Swan" events (outliers) based on incident embeddings.

Usage:
    python src/models/train_anomaly_detector.py
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.backend.integration import IncidentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnomalyTrainer")

def train_anomaly_detector():
    """
    Train Isolation Forest on Golden Runs embeddings.
    """
    try:
        logger.info("🚀 Starting Anomaly Detector training...")
        
        # 1. Initialize Pipeline (to get embeddings)
        pipeline = IncidentPipeline()
        
        # 2. Load Training Data (Golden Runs)
        golden_runs = pipeline.storage.get_golden_runs()
        if not golden_runs:
            logger.error("❌ No golden runs found! Cannot train anomaly detector.")
            return
            
        logger.info(f"📊 Found {len(golden_runs)} golden runs for training")
        
        # 3. Generate Embeddings for each run
        X_train = []
        
        for i, run in enumerate(golden_runs):
            logger.info(f"   Processing {i+1}/{len(golden_runs)}: {run.get('incident_id')}")
            
            # Extract text for semantic embedding
            text = run.get('description', '') 
            
            # --- Synthesize a "result" object like pipeline.process() would produce ---
            # This is a bit of a hack to reuse pipeline components without running full process()
            # We need: semantic (384), structural (64), temporal (64)
            
            # A. Semantic
            if pipeline.semantic_encoder:
                sem_vec = pipeline.semantic_encoder.encode(text).tolist()
            else:
                sem_vec = [0.0] * 384
                
            # B. Structural (GNN)
            # We need to map location to station_id to get GNN node features
            # Simplified: Use zero vector if complex to reconstruct, or try to run pipeline features
            # Better approach: Let's run the actual pipeline feature extraction if possible?
            # Processing raw text through pipeline is expensive and might not match golden run data structure perfectly.
            # Let's stick to Semantic for now + placeholders if GNN/LSTM not easily accessible from just 'run' dict
            # WAIT! The request says "Use Scikit-learn's IsolationForest on your existing embeddings". 
            # The integration.py generates a concatenation. 
            
            # Let's try to do a "light" feature extraction
            # Structural/Temporal might be hard to get perfect without full simulation state context
            # COMPROMISE: We will trust the Semantic embedding as the primary signal for "Black Swan" 
            # (unprecedented descriptions) but still allocate slots for others to match 512-dim format.
            
            # ...Actually, let's try to get them if we can.
            # The run object has 'location' -> 'from_station', etc.
            
            struct_vec = [0.0] * 64
            temp_vec = [0.0] * 64
            
            # If we really want to be robust, we'd need to simulate the GNN/LSTM inputs.
            # For this MVP Anomaly Detector, let's use the Semantic Vector primarily, 
            # but pad to 512 dimensions so it's compatible if we add the others later.
            # OR: Since integration.py produces 512-dim vector for inference, 
            # our training data MUST be 512-dim distributions.
            
            # Optimization: If we just use params from integration.py default fallback...
            # Let's try to use the pipeline to process the description text directly! 
            # This ensures training data distribution matches inference distribution exactly.
            
            try:
                # We interpret the golden run description as a new incident
                # This is the most robust way to get matching embeddings
                processed_result = pipeline.process(text)
                emb = processed_result['embeddings']
                
                # Concatenate: Semantic (384) + Structural (64) + Temporal (64) = 512
                # Note: integration.py returns lists
                vector = (
                    emb.get('semantic', [0.0]*384) + 
                    emb.get('structural', [0.0]*64) + 
                    emb.get('temporal', [0.0]*64)
                )
                
                if len(vector) != 512:
                    logger.warning(f"⚠️ Vector dimension mismatch: {len(vector)} (expected 512). Padding/Truncating.")
                    vector = vector[:512] + [0.0] * max(0, 512 - len(vector))
                
                X_train.append(vector)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process run {run.get('incident_id')}: {e}")
                continue

        if not X_train:
            logger.error("❌ No valid training vectors generated.")
            return

        X_train = np.array(X_train)
        logger.info(f"🧠 Training data shape: {X_train.shape}")

        # 4. Train Isolation Forest
        # contamination='auto' -> let model decide threshold
        # random_state=42 -> reproducible
        clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        clf.fit(X_train)
        
        # 5. Save Model
        output_dir = Path("checkpoints/anomaly_detector")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model.pkl"
        
        with open(output_path, 'wb') as f:
            pickle.dump(clf, f)
            
        logger.info(f"✅ Model saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_anomaly_detector()
