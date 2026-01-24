import sys
import os
import json
import random
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# 1. SETUP PATHS FIRST (Critical for imports)
# Add project root to path so we can import 'src'
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. LOAD ENV
current_path = Path(__file__).resolve()
for _ in range(3):
    env_path = current_path / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        break
    current_path = current_path.parent
else:
    load_dotenv()

# 3. NOW IMPORT LOCAL MODULES - ROBUST DIRECT IMPORT
# Since standard import is failing due to path confusion/cache, we load the file directly.
import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot allow module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

models_dir = Path(project_root) / "src" / "models"
try:
    gnn_module = load_module_from_path("gnn_encoder", models_dir / "gnn_encoder.py")
    # FIX: Class name IS HeterogeneousGATEncoder, not GNNEncoder
    GNNEncoder = gnn_module.HeterogeneousGATEncoder
    print("✅ DEBUG: Manually loaded HeterogeneousGATEncoder class")

    # FIX: Correct Path for LSTM is in 'cascade' subdir
    lstm_module = load_module_from_path("lstm_encoder", models_dir / "cascade" / "lstm_encoder.py")
    LSTMEncoder = lstm_module.LSTMEncoder
    print("✅ DEBUG: Manually loaded LSTMEncoder class")
except Exception as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    raise e

from src.backend.database import StorageManager
import torch

# Graceful Fallback for Semantic Encoder (dependency issues)
try:
    from src.models.semantic_encoder import SemanticEncoder
    SEMANTIC_AVAILABLE = True
except (ImportError, ValueError) as e:
    print(f"⚠️ SemanticEncoder import failed: {e}")
    print("   (Proceeding with dummy text vectors for now)")
    SEMANTIC_AVAILABLE = False
    class SemanticEncoder: # Dummy class
         def encode(self, text): return torch.zeros(384)

class MemoryUploader:
    def __init__(self, data_dir: str = "data"):
        print("📤 Initializing Memory Uploader...")
        
        # Use project_root to ensure we find the GLOBAL data folder
        # even if run from inside src/backend/
        abs_data_path = str(Path(project_root) / "data")
        
        # Get credentials
        url = os.getenv("QDRANT_URL")
        key = os.getenv("QDRANT_API_KEY")
        
        # Initialize storage
        self.storage = StorageManager(
            data_dir=abs_data_path,
            qdrant_url=url,
            qdrant_api_key=key
        )
        
        # Initialize Encoders (Models 1, 2, 3)
        self.gnn = GNNEncoder()
        self.lstm = LSTMEncoder()
        self.sem = SemanticEncoder()
        
        # Ensure collection and payload indices exist
        print("   Initializing Qdrant collection and indexing...")
        self.storage.init_operational_memory()

    def load_training_data(self) -> List[Dict]:
        """Load REAL incidents.json and generate embeddings using models."""
        print("   Loading real training data...")
        
        # PRIORITY FIX: Look for exact filename incidents.json first
        # previous logic was calling get_incidents("train") which looks for train.json
        incidents = self.storage.load_json("incidents.json")
        
        if not incidents and hasattr(self.storage, 'get_incidents'):
             incidents = self.storage.get_incidents("train")
        
        # NESTED STRUCTURE FIX: 
        # incidents.json is {"metadata": ..., "train": [...]}
        # We need the list under 'train'
        if isinstance(incidents, dict):
            if "train" in incidents:
                incidents = incidents["train"]
            elif "incidents" in incidents:
                incidents = incidents["incidents"]
            else:
                # If it's a dict but no known key, maybe it's a single incident?
                # Safer to keep as is or wrap in list if it looks like an incident
                if "incident_id" in incidents:
                    incidents = [incidents]
                else:
                    print("⚠️ JSON is a dictionary but contains no 'train' or 'incidents' key.")
                    incidents = []

        if not incidents:
            print("⚠️ incidents.json not found or empty!")
            print("   Using 5 dummy incidents JUST to test the pipeline...")
            return self._generate_dummy_data(5)
            
        print(f"   Loaded {len(incidents)} real incidents.")
        
        # 3. Process with Encoders
        processed_incidents = []
        print("   Generating embeddings with AI models (this may take a moment)...")
        
        for inc in incidents:
            # TYPE-SAFETY FIX: Skip anything that isn't a dictionary (like metadata strings)
            if not isinstance(inc, dict):
                continue

            # 1. Semantic (Text)
            text = inc.get('log', '') or inc.get('semantic_description', '') or str(inc)
            sem_vec = self.sem.encode(text).tolist()
            
            # 2. Structural (GNN)
            dummy_graph = self._create_dummy_graph_input()
            struct_vec = self.gnn(dummy_graph).detach().numpy()[0].tolist()
            
            # 3. Temporal (LSTM)
            dummy_seq = torch.randn(1, 10, 4) 
            temp_vec = self.lstm(dummy_seq).detach().numpy()[0].tolist()
            
            # Add to incident
            inc["embeddings"] = {
                "semantic": sem_vec,
                "structural": struct_vec,
                "temporal": temp_vec
            }
            # Add integer ID if missing or string
            if "incident_id" not in inc or isinstance(inc['incident_id'], str):
                 import uuid
                 inc['org_id'] = inc.get('incident_id')
                 # Safely use current timestamp or random for hash if needed
                 inc['incident_id'] = str(uuid.uuid4())
            
            processed_incidents.append(inc)
            
        return processed_incidents

    def _create_dummy_graph_input(self):
        """Helper to create dummy input for GNN (until graph builder is ready)."""
        from torch_geometric.data import Data
        x = torch.randn(10, 14) 
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        node_type = torch.zeros(10, dtype=torch.long)
        edge_attr = torch.randn(20, 8) 
        return Data(x=x, edge_index=edge_index, batch=batch, node_type=node_type, edge_attr=edge_attr)

    def _generate_dummy_data(self, count: int) -> List[Dict]:
        """Backup generator."""
        import uuid
        data = []
        for i in range(count):
            data.append({
                "incident_id": str(uuid.uuid4()), 
                "timestamp": "2024-01-01T12:00:00",
                "log": f"Simulation incident {i}",
                "meta": {"archetype": "Signal Failure", "day": "Monday"},
                "location_id": "SEG_001",
                "resolution": {"outcome_score": 0.9, "is_golden": i % 10 == 0},
                "is_dummy": True, # MARKER: So we can delete ONLY these later
                "embeddings": {
                    "semantic": [random.random() for _ in range(384)],
                    "structural": [random.random() for _ in range(64)],
                    "temporal": [random.random() for _ in range(64)]
                }
            })
        return data

    def upload(self, clear_dummy_only: bool = True):
        """Main upload process."""
        incidents = self.load_training_data()
        
        # SAFE CLEANUP: Only delete points marked as 'is_dummy'
        if clear_dummy_only and self.storage.client:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            print("🧹 Cleaning up ONLY temporary dummy data from Qdrant...")
            try:
                self.storage.client.delete(
                    collection_name="operational_memory",
                    points_selector=Filter(
                        must=[FieldCondition(key="payload.is_dummy", match=MatchValue(value=True))]
                    )
                )
            except Exception as e:
                print(f"   (Cleanup skipped or failed: {e})")

        print(f"🚀 Uploading {len(incidents)} memories to Qdrant Cloud...")
        
        # Call the upload method
        self.storage.upload_incident_memory(incidents)
        
        print("✅ Process complete.")

if __name__ == "__main__":
    uploader = MemoryUploader()
    uploader.upload()

"""
==============================================================================
📘 DOCUMENTATION & HANDOFF GUIDE (READ ME)
==============================================================================

WHAT IS THIS SCRIPT?
--------------------
This is the "Teacher" of the Neural Rail system. Its sole job is to take raw historical data, 
process it through our AI models (GNN, LSTM, Semantic), and save the "memories" into Qdrant Cloud.

INPUT DATA:
-----------
1. `data/processed/incidents.json` (The raw history log)
   - Contains: timestamp, location_id, log text, delay duration.
2. `src/models/*.py` (The untrained AI brains)
   - Reads `HeterogeneousGATEncoder` (Topology)
   - Reads `LSTMEncoder` (Sequence)
   - Reads `SemanticEncoder` (Text)

HOW IT WORKS (The "Pipeline"):
------------------------------
1. LOAD: It reads the JSON file.
2. ENCODE: It runs every incident through the 3 models:
   - Text -> Semantic Vector (384-dim)
   - Location -> GNN Vector (64-dim) [Currently using dummy inputs until Trainer is ready]
   - Timeline -> LSTM Vector (64-dim) [Currently using dummy inputs until Trainer is ready]
3. UPLOAD: It sends a "Point" to Qdrant containing:
   - The 3 Vectors (for search)
   - The Original JSON Payload (for display)

WHERE TO "PLUG IT":
-------------------
- This script is STANDALONE. 
- Run it manually: `python src/backend/uploader.py`
- Run it WHENEVER you have new training data (e.g., once a week).
- It DOES NOT run during the live demo (it prepares the brain beforehand).

OUTPUT:
-------
- Populates the `operational_memory` collection in Qdrant Cloud.
- This allows `search_engine.py` (and the API) to find similar past cases.

FAILSAFE (WHAT IF NO DATA?):
----------------------------
If `incidents.json` is missing or empty, the script enters "BOOTSTRAP MODE":
- It automatically generates 5 dummy incidents with random vectors.
- This ensures you can always test the Qdrant connection and API workflow even 
  before the data generation or model training is finished.
- You will see a `⚠️ incidents.json not found` warning in the console.

NEXT STEPS (For the Team):
--------------------------
1. Use `src/models/trainer.py` to actually TRAIN the GNN and LSTM models.
   (Right now, they are untrained, so the vectors are random "noise").
2. Once trained, run this uploader again to overwrite the memory with "Smart" vectors.
==============================================================================
"""
