"""
================================================================================
Neural Rail Conductor - Memory Uploader (Enhanced Version with Step Comments)
================================================================================
ROLE: The "Teacher" - Populates Qdrant with Historical Knowledge
    - Loads raw incidents from JSON files
    - Generates embeddings using AI models (GNN, LSTM,Semantic)
    - Uploads to Qdrant Cloud with deduplication
    - Provides bootstrap mode for testing
WORKFLOW:
    1. Load incidents.json
    2. For each incident:
       a. Generate semantic vector (384-dim) using MiniLM
       b. Generate structural vector (64-dim) using GNN
       c. Generate temporal vector (64-dim) using LSTM
    3. Upload to Qdrant in batches (auto-deduplication via stable IDs)
NEXT STEPS AFTER THIS FILE:
    1. First ensure models are trained (run train_gnn.py, train_lstm.py)
    2. Then run this: python src/backend/uploader.py
    3. Then use search_engine.py to query the populated database
================================================================================
"""
import sys
import os
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
# =====================================================================
# === STEP 1: Setup Paths and Environment ===
# =====================================================================
# This MUST happen before importing local modules
# NEXT STEP: Python can now import from src/ directory
# Add project root to sys.path so we can do "from src.backend import..."
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
# === STEP 2: Load Environment Variables ===
# Searches up to 3 parent directories for .env file
# NEXT STEP: Credentials for Qdrant Cloud are now available via os.getenv()
current_path = Path(__file__).resolve()
for _ in range(3):
    env_path = current_path / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env from: {env_path}")
        break
    current_path = current_path.parent
else:
    # Fallback to current working directory
    load_dotenv()
# =====================================================================
# === STEP 3: Import AI Models (Robust Loading) ===
# =====================================================================
# We use direct file loading to avoid Python module cache issues
# NEXT STEP: Models are now available for embedding generation
import importlib.util
def load_module_from_path(module_name, file_path):
    """
    Load a Python module directly from file path.
    
    === WHY WE DO THIS ===
    - Avoids Python module cache conflicts
    - Works even if package isn't installed
    - More reliable for development
    
    NEXT STEP: Use the returned module's classes normally
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
# === STEP 3A: Load Model 1 (GNN Encoder) ===
# Generates 64-dim structural embeddings from network topology
# NEXT STEP: GNNEncoder class is now available
models_dir = Path(project_root) / "src" / "models"
try:
    gnn_module = load_module_from_path("gnn_encoder", models_dir / "gnn_encoder.py")
    GNNEncoder = gnn_module.HeterogeneousGATEncoder
    print("✅ Loaded GNNEncoder (Model 1: Topology)")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load GNNEncoder: {e}")
    print("   NEXT STEP: Ensure src/models/gnn_encoder.py exists and is valid")
    raise e
# === STEP 3B: Load Model 2 (LSTM Encoder) ===
# Generates 64-dim temporal embeddings from delay sequences
# NEXT STEP: LSTMEncoder class is now available
try:
    lstm_module = load_module_from_path("lstm_encoder", models_dir / "cascade" / "lstm_encoder.py")
    LSTMEncoder = lstm_module.LSTMEncoder
    print("✅ Loaded LSTMEncoder (Model 2: Cascade)")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load LSTMEncoder: {e}")
    print("   NEXT STEP: Ensure src/models/cascade/lstm_encoder.py exists")
    raise e
# === STEP 3C: Load Model 3 (Semantic Encoder) with Fallback ===
# Generates 384-dim text embeddings from incident descriptions
# NEXT STEP: SemanticEncoder is available (or dummy version if import fails)
try:
    from src.models.semantic_encoder import SemanticEncoder
    SEMANTIC_AVAILABLE = True
    print("✅ Loaded SemanticEncoder (Model 3: Semantic)")
except (ImportError, ValueError) as e:
    print(f"⚠️ SemanticEncoder import failed: {e}")
    print("   Using dummy fallback (all zeros)")
    print("   NEXT STEP: Install sentence-transformers: pip install sentence-transformers")
    SEMANTIC_AVAILABLE = False
    
    # Dummy fallback class
    class SemanticEncoder:
        def encode(self, text):
            import torch
            return torch.zeros(384)
# === STEP 4: Import Database and Torch ===
# NEXT STEP: StorageManager ready for use
from src.backend.database import StorageManager
import torch
class MemoryUploader:
    """
    Orchestrates the upload of historical incidents to Qdrant.
    
    === WORKFLOW ===
    1. __init__: Connect to Qdrant, load AI models
    2. load_training_data: Load JSON, generate embeddings
    3. upload: Clean dummy data, batch upload to Qdrant
    
    === NEXT STEP ===
    Instantiate and call upload():
        uploader = MemoryUploader()
        uploader.upload()
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the uploader with database connection and AI models.
        
        === INITIALIZATION STEPS ===
        1. Connect to Qdrant Cloud
        2. Load GNN, LSTM, Semantic encoders
        3. Create collection if it doesn't exist
        
        NEXT STEP: Call load_training_data() to process incidents
        """
        print("📤 Initializing Memory Uploader...")
        
        # === STEP 1: Setup Database Connection ===
        # Ensures we use global data/ folder even if run from src/backend/
        # NEXT STEP: StorageManager is ready
        abs_data_path = str(Path(project_root) / "data")
        
        # Get Qdrant credentials from environment
        url = os.getenv("QDRANT_URL")
        key = os.getenv("QDRANT_API_KEY")
        
        if not url or not key:
            print("⚠️ WARNING: QDRANT_URL or QDRANT_API_KEY not found in .env")
            print("   NEXT STEP: Add these to your .env file before uploading")
        
        self.storage = StorageManager(
            data_dir=abs_data_path,
            qdrant_url=url,
            qdrant_api_key=key
        )
        
        # === STEP 2: Initialize AI Models ===
        # These generate the embeddings for each incident
        # NEXT STEP: Models ready for inference (not training)
        print("   Loading AI models...")
        self.gnn = GNNEncoder()
        self.lstm = LSTMEncoder()
        self.sem = SemanticEncoder()
        print("   ✓ All models loaded")
        
        # === STEP 3: Ensure Qdrant Collection Exists ===
        # Creates 'operational_memory' with triple-vector config
        # NEXT STEP: Collection ready to receive data
        print("   Initializing Qdrant collection...")
        self.storage.init_operational_memory()
        print("✅ MemoryUploader ready!")
    
    def load_training_data(self) -> List[Dict]:
        """
        Load incidents from JSON and generate embeddings.
        
        === PROCESS FLOW ===
        1. Load incidents.json (or fallback to train.json)
        2. Handle nested JSON structure if needed
        3. For each incident:
           - Generate semantic vector (384-dim)
           - Generate structural vector (64-dim)
           - Generate temporal vector (64-dim)
           - Add stable ID for deduplication
        4. Return list of incidents with embeddings
        
        === DATA FORMAT ===
        Input: {"incident_id": "...", "log": "...", "timestamp": "..."}
        Output: Same + {"embeddings": {"semantic": [...], "structural": [...], "temporal": [...]}}
        
        NEXT STEP: Pass result to upload() method
        """
        print("📂 Loading training data...")
        
        # === STEP 1: Load incidents.json ===
        # Try multiple file names for flexibility
        # NEXT STEP: If found, proceed to parsing
        incidents = self.storage.load_json("incidents.json")
        
        # Fallback to train.json if incidents.json doesn't exist
        if not incidents and hasattr(self.storage, 'get_incidents'):
            incidents = self.storage.get_incidents("train")
        
        # === STEP 2: Handle Nested JSON Structure ===
        # Some generators create {"metadata": ..., "train": [...]}
        # We need just the array of incidents
        # NEXT STEP: incidents is now a flat list
        if isinstance(incidents, dict):
            if "train" in incidents:
                incidents = incidents["train"]
                print("   Extracted 'train' key from nested structure")
            elif "incidents" in incidents:
                incidents = incidents["incidents"]
                print("   Extracted 'incidents' key from nested structure")
            # Single incident or unknown structure
            elif "incident_id" in incidents:
                incidents = [incidents]
                print("   Wrapped single incident in list")
            else:
                print("⚠️ JSON structure unrecognized (no 'train' or 'incidents' key)")
                incidents = []
        
        # === STEP 2.5: Load Golden Runs ===
        # Also upload the 50 "Golden Run" examples so they are searchable
        print("   Loading Golden Runs...")
        golden_runs = self.storage.get_golden_runs()
        if golden_runs:
            print(f"   Found {len(golden_runs)} golden runs")
            # Mark them explicitly as golden for the uploader logic
            for gr in golden_runs:
                gr['is_golden'] = True
                # Ensure they have a log/description for embedding
                if 'description' in gr and 'log' not in gr:
                    gr['log'] = gr['description']
            
            # Combine lists
            incidents.extend(golden_runs)
            print(f"   Combined total: {len(incidents)} incidents (Training + Golden)")
        else:
            print("   ⚠️ No golden runs found")
        
        # === STEP 3: Fallback to Dummy Data ===
        # If no real data exists, generate test data for pipeline verification
        # NEXT STEP: If using dummy data, run a real data generator soon
        if not incidents:
            print("⚠️ No incidents found in data/processed/")
            print("   NEXT STEP: Run python data_gen/generate_incidents.py")
            print("   Using 5 dummy incidents to test the pipeline...")
            return self._generate_dummy_data(5)
        
        print(f"   Loaded {len(incidents)} real incidents from JSON")
        
        # === STEP 4: Generate Embeddings ===
        # This is the AI "intelligence" layer
        # NEXT STEP: Each incident now has 3 vector embeddings
        processed_incidents = []
        print(f"   Generating embeddings (GNN + LSTM + Semantic)...")
        print(f"   This may take a few minutes for {len(incidents)} incidents...")
        
        for idx, inc in enumerate(incidents):
            # === STEP 4A: Type Safety Check ===
            # Skip non-dict entries (like metadata strings)
            # NEXT STEP: Process valid incident
            if not isinstance(inc, dict):
                continue
            
            # Progress indicator every 100 incidents
            if (idx + 1) % 100 == 0:
                print(f"   ... processed {idx + 1}/{len(incidents)}")
            
            # === STEP 4B: Extract Text for Semantic Encoding ===
            # Try multiple fallback keys in case schema varies
            # NEXT STEP: Text ready for Model 3
            text = inc.get('log', '') or inc.get('semantic_description', '') or str(inc)
            
            # === STEP 4C: Generate Semantic Vector (Model 3) ===
            # Encodes the incident description as 384-dim vector
            # NEXT STEP: semantic embedding ready
            sem_vec = self.sem.encode(text).tolist()
            
            # === STEP 4D: Generate Structural Vector (Model 1: GNN) ===
            # TODO: Replace dummy graph with real topology from incident
            # For now, uses random graph to test pipeline
            # NEXT STEP: Once graph_builder.py is ready, use real network topology
            dummy_graph = self._create_dummy_graph_input()
            struct_vec = self.gnn(dummy_graph, return_embedding=True).detach().numpy()[0].tolist()
            
            # === STEP 4E: Generate Temporal Vector (Model 2: LSTM) ===
            # TODO: Replace dummy sequence with real delay history
            # For now, uses random sequence to test pipeline
            # NEXT STEP: Once LSTM is trained, use real telemetry data
            dummy_seq = torch.randn(1, 10, 4)  # [batch=1, seq_len=10, features=4]
            temp_vec = self.lstm(dummy_seq).detach().numpy()[0].tolist()
            
            # === STEP 4F: Add Embeddings to Incident ===
            # Keeps all original data + adds embeddings
            # NEXT STEP: Incident ready for upload
            inc["embeddings"] = {
                "semantic": sem_vec,
                "structural": struct_vec,
                "temporal": temp_vec
            }
            
            # === STEP 4G: Generate Stable ID for Deduplication ===
            # Uses MD5 hash of timestamp + log text
            # Running uploader twice won't create duplicates
            # NEXT STEP: ID prevents duplicate uploads
            raw_id_seed = f"{inc.get('timestamp', '')}-{inc.get('log', '')}"
            stable_id = hashlib.md5(raw_id_seed.encode()).hexdigest()
            inc["incident_id"] = stable_id
            
            processed_incidents.append(inc)
        
        print(f"✅ Generated embeddings for {len(processed_incidents)} incidents")
        print(f"   NEXT STEP: Upload to Qdrant using upload() method")
        return processed_incidents
    
    def _create_dummy_graph_input(self):
        """
        Create dummy graph input for GNN (temporary until graph builder is ready).
        
        === TEMPORARY PLACEHOLDER ===
        - Real implementation should build graph from incident's affected stations
        - For now, generates random graph to test the pipeline
        
        === GRAPH STRUCTURE ===
        - 10 nodes (stations)
        - 20 edges (track segments)
        - Node features: 14-dim (from blueprint)
        - Edge features: 8-dim
        
        NEXT STEP: Replace this with real graph_builder.py output
        
        TODO (For Team):
            1. Implement src/backend/graph_builder.py
            2. Use incident['station_ids'] to extract subgraph
            3. Replace this dummy with real topology
        """
        from torch_geometric.data import Data
        
        # Random node features (14-dim per node)
        x = torch.randn(10, 14)
        
        # Random edge connections
        edge_index = torch.randint(0, 10, (2, 20))
        
        # Batch indicator (all nodes in same graph)
        batch = torch.zeros(10, dtype=torch.long)
        
        # Node type (0 = station, 1 = junction, etc.)
        node_type = torch.zeros(10, dtype=torch.long)
        
        # Edge features (8-dim per edge)
        edge_attr = torch.randn(20, 8)
        
        return Data(
            x=x,
            edge_index=edge_index,
            batch=batch,
            node_type=node_type,
            edge_attr=edge_attr
        )
    
    def _generate_dummy_data(self, count: int) -> List[Dict]:
        """
        Generate dummy test incidents for pipeline testing.
        
        === WHEN THIS IS USED ===
        - incidents.json doesn't exist yet
        - Allows testing Qdrant upload without real data
        - All dummy incidents are marked with is_dummy=True
        
        === CLEANUP ===
        - Can be safely deleted using storage.delete_dummy_data()
        - Won't affect real incident data
        
        NEXT STEP: Use for testing, then replace with real data
        """
        import uuid
        
        print(f"   Generating {count} dummy incidents for testing...")
        data = []
        
        for i in range(count):
            data.append({
                "incident_id": str(uuid.uuid4()),
                "timestamp": "2024-01-01T12:00:00",
                "log": f"Test incident {i}: Signal failure at test station",
                "meta": {"archetype": "Signal Failure", "day": "Monday"},
                "location_id": f"SEG_{i:03d}",
                "resolution": {"outcome_score": 0.9, "is_golden": i % 10 == 0},
                "is_dummy": True,  # MARKER: Allows selective deletion later
                "embeddings": {
                    "semantic": [random.random() for _ in range(384)],
                    "structural": [random.random() for _ in range(64)],
                    "temporal": [random.random() for _ in range(64)]
                }
            })
        
        print(f"   ✅ Generated {count} dummy incidents")
        print(f"   NEXT STEP: These can be deleted later with delete_dummy_data()")
        return data
    
    def upload(self, clear_dummy_only: bool = True):
        """
        Main upload orchestrator.
        
        === PROCESS FLOW ===
        1. Load incidents and generate embeddings
        2. Clean up old dummy data (if clear_dummy_only=True)
        3. Upload to Qdrant in batches
        
        === SAFETY FEATURES ===
        - Only deletes data marked as dummy (is_dummy=True)
        - Real incident data is never deleted
        - Batch uploads prevent timeouts
        - Auto-deduplication via stable IDs
        
        NEXT STEP: After this completes, run search_engine.py to test queries
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Memory Upload Process")
        print("=" * 70)
        
        # === STEP 1: Load and Process Data ===
        # Generates embeddings for all incidents
        # NEXT STEP: Incidents ready for upload
        incidents = self.load_training_data()
        
        # === STEP 2: Clean Up Old Dummy Data ===
        # Only affects test data from previous runs
        # NEXT STEP: Qdrant ready for fresh upload
        if clear_dummy_only and self.storage.client:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            print("\n🧹 Cleaning up old dummy data...")
            try:
                self.storage.client.delete(
                    collection_name="operational_memory",
                    points_selector=Filter(
                        must=[FieldCondition(
                            key="is_dummy",
                            match=MatchValue(value=True)
                        )]
                    )
                )
                print("   ✓ Dummy data removed")
            except Exception as e:
                print(f"   (Cleanup skipped: {e})")
        
        # === STEP 3: Upload to Qdrant ===
        # Batch upload with auto-deduplication
        # NEXT STEP: Check Qdrant Cloud dashboard to verify upload
        print(f"\n📤 Uploading {len(incidents)} incidents to Qdrant Cloud...")
        uploaded_count = self.storage.upload_incident_memory(incidents)
        
        # === STEP 4: Summary ===
        # NEXT STEP: Proceed to search_engine.py or integration.py
        print("\n" + "=" * 70)
        print(f"✅ Upload Complete!")
        print(f"   Total Incidents: {len(incidents)}")
        print(f"   Successfully Uploaded: {uploaded_count}")
        print(f"   Collection: operational_memory")
        print("\n   NEXT STEPS:")
        print("   1. Verify in Qdrant Cloud dashboard")
        print("   2. Test search: python src/backend/search_engine.py")
        print("   3. Run full pipeline: python src/backend/integration.py")
        print("=" * 70)
# =====================================================================
# === MAIN EXECUTION (Run this file directly) ===
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚄 Neural Rail Conductor - Memory Uploader")
    print("=" * 70)
    
    # === STEP 1: Create Uploader ===
    # NEXT STEP: Uploader connects to Qdrant and loads models
    uploader = MemoryUploader()
    
    # === STEP 2: Run Upload Process ===
    # NEXT STEP: Incidents uploaded to Qdrant Cloud
    uploader.upload()
    
    print("\n✅ All done! You can now run search_engine.py to test queries.")
"""
================================================================================
📘 DETAILED DOCUMENTATION (For Team Handoff)
================================================================================
WHAT IS THIS SCRIPT?
--------------------
This is the "Teacher" of the Neural Rail system. It:
1. Reads historical incidents from JSON files
2. Processes them through 3 AI models to generate embeddings
3. Uploads to Qdrant Cloud for future similarity search
INPUT DATA:
-----------
1. data/processed/incidents.json - Raw historical incidents
   Format: {"incident_id": "...", "timestamp": "...", "log": "..."}
2. AI Models (loaded at runtime):
   - Model 1 (GNN): src/models/gnn_encoder.py
   - Model 2 (LSTM): src/models/cascade/lstm_encoder.py
   - Model 3 (Semantic): src/models/semantic_encoder.py
HOW IT WORKS:
-------------
1. LOAD: Reads incidents.json
2. EMBED: Runs each incident through 3 encoders:
   - Text → Semantic Vector (384-dim) via MiniLM
   - Topology → Structural Vector (64-dim) via GNN
   - Timeline → Temporal Vector (64-dim) via LSTM
3. DEDUPLICATE: Generates stable MD5 ID from timestamp + log
4. UPLOAD: Sends to Qdrant in batches of 50 to avoid timeouts
OUTPUT:
-------
- Populates 'operational_memory' collection in Qdrant Cloud
- Each point contains:
  * 3 vectors (for multi-vector search)
  * Full incident payload (for display)
  * Stable ID (prevents duplicates)
CURRENT LIMITATIONS (TODO):
----------------------------
1. GNN uses dummy graph (needs graph_builder.py)
2. LSTM uses dummy sequence (needs real telemetry)
3. Both work for testing but need real inputs for production
NEXT STEPS FOR TEAM:
---------------------
1. ✅ DONE: Models are loaded (GNN, LSTM, Semantic)
2. ✅ DONE: Deduplication via stable IDs
3. ✅ DONE: Batch uploads to prevent timeouts
4. TODO: Implement graph_builder.py for real GNN input
5. TODO: Use real telemetry windows for LSTM input
6. TODO: Train models (currently using untrained weights)
WHEN TO RUN THIS:
-----------------
- After data generation (generate_incidents.py)
- After model training (train_gnn.py, train_lstm.py)
- Whenever incidents.json is updated
- NOT during live demo (this is preprocessing)
HOW TO RUN:
-----------
cd QRail
python src/backend/uploader.py
ERROR TROUBLESHOOTING:
----------------------
1. "Qdrant connection failed"
   → Check QDRANT_URL and QDRANT_API_KEY in .env
2. "incidents.json not found"
   → Run: python data_gen/generate_incidents.py
3. "Module not found"
   → Ensure you're running from project root (QRail/)
4. "Invalid vector dimensions"
   → Check model output sizes match (64, 64, 384)
================================================================================
"""