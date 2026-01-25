"""
================================================================================
Neural Rail Conductor - Storage Manager (Enhanced Version with Step Comments)
================================================================================
ROLE: Central Data Hub
    - Manages JSON files (stations, segments, incidents)
    - Connects to Qdrant Cloud for vector operations
    - Provides deduplication & batch upload
USAGE:
    from src.backend.database import StorageManager
    
    storage = StorageManager(
        data_dir="data",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )
NEXT STEPS AFTER THIS FILE:
    1. Run this standalone to test connection: python src/backend/database.py
    2. Then use in uploader.py to populate Qdrant
    3. Then use in search_engine.py to query vectors
================================================================================
"""
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
# === STEP 1: Load Environment Variables ===
# Searches for .env file in project root
# NEXT STEP: Make sure .env contains QDRANT_URL and QDRANT_API_KEY
load_dotenv()
# === STEP 2: Import Qdrant Client (with Graceful Fallback) ===
# If qdrant-client isn't installed, we disable vector operations
# NEXT STEP: Run `pip install qdrant-client` if you see the warning below
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue, PayloadSchemaType
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️ qdrant-client not installed. Vector operations disabled.")
class StorageManager:
    """
    Central data access layer for Neural Rail Conductor.
    
    === WHAT IT DOES ===
    1. Loads JSON files (stations, segments, incidents) from data/ folder
    2. Connects to Qdrant Cloud for vector storage
    3. Uploads incident embeddings in batches
    4. Provides deduplication using stable IDs
    
    === NEXT STEP AFTER INIT ===
    Call storage.init_operational_memory() to set up Qdrant collection
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        qdrant_url: Optional[str] = None,
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize the StorageManager.
        
        === INITIALIZATION STEPS ===
        1. Set up data directory structure
        2. Connect to Qdrant (local or cloud)
        3. Test connection
        4. Initialize cache
        
        === NEXT STEP ===
        After this runs successfully, call init_operational_memory()
        """
        
        # === STEP 1A: Set Up Data Directory ===
        # Creates data/, data/network/, data/processed/ if they don't exist
        # NEXT STEP: Generate network files using data_gen/generate_network.py
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.network_dir = self.data_dir / "network"
        self.network_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Compatibility alias for older scripts
        self.base_path = self.data_dir
        
        # === STEP 1B: Initialize Cache ===
        # Stores frequently accessed JSON files in memory
        # NEXT STEP: First call to load_json() will populate this
        self._cache: Dict[str, Any] = {}
        
        # === STEP 2: Connect to Qdrant ===
        # Tries env variables first, then passed parameters
        # NEXT STEP: Check console for connection success/failure message
        self.client = None
        self.qdrant = None  # Alias for backward compatibility
        
        if not QDRANT_AVAILABLE:
            print("⚠️ Qdrant client not available. Run: pip install qdrant-client")
            return
        
        # Get credentials from environment if not provided
        env_url = os.getenv("QDRANT_URL")
        env_key = os.getenv("QDRANT_API_KEY")
        
        final_url = qdrant_url if qdrant_url else (env_url or "localhost")
        final_key = qdrant_api_key if qdrant_api_key else env_key
        
        # === STEP 2A: Determine Connection Type (Local vs Cloud) ===
        # Local: Uses host/port (no API key needed)
        # Cloud: Uses URL + API key
        # NEXT STEP: Verify connection by checking console output
        try:
            if "localhost" in final_url or "127.0.0.1" in final_url:
                # Local Qdrant (Docker or binary)
                self.client = QdrantClient(host=final_url, port=qdrant_port)
                print(f"✅ Connected to local Qdrant at {final_url}:{qdrant_port}")
            else:
                # Qdrant Cloud
                if not final_key:
                    raise ValueError("QDRANT_API_KEY required for cloud connection!")
                self.client = QdrantClient(url=final_url, api_key=final_key)
                print(f"✅ Connected to Qdrant Cloud at {final_url[:40]}...")
            
            # === STEP 2B: Test Connection ===
            # Tries to get collections list to verify connectivity
            # NEXT STEP: If this fails, check your URL and API key in .env
            self.client.get_collections()
            self.qdrant = self.client  # Backup alias
            
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            print("   NEXT STEP: Check QDRANT_URL and QDRANT_API_KEY in .env file")
            self.client = None
            self.qdrant = None
    
    # =====================================================================
    # JSON FILE OPERATIONS (No Qdrant Required)
    # =====================================================================
    
    def load_json(self, filename: str) -> Optional[Any]:
        """
        Load a JSON file from data directory.
        
        === SEARCH ORDER ===
        1. Check cache (fastest)
        2. Check data/processed/
        3. Check data/network/
        4. Check data/ (root)
        
        === NEXT STEP ===
        After loading, data is cached. Second calls will be instant.
        
        Args:
            filename: e.g., "stations.json" or "network/stations.json"
        
        Returns:
            Parsed JSON data or None if not found
        """
        # === STEP 1: Check Cache ===
        # If we've loaded this file before, return immediately
        # NEXT STEP: None - cache hit means we're done!
        if filename in self._cache:
            return self._cache[filename]
        
        # === STEP 2: Try Multiple Locations ===
        # Files might be in different subdirectories depending on how they were created
        # NEXT STEP: If file not found, generate it using data_gen/ scripts
        possible_paths = [
            self.processed_dir / filename,
            self.network_dir / filename,
            self.data_dir / filename,
            # Also try with basename in case full path was passed
            self.processed_dir / Path(filename).name,
            self.network_dir / Path(filename).name,
            self.data_dir / Path(filename).name,
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # === STEP 3: Cache for Next Time ===
                        self._cache[filename] = data
                        return data
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse error in {path}: {e}")
                    print("   NEXT STEP: Fix JSON syntax or regenerate the file")
                    return None
        
        # Not found in any location
        return None
    
    def save_json(self, filename: str, data: Any) -> bool:
        """
        Save data to a JSON file.
        
        === AUTO-ROUTING ===
        - Files with "network" in name → data/network/
        - Everything else → data/processed/
        
        === NEXT STEP ===
        After saving, file is automatically cached. Call load_json() to retrieve.
        
        Args:
            filename: Target filename (e.g., "stations.json")
            data: JSON-serializable data
        
        Returns:
            True if successful, False otherwise
        """
        # === STEP 1: Determine Target Directory ===
        # Network infrastructure files go in network/
        # Operational data goes in processed/
        # NEXT STEP: File will be created automatically
        if "network" in filename or filename in ["stations.json", "segments.json", "timetable.json"]:
            path = self.network_dir / Path(filename).name
        else:
            path = self.processed_dir / Path(filename).name
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # === STEP 2: Write to Disk ===
        # Uses indent=2 for human readability
        # NEXT STEP: File is now available via load_json(filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # === STEP 3: Update Cache ===
            self._cache[filename] = data
            print(f"✅ Saved {path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save {path}: {e}")
            print("   NEXT STEP: Check file permissions or disk space")
            return False
    
    # === Convenience Wrappers ===
    # NEXT STEP: Use these instead of save_json() for clarity
    def save_stations(self, data): return self.save_json("stations.json", data)
    def save_segments(self, data): return self.save_json("segments.json", data)
    def save_timetable(self, data): return self.save_json("timetable.json", data)
    def save_live_status(self, data): return self.save_json("live_status.json", data)
    
    # =====================================================================
    # DATA GETTERS (Convenience Methods)
    # =====================================================================
    
    def get_stations(self) -> List[Dict]:
        """
        Load stations from network/stations.json.
        
        NEXT STEP: If empty, run: python data_gen/generate_network.py
        """
        data = self.load_json("network/stations.json")
        return data if isinstance(data, list) else []
    
    def get_segments(self) -> List[Dict]:
        """
        Load segments from network/segments.json.
        
        NEXT STEP: If empty, run: python data_gen/generate_network.py
        """
        data = self.load_json("network/segments.json")
        return data if isinstance(data, list) else []
    
    def get_timetable(self) -> List[Dict]:
        """
        Load timetable from network/timetable.json.
        
        NEXT STEP: If empty, generate using timetable generator
        """
        data = self.load_json("network/timetable.json")
        return data if isinstance(data, list) else []
    
    def get_incidents(self, split: str = "train") -> List[Dict]:
        """
        Load incidents from processed folder.
        
        Supports two formats:
        1. Separate files: train.json, test.json
        2. Combined file: incidents.json with {"train": [...], "test": [...]}
        
        NEXT STEP: If empty, run: python data_gen/generate_incidents.py
        
        Args:
            split: "train" (800 items) or "test" (200 items)
        """
        # Try dedicated file first
        data = self.load_json(f"processed/{split}.json")
        
        # Fallback to combined incidents.json
        if not data:
            incidents_data = self.load_json("processed/incidents.json")
            if isinstance(incidents_data, dict):
                data = incidents_data.get(split, [])
        
        return data if isinstance(data, list) else []
    
    def get_golden_runs(self) -> List[Dict]:
        """
        Load the 50 curated 'perfect resolution' examples.
        
        NEXT STEP: If empty, run: python data_gen/generate_golden_runs.py
        """
        # Try multiple possible filenames
        for filename in ["processed/golden_runs_accidents.json", "processed/golden_runs.json"]:
            data = self.load_json(filename)
            if data:
                return data if isinstance(data, list) else []
        return []
    
    def get_live_status(self) -> Optional[Dict]:
        """
        Load real-time network state from live_status.json.
        
        NEXT STEP: If empty, run: python data_gen/live_status_generator.py
        """
        return self.load_json("processed/live_status.json")
    
    # =====================================================================
    # QDRANT OPERATIONS (Vector Database)
    # =====================================================================
    
    def init_operational_memory(self) -> bool:
        """
        Initialize Qdrant collection for operational memory.
        
        === WHAT THIS DOES ===
        1. Creates 'operational_memory' collection with 3 vector types
        2. Sets up payload indexes for fast filtering
        3. Prepares for batch uploads
        
        === NEXT STEP ===
        After this succeeds, run uploader.py to populate with incidents
        
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            print("⚠️ Qdrant client not initialized. Skipping collection setup.")
            print("   NEXT STEP: Make sure QDRANT_URL and QDRANT_API_KEY are set")
            return False
        
        # === STEP 1: Create Collection ===
        # NEXT STEP: Check console for "Created collection" message
        success = self.init_qdrant_collection()
        
        if success:
            # === STEP 2: Create Payload Indexes ===
            # These speed up filtering during search
            # NEXT STEP: None - indexes are automatic
            try:
                self.client.create_payload_index(
                    collection_name="operational_memory",
                    field_name="is_dummy",
                    field_schema=PayloadSchemaType.BOOL
                )
                
                self.client.create_payload_index(
                    collection_name="operational_memory",
                    field_name="is_golden_run",
                    field_schema=PayloadSchemaType.BOOL
                )
                
                self.client.create_payload_index(
                    collection_name="operational_memory",
                    field_name="archetype",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print("✅ Payload indexes created")
                
            except Exception:
                pass  # Indexes may already exist - not a problem
        
        return success
    
    def init_qdrant_collection(self, collection_name: str = "operational_memory") -> bool:
        """
        Create Qdrant collection with triple-vector configuration.
        
        === VECTOR SCHEMA ===
        - structural: 64-dim (GNN topology encoding from Model 1)
        - temporal: 64-dim (LSTM cascade encoding from Model 2)
        - semantic: 384-dim (Transformer text encoding from Model 3)
        
        === NEXT STEP ===
        After creation, use upload_incident_memory() to add data
        
        Args:
            collection_name: Name of collection (default: "operational_memory")
        
        Returns:
            True if collection exists or was created successfully
        """
        if not self.client:
            print("❌ Qdrant client not initialized")
            print("   NEXT STEP: Fix Qdrant connection in __init__")
            return False
        
        try:
            # === STEP 1: Check if Collection Already Exists ===
            # If it exists, we're done!
            # NEXT STEP: If exists, proceed to upload data
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                print(f"ℹ️ Collection '{collection_name}' already exists")
                return True
            
            # === STEP 2: Create New Collection ===
            # Uses cosine distance for all three vector types
            # NEXT STEP: Collection ready for data upload
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "structural": VectorParams(size=64, distance=Distance.COSINE),
                    "temporal": VectorParams(size=64, distance=Distance.COSINE),
                    "semantic": VectorParams(size=384, distance=Distance.COSINE),
                }
            )
            print(f"✅ Created collection '{collection_name}' with triple-vector config")
            
            # === STEP 3: Create is_dummy Index for Cleanup ===
            # Allows selective deletion of test data
            # NEXT STEP: uploader.py can now safely clean up dummy data
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="is_dummy",
                field_schema=PayloadSchemaType.BOOL
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            print("   NEXT STEP: Check Qdrant Cloud dashboard or restart local Qdrant")
            return False
    
    def upload_incident_memory(
        self,
        incidents: List[Dict],
        collection_name: str = "operational_memory",
        batch_size: int = 50
    ) -> int:
        """
        Upload incidents with embeddings to Qdrant.
        
        === FEATURES ===
        ✅ Automatic deduplication using stable IDs
        ✅ Batch processing to avoid timeouts
        ✅ Supports both 'embeddings' and 'vectors' keys
        ✅ Validates vector dimensions
        
        === NEXT STEP ===
        After upload, use search_engine.py to query similar incidents
        
        Args:
            incidents: List of dicts with 'embeddings' key
            collection_name: Target collection
            batch_size: Points per batch (default 50)
        
        Returns:
            Number of points successfully uploaded
        """
        if not self.client:
            print("❌ Qdrant client not initialized. Cannot upload.")
            print("   NEXT STEP: Fix Qdrant connection first")
            return 0
        
        total_uploaded = 0
        all_points = []
        skipped_count = 0
        
        # === STEP 1: Convert Incidents to Qdrant Points ===
        # NEXT STEP: Points will be uploaded in batches below
        for inc in incidents:
            # Get vectors (support both naming conventions)
            vecs = inc.get("vectors") or inc.get("embeddings")
            if not vecs:
                skipped_count += 1
                continue
            
            # === STEP 1A: Validate Vector Dimensions ===
            # Must be exactly 64, 64, 384 to match collection config
            # NEXT STEP: If invalid, check your encoder output sizes
            if not self._validate_vectors(vecs):
                print(f"⚠️ Skipping incident with invalid vector dimensions")
                print(f"   Expected: structural=64, temporal=64, semantic=384")
                print(f"   Got: structural={len(vecs.get('structural', []))}, "
                      f"temporal={len(vecs.get('temporal', []))}, "
                      f"semantic={len(vecs.get('semantic', []))}")
                skipped_count += 1
                continue
            
            # === STEP 1B: Generate Stable ID for Deduplication ===
            # Same incident → same ID → upsert overwrites duplicate
            # NEXT STEP: ID is now ready for use
            point_id = self._generate_stable_id(inc)
            
            # === STEP 1C: Separate Payload from Vectors ===
            # Qdrant stores vectors separately from metadata
            # NEXT STEP: Point is ready to add to batch
            payload = {k: v for k, v in inc.items() if k not in ["vectors", "embeddings"]}
            
            # Ensure incident_id exists in payload
            if "incident_id" not in payload:
                payload["incident_id"] = point_id
            
            # === STEP 1D: Create Qdrant Point ===
            # NEXT STEP: Point added to batch
            all_points.append(PointStruct(
                id=point_id,
                vector={
                    "structural": vecs.get("structural", [0.0] * 64),
                    "temporal": vecs.get("temporal", [0.0] * 64),
                    "semantic": vecs.get("semantic", [0.0] * 384),
                },
                payload=payload
            ))
        
        if not all_points:
            print(f"⚠️ No valid points to upload (skipped: {skipped_count})")
            print("   NEXT STEP: Check that incidents have 'embeddings' key with correct dimensions")
            return 0
        
        # === STEP 2: Upload in Batches ===
        # Prevents timeout errors on large datasets
        # NEXT STEP: Check console for progress updates
        print(f"📤 Uploading {len(all_points)} points in batches of {batch_size}...")
        
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i : i + batch_size]
            
            try:
                # Use upsert (not insert) to allow overwriting duplicates
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                total_uploaded += len(batch)
                print(f"   ✓ Batch {i // batch_size + 1}: {len(batch)} points ({total_uploaded}/{len(all_points)})")
                
            except Exception as e:
                print(f"❌ Batch upload failed at index {i}: {e}")
                print(f"   NEXT STEP: Check Qdrant Cloud storage limits or network connection")
                break
        
        print(f"✅ Upload complete: {total_uploaded} points (skipped: {skipped_count})")
        print(f"   NEXT STEP: Run search_engine.py to test similarity search")
        return total_uploaded
    
    def _generate_stable_id(self, incident: Dict) -> str:
        """
        Generate a stable, deterministic ID for deduplication.
        
        === LOGIC ===
        1. Use existing incident_id if present
        2. Otherwise generate MD5 hash from timestamp + log text
        
        === WHY THIS MATTERS ===
        - Running uploader.py twice won't create duplicates
        - Same incident always gets same ID → upsert overwrites old version
        
        NEXT STEP: Use this ID as Qdrant point ID
        """
        # Use existing incident_id if present
        if "incident_id" in incident and incident["incident_id"]:
            return str(incident["incident_id"])
        
        # Generate hash from content
        content = f"{incident.get('timestamp', '')}-{incident.get('log', '')}-{incident.get('semantic_description', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _validate_vectors(self, vecs: Dict) -> bool:
        """
        Validate that vector dimensions match expected sizes.
        
        === EXPECTED DIMENSIONS ===
        - structural: 64 (from GNN)
        - temporal: 64 (from LSTM)
        - semantic: 384 (from MiniLM)
        
        NEXT STEP: If this returns False, check your model output dimensions
        """
        structural = vecs.get("structural", [])
        temporal = vecs.get("temporal", [])
        semantic = vecs.get("semantic", [])
        
        return (
            len(structural) == 64 and
            len(temporal) == 64 and
            len (semantic) == 384
        )
    
    def delete_dummy_data(self) -> bool:
        """
        Delete all points marked as dummy/test data.
        
        === SAFE CLEANUP ===
        - Only deletes points with payload.is_dummy = true
        - Real incident data is never touched
        
        NEXT STEP: After cleanup, upload real data using uploader.py
        
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            self.client.delete(
                collection_name="operational_memory",
                points_selector=Filter(
                    must=[FieldCondition(
                        key="is_dummy",
                        match=MatchValue(value=True)
                    )]
                )
            )
            print("✅ Dummy data cleaned up")
            print("   NEXT STEP: Upload real incidents using uploader.py")
            return True
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
            return False
# =====================================================================
# STANDALONE TEST (Run this file directly to test)
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 StorageManager Test")
    print("=" * 70)
    
    # === STEP 1: Initialize ===
    # NEXT STEP: Check console for connection success
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    
    if url and key:
        print(f"☁️ Testing with Qdrant Cloud: {url[:40]}...")
        storage = StorageManager(qdrant_url=url, qdrant_api_key=key)
    else:
        print("🏠 Testing with Localhost (Default)")
        storage = StorageManager()
    
    # === STEP 2: Test JSON Operations ===
    # NEXT STEP: If counts are 0, run data generators
    print("\n📁 JSON Operations:")
    stations = storage.get_stations()
    print(f"   Stations: {len(stations)}")
    
    segments = storage.get_segments()
    print(f"   Segments: {len(segments)}")
    
    incidents = storage.get_incidents("train")
    print(f"   Train Incidents: {len(incidents)}")
    
    # === STEP 3: Test Qdrant Connection ===
    # NEXT STEP: If connected, run uploader.py to populate
    print("\n🔗 Qdrant Connection:")
    if storage.client:
        print("   ✅ Connected to Qdrant")
        
        # Initialize collection
        storage.init_operational_memory()
        
        # Get collection info
        try:
            info = storage.client.get_collection("operational_memory")
            print(f"   Collection points: {info.points_count}")
            print(f"   NEXT STEP: Run uploader.py to add more data")
        except:
            print("   Collection not yet populated")
            print("   NEXT STEP: Run uploader.py to add incidents")
    else:
        print("   ⚠️ Qdrant not available")
        print("   NEXT STEP: Set QDRANT_URL and QDRANT_API_KEY in .env")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)