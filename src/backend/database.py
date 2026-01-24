"""
Neural Rail Conductor - Storage Manager (Part 4.1)
Handles JSON data loading and Qdrant Cloud integration.

This is the central hub for:
1. Loading static infrastructure (stations, segments)
2. Loading processed data (incidents, live status)
3. Connecting to Qdrant Cloud for vector operations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import Qdrant - graceful fallback if not installed
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️ qdrant-client not installed. Vector operations disabled.")


class StorageManager:
    """
    Central data access layer for Neural Rail Conductor.
    
    Handles two storage systems:
    1. JSON Files (Local): Static infrastructure & historical incidents
    2. Qdrant Cloud: Vector embeddings for semantic search
    """
    
    def __init__(self, data_dir: str = "data", 
                 qdrant_url: str = "localhost", 
                 qdrant_port: int = 6333,
                 qdrant_api_key: Optional[str] = None):
        """
        Initialize the StorageManager.
        
        Args:
            base_path: Root path to the data folder. Auto-detected if None.
        """
        # Base data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Network data directory (infrastructure)
        self.network_dir = self.data_dir / "network"
        self.network_dir.mkdir(parents=True, exist_ok=True)
        
        # Processed data directory (operational)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client
        # Fix: Support both local (host/port) and Cloud (url/api_key)
        self.client = None
        if QDRANT_AVAILABLE:
            if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
                 self.client = QdrantClient(host=qdrant_url, port=qdrant_port)
            else:
                 # Cloud connection
                 self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Alias for compatibility with some scripts
        self.qdrant = self.client
        
        # Initialize if client exists
        if self.client:
            self._init_qdrant()
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Any] = {}
        
        # Compat: self.base_path for newer scripts
        self.base_path = self.data_dir
    
    def _init_qdrant(self):
        """
        Initialize Qdrant Cloud connection using environment variables.
        """
        if not self.client:
            return
            
        try:
            # Test connection
            self.client.get_collections()
            # print(f"✅ Connected to Qdrant")
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            self.client = None
    
    # ==================== JSON FILE OPERATIONS ====================
    
    def load_json(self, filename: str) -> Optional[Any]:
        """
        Load a JSON file from the data directory.
        
        Args:
            filename: Relative path like "network/stations.json" or "processed/train.json"
        
        Returns:
            Parsed JSON data or None if not found
        """
        # Check cache first
        if filename in self._cache:
            return self._cache[filename]
        
        # Try different base paths
        possible_paths = [
            self.base_path / filename,
            self.network_dir / filename,
            self.processed_dir / filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._cache[filename] = data
                        return data
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse error in {path}: {e}")
                    return None
        
        # print(f"⚠️ File not found: {filename}")
        return None
    
    def save_json(self, filename: str, data: Any) -> bool:
        """
        Save data to a JSON file.
        
        Args:
            filename: Target path like "processed/incidents.json"
            data: Data to save (must be JSON-serializable)
        
        Returns:
            True if successful
        """
        if "network" in filename:
            path = self.network_dir / Path(filename).name
        else:
            path = self.processed_dir / Path(filename).name
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # Update cache
            self._cache[filename] = data
            return True
        except Exception as e:
            print(f"❌ Failed to save {path}: {e}")
            return False
            
    # Compatibility Wrappers
    def save_stations(self, data): self.save_json("stations.json", data)
    def save_segments(self, data): self.save_json("segments.json", data)
    def save_timetable(self, data): self.save_json("timetable.json", data)
    def save_live_status(self, data): self.save_json("live_status.json", data)
    
    # ==================== CONVENIENCE GETTERS ====================
    
    def get_stations(self) -> List[Dict]:
        """Load stations from network/stations.json"""
        data = self.load_json("network/stations.json")
        return data if isinstance(data, list) else []
    
    def get_segments(self) -> List[Dict]:
        """Load segments from network/segments.json"""
        data = self.load_json("network/segments.json")
        return data if isinstance(data, list) else []
    
    def get_timetable(self) -> List[Dict]:
        """Load timetable from network/timetable.json"""
        data = self.load_json("network/timetable.json")
        return data if isinstance(data, list) else []
    
    def get_incidents(self, split: str = "train") -> List[Dict]:
        """
        Load incidents from processed folder.
        
        Args:
            split: "train" (800 items) or "test" (200 items)
        """
        data = self.load_json(f"processed/{split}.json")
        return data if isinstance(data, list) else []
    
    def get_golden_runs(self) -> List[Dict]:
        """Load the 50 curated 'perfect resolution' examples"""
        data = self.load_json("processed/golden_runs.json")
        return data if isinstance(data, list) else []
    
    def get_live_status(self) -> Optional[Dict]:
        """Load real-time network state from live_status.json"""
        return self.load_json("processed/live_status.json")
    
    # ==================== QDRANT OPERATIONS ====================
    
    def init_operational_memory(self) -> bool:
        """
        Create the 'operational_memory' collection in Qdrant.
        """
        success = self.init_qdrant_collection()
        if success and self.client:
             # Ensure index exists for cleanup logic
             try:
                 from qdrant_client.models import PayloadSchemaType
                 self.client.create_payload_index(
                     collection_name="operational_memory",
                     field_name="is_dummy",
                     field_schema=PayloadSchemaType.BOOL
                 )
             except Exception:
                 pass # Index might already exist
        return success

    def init_qdrant_collection(self, collection_name: str = "operational_memory") -> bool:
        """
        Initialize Qdrant collection with triple-vector configuration
        """
        if not self.client:
            print("❌ Qdrant client not initialized")
            return False
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                # print(f"ℹ️ Collection '{collection_name}' already exists")
                return True
            
            # Create new collection with triple-vector config
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    # Model 1: GNN (Topology)
                    "structural": VectorParams(size=64, distance=Distance.COSINE),
                    # Model 2: LSTM (Cascade)
                    "temporal": VectorParams(size=64, distance=Distance.COSINE),
                    # Model 3: Semantic (Transformer)
                    "semantic": VectorParams(size=384, distance=Distance.COSINE),
                }
            )
            print(f"✅ Created collection '{collection_name}' with triple-vector config")
            
            # Indexing FIX: Create index for 'is_dummy' to allow filtered deletion
            from qdrant_client.models import PayloadSchemaType
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="is_dummy",
                field_schema=PayloadSchemaType.BOOL
            )
            return True
            
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            return False
    
    def upload_incident_memory(self, incidents: List[Dict], collection_name: str = "operational_memory", batch_size: int = 50) -> int:
        """
        Upload incidents with embeddings to Qdrant using BATCHING to prevent timeouts.
        """
        if not self.client:
            print("❌ Qdrant client not initialized")
            return 0
        
        import uuid
        total_uploaded = 0
        all_points = []
        
        for inc in incidents:
            # Support both 'vectors' and 'embeddings' keys
            vecs = inc.get("vectors") or inc.get("embeddings")
            if not vecs:
                continue
            
            point_id = inc.get("incident_id")
            if point_id is None:
                point_id = str(uuid.uuid4())
            
            payload = inc.copy()
            if "vectors" in payload: del payload["vectors"]
            if "embeddings" in payload: del payload["embeddings"]
            
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
            print("⚠️ No valid points to upload")
            return 0
        
        # BATCHING FIX: Upload in smaller chunks
        print(f"   Batching upload ({len(all_points)} points, {batch_size} per batch)...")
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i : i + batch_size]
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                total_uploaded += len(batch)
                print(f"   ✓ Uploaded {total_uploaded}/{len(all_points)}...")
            except Exception as e:
                print(f"❌ Batch upload failed at index {i}: {e}")
                break
                
        return total_uploaded

    def search_similar_incidents(
        self,
        vectors: Dict[str, List[float]],
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar historical incidents.
        """
        pass # Not implementing full search here, relying on search_engine.py

# ==================== STANDALONE TEST ====================

if __name__ == "__main__":
    print("=" * 50)
    print("🚄 Neural Rail Conductor - StorageManager Test")
    print("=" * 50)
    
    # Load credentials for test
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    
    # Initialize with cloud creds if available
    if url and key:
        print(f"☁️  Testing with Qdrant Cloud: {url[:30]}...")
        storage = StorageManager(qdrant_url=url, qdrant_api_key=key)
    else:
        print("🏠 Testing with Localhost (Default)")
        storage = StorageManager()
    
    # Test JSON loading
    stations = storage.get_stations()
    print(f"\n📍 Stations loaded: {len(stations)}")
    
    if storage.client:
        print(f"\n✅ Qdrant connected!")
    else:
        print(f"\n⚠️ Qdrant not configured")
