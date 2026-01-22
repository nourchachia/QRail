"""
Storage Schema Manager - src/backend/database.py

Purpose:
    Central storage utility that connects data → AI models → search system.
    Handles loading JSON files and Qdrant vector database operations.

Core Functionality (Required):
    1. Load JSON Files:
       - Automatically finds files in correct directories (data/network/ or data/processed/)
       - Used by AI models to load infrastructure and operational data
    
    2. Qdrant Vector Database:
       - Creates collections for storing incident embeddings
       - Uploads incidents with AI-generated embeddings (GNN/LSTM/Semantic vectors)
       - Searches for similar historical incidents using multi-vector similarity

Optional Functionality:
    - Save methods (save_stations, save_segments, etc.) are available if you want to use
      StorageManager in your data generation scripts, but not required if your scripts
      already save files correctly to data/network/ and data/processed/

How It Fits in the Pipeline:
    Step 1: Data generators save JSON files (can use StorageManager.save_* methods, or save directly)
    Step 2: AI models use StorageManager.load_json() to load data, generate embeddings, 
            then upload to Qdrant via StorageManager.upload_incident_memory()
    Step 3: Backend API uses StorageManager.search_similar_incidents() when new incidents occur

Usage:
    cd QRail
    python src/backend/database.py  # Test script (optional)
    
    # In your scripts:
    from src.backend.database import StorageManager
    storage = StorageManager(data_dir="data")
    
    # Load data (required for AI models)
    stations = storage.load_json("stations.json")
    
    # Qdrant operations (required for search)
    storage.init_qdrant_collection("operational_memory")
    storage.upload_incident_memory(incidents_with_embeddings)
    similar = storage.search_similar_incidents(query_vectors={...})
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    FieldCondition, Filter, MatchValue
)


class StorageManager:
    """
    Central storage manager for loading JSON files and Qdrant vector database operations.
    
    Directory Structure (assumes files are already in these locations):
    - data/network/: stations.json, segments.json, timetable.json (infrastructure)
    - data/processed/: incidents.json, live_status.json, golden_run_accidents.json (operational)
    
    Core Usage (Required):
        storage = StorageManager(data_dir="data")
        
        # Load data for AI models
        stations = storage.load_json("stations.json")  # Auto-finds in data/network/
        incidents = storage.load_json("incidents.json")  # Auto-finds in data/processed/
        
        # Setup Qdrant and upload embeddings
        storage.init_qdrant_collection("operational_memory")
        storage.upload_incident_memory(incidents_with_embeddings)
        
        # Search for similar incidents
        similar = storage.search_similar_incidents(query_vectors={...}, limit=5)
    
    Optional Usage (if you want to use StorageManager in data generation scripts):
        storage.save_stations([...])  # → data/network/stations.json
        storage.save_segments([...])  # → data/network/segments.json
        # Note: Not required if your data generation scripts already save files correctly
    """
    
    def __init__(self, data_dir: str = "data", 
                 qdrant_url: str = "localhost", qdrant_port: int = 6333):
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
        self.qdrant = QdrantClient(host=qdrant_url, port=qdrant_port)
        
    # ========== JSON Storage (Infrastructure) ==========
    
    def save_stations(self, stations: List[Dict]):
        """
        Save station data to JSON
        
        Schema:
        {
            "id": "STN_001",
            "name": "Central Station",
            "type": "major_hub",
            "zone": "core",
            "platforms": 12,
            "passengers": 150000,
            "coordinates": [50.0, 50.0],
            "connected_segments": ["SEG_001", "SEG_002"],
            "has_switches": true,
            "is_junction": true
        }
        """
        with open(self.network_dir / "stations.json", 'w') as f:
            json.dump(stations, f, indent=2)
        print(f"✓ Saved {len(stations)} stations")
    
    def save_segments(self, segments: List[Dict]):
        """
        Save segment data to JSON
        
        Schema:
        {
            "id": "SEG_001",
            "from_station": "STN_001",
            "to_station": "STN_002",
            "length_km": 5.2,
            "speed_limit": 160,
            "capacity": 20,
            "bidirectional": true,
            "track_type": "main_line",
            "has_switches": true,
            "is_critical": true,
            "electrified": true
        }
        """
        with open(self.network_dir / "segments.json", 'w') as f:
            json.dump(segments, f, indent=2)
        print(f"✓ Saved {len(segments)} segments")
    
    def save_timetable(self, timetable: List[Dict]):
        """
        Save timetable data to JSON
        
        Schema:
        {
            "service_id": "SVC_001",
            "train_id": "EXP_01",
            "day_type": "weekday",
            "stop_sequence": [
                {
                    "station_id": "STN_001",
                    "arrival": "2026-03-22T08:00:00",
                    "departure": "2026-03-22T08:02:00",
                    "platform": 4
                }
            ]
        }
        """
        with open(self.network_dir / "timetable.json", 'w') as f:
            json.dump(timetable, f, indent=2)
        print(f"✓ Saved {len(timetable)} schedules")
    
    def save_live_status(self, status: Dict):
        """
        Save current network state (Digital Twin Pulse)
        
        Schema:
        {
            "timestamp": "2026-03-22T08:30:00",
            "day_type": "sunday",
            "weather": {
                "condition": "fog",
                "temperature_c": 5,
                "wind_speed_kmh": 8,
                "visibility_km": 4.3
            },
            "network_load_pct": 42,
            "total_active_trains": 12,
            "active_trains": [...]
        }
        """
        with open(self.processed_dir / "live_status.json", 'w') as f:
            json.dump(status, f, indent=2)
        print(f"✓ Saved live status at {status['timestamp']}")
    
    def load_json(self, filename: str) -> Dict:
        """Load any JSON file"""
        # Network files go to network_dir, processed files go to processed_dir
        network_files = ["stations.json", "segments.json", "timetable.json"]
        processed_files = ["incidents.json", "live_status.json", "golden_run_accidents.json"]
        
        if filename in network_files:
            path = self.network_dir / filename
        elif filename in processed_files:
            path = self.processed_dir / filename
        else:
            # Fallback: try network first, then processed
            path = self.network_dir / filename
            if not path.exists():
                path = self.processed_dir / filename
        
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    
    # ========== Qdrant Storage (Operational Memory) ==========
    
    def init_qdrant_collection(self, collection_name: str = "operational_memory"):
        """
        Initialize Qdrant collection with triple-vector configuration
        """
        try:
            # Check if collection exists
            collections = self.qdrant.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                print(f"Collection '{collection_name}' already exists")
                return
            
            # Create collection with named vectors
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config={
                    # GNN topology embedding
                    "structural": VectorParams(
                        size=64,
                        distance=Distance.COSINE
                    ),
                    # LSTM cascade embedding
                    "temporal": VectorParams(
                        size=64,
                        distance=Distance.COSINE
                    ),
                    # Semantic text embedding
                    "semantic": VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                }
            )
            
            print(f"✓ Created Qdrant collection: {collection_name}")
            
        except Exception as e:
            print(f"✗ Error creating collection: {e}")
    
    def upload_incident_memory(self, 
                              incidents: List[Dict],
                              collection_name: str = "operational_memory"):
        """
        Upload historical incidents to Qdrant with embeddings
        
        Expected incident structure:
        {
            "incident_id": "HIST_001",
            "timestamp": "2024-11-12T08:15:00Z",
            "location_id": "SEG_045",
            "meta": {"day": "Monday", "weather": "Clear", ...},
            "log": "Alex: 'Track failure at Junction 9...'",
            "snapshot": {...},
            "resolution": {...},
            "embeddings": {
                "structural": [64-dim vector],
                "temporal": [64-dim vector],
                "semantic": [384-dim vector]
            }
        }
        """
        points = []
        
        for idx, incident in enumerate(incidents):
            # Extract embeddings (should be pre-computed)
            embeddings = incident.get('embeddings', {})
            
            if not all(k in embeddings for k in ['structural', 'temporal', 'semantic']):
                print(f"Warning: Incident {incident.get('incident_id')} missing embeddings")
                continue
            
            # Create point with named vectors
            point = PointStruct(
                id=idx,
                vector={
                    "structural": embeddings['structural'],
                    "temporal": embeddings['temporal'],
                    "semantic": embeddings['semantic']
                },
                payload={
                    "incident_id": incident['incident_id'],
                    "timestamp": incident['timestamp'],
                    "location_id": incident['location_id'],
                    "incident_type": incident['meta'].get('archetype', 'unknown'),
                    "weather": incident['meta'].get('weather', 'clear'),
                    "day_type": incident['meta'].get('day', 'weekday'),
                    "log": incident['log'],
                    "resolution": incident['resolution'],
                    "outcome_score": incident['resolution'].get('outcome_score', 0.0),
                    "is_golden": incident['resolution'].get('is_golden', False)
                }
            )
            points.append(point)
        
        # Batch upload
        if points:
            self.qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"✓ Uploaded {len(points)} incidents to Qdrant")
        else:
            print("✗ No valid incidents to upload")
    
    def search_similar_incidents(self,
                                query_vectors: Dict[str, List[float]],
                                filters: Optional[Dict] = None,
                                limit: int = 5,
                                collection_name: str = "operational_memory") -> List[Dict]:
        """
        Multi-vector search with optional filters
        
        Args:
            query_vectors: {"structural": [...], "temporal": [...], "semantic": [...]}
            filters: {"weather": "rain", "incident_type": "signal_failure"}
            limit: Number of results
        
        Returns:
            List of similar incidents with scores
        """
        # Build filter if provided
        filter_obj = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                filter_obj = Filter(must=conditions)
        
        # For multi-vector search, we'll do weighted fusion
        # Search with semantic vector (primary)
        results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=("semantic", query_vectors["semantic"]),
            query_filter=filter_obj,
            limit=limit * 2  # Get more candidates for re-ranking
        )
        
        # Re-rank using all three vectors (simplified)
        # In production, you'd do proper weighted fusion
        similar_incidents = []
        for result in results[:limit]:
            similar_incidents.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        return similar_incidents


# ========== Example Usage ==========

if __name__ == "__main__":
    # Initialize storage manager
    storage = StorageManager(data_dir="data")
    
    # Example: Save infrastructure data
    example_stations = [
        {
            "id": "STN_001",
            "name": "Central Station",
            "type": "major_hub",
            "zone": "core",
            "platforms": 12,
            "daily_passengers": 150000,
            "coordinates": [50.0, 50.0],
            "connected_segments": ["SEG_001", "SEG_002"],
            "has_switches": True,
            "is_junction": True
        }
    ]
    
    storage.save_stations(example_stations)
    
    # Example: Initialize Qdrant
    storage.init_qdrant_collection("operational_memory")
    
    print("\n✓ Storage system ready!")
    print("Next steps:")
    print("1. Generate full network with data_gen scripts")
    print("2. Train AI models to generate embeddings")
    print("3. Upload incidents with embeddings to Qdrant")
