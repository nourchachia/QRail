r"""
Data-to-Intelligence Mapping System - src/backend/feature_extractor.py

Purpose:
    Converts raw operational data (JSON files) into feature vectors ready for AI model inputs.
    Maps data according to Section 2.6 of the blueprint - each model gets specific data sources.

What It Does:
    1. Loads infrastructure data (stations, segments, timetable) from data/network/
    2. Loads operational data (live_status, incidents) from data/processed/
    3. Extracts features for each AI model:
       - GNN (Model 1): Graph structure (nodes, edges, global context)
       - LSTM (Model 2): Time-series sequences (delay history)
       - Semantic (Model 3): Natural language text descriptions
       - Conflict Classifier (Model 4): Combined context features
       - Outcome Predictor (Model 5): Context + action features

How It Fits in the Pipeline:
    Step 1: Data generation scripts create JSON files → saved to data/network/ and data/processed/
    Step 2: FeatureExtractor loads JSON files and converts them to model-ready features
    Step 3: AI models receive these features as input

How to Run:
    =========
    1. Make sure you're in the QRail project root directory:
       cd [your QRail project path]
    
    2. Ensure data files exist:
       - data/network/stations.json
       - data/network/segments.json
       - data/network/timetable.json
    
    3. Run the test script:
       python src/backend/feature_extractor.py
    
    4. In your code, import and use:
       from src.backend.feature_extractor import DataFuelPipeline
       pipeline = DataFuelPipeline(data_dir="data")
       features = pipeline.extract_gnn_features(incident)

Note on Real-World Usage:
    In real incidents, many stations can be affected (e.g., power outage at a hub affects 10+ stations).
    The extract_gnn_features() method will extract features for ALL stations listed in 
    incident['location']['station_ids'] or incident['station_ids']. 
    So if an incident affects 15 stations, you'll get 15 nodes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class DataMapping:
    """Defines how raw data flows into each AI model"""
    
    # Model 1: GNN (Topology Encoder)
    gnn_sources = [
        "stations.json",      # Node features
        "segments.json",      # Edge features
        "constraints.json"    # Infrastructure rules
    ]
    
    # Model 2: LSTM (Cascade Encoder)
    lstm_sources = [
        "timetable.json",     # Planned schedules
        "live_status.json"    # Real-time telemetry
    ]
    
    # Model 3: Semantic Encoder
    semantic_sources = [
        "incident.log",       # Alex's text descriptions
        "operational_memory"  # Historical operator logs
    ]
    
    # Model 4: Conflict Classifier
    conflict_sources = [
        "live_status.json",   # Track occupancy
        "timetable.json",     # Schedule conflicts
        "constraints.json"    # Capacity limits
    ]
    
    # Model 5: Outcome Predictor
    outcome_sources = [
        "operational_memory", # Historical actions
        "resolution.json",    # Past outcomes
        "live_status.json"    # Current context (weather, load)
    ]


class DataFuelPipeline:
    """
    Converts raw operational data into model-ready features.
    
    Automatically loads files from correct directories:
    - data/network/: stations.json, segments.json, timetable.json
    - data/processed/: live_status.json, incidents.json
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # INTELLIGENT PATH FINDING 🧠
        # If absolute path doesn't exist, try resolving relative to this script
        if not self.data_dir.exists() or not (self.data_dir / "network").exists():
            script_path = Path(__file__).resolve()
            # Try 1: ../../../data (Standard project structure)
            candidate_1 = script_path.parent.parent.parent / "data"
            # Try 2: CW/data (Current working directory)
            candidate_2 = Path.cwd() / "data"
            
            if candidate_1.exists() and (candidate_1 / "network").exists():
                print(f"   ✓ Auto-corrected data path to: {candidate_1}")
                self.data_dir = candidate_1
            elif candidate_2.exists() and (candidate_2 / "network").exists():
                print(f"   ✓ Auto-corrected data path to: {candidate_2}")
                self.data_dir = candidate_2
        
        # Network data directory (infrastructure)
        self.network_dir = self.data_dir / "network"
        
        # Processed data directory (operational)
        self.processed_dir = self.data_dir / "processed"
        
        # Load infrastructure data
        self.stations = self._load_json("stations.json")
        self.segments = self._load_json("segments.json")
        self.timetable = self._load_json("timetable.json")
        
    def _load_json(self, filename: str) -> Any:
        """Load JSON file from correct directory (network or processed)"""
        # Network files
        network_files = ["stations.json", "segments.json", "timetable.json"]
        # Processed files
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
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle both list and dict returns
                return data if data else ([] if filename.endswith('s.json') else {})
        except FileNotFoundError:
            print(f"Warning: {filename} not found at {path}. Using empty structure.")
            # Return empty list for plural files, empty dict for singular
            return [] if filename.endswith('s.json') else {}
        except json.JSONDecodeError as e:
            print(f"Warning: {filename} is not valid JSON: {e}. Using empty structure.")
            return [] if filename.endswith('s.json') else {}
    
    def extract_gnn_features(self, incident: Dict) -> Dict[str, Any]:
        """
        Extract graph features for GNN (Model 1)
        
        Note: Extracts features for ALL affected stations listed in incident.
        Handles both old format (incident['location']['station_ids']) and 
        new format (incident['station_ids']).
        
        In real incidents, this can be many stations (e.g., power outage at hub affects 10-15 stations).
        The method focuses on the incident area, not the entire network. This is correct behavior - 
        the model needs to focus on the problem area.
        
        Examples:
            - Small incident: 2 stations affected → 2 nodes
            - Medium incident: 5-10 stations affected → 5-10 nodes
            - Large incident: 15+ stations affected → 15+ nodes
        
        Returns:
            nodes: List of station features (all affected stations from incident)
            edges: List of segment connections (all segments connected to affected stations)
            global_features: Network-wide context
        """
        # Handle both old and new incident formats
        if 'location' in incident and isinstance(incident['location'], dict):
            # Old format: incident['location']['station_ids']
            affected_stations = incident['location'].get('station_ids', [])
        else:
            # New format: incident['station_ids']
            affected_stations = incident.get('station_ids', [])
        
        # Fallback: if still empty, try to get from location_id
        if not affected_stations and 'location_id' in incident:
            affected_stations = [incident['location_id']]
        
        # DEBUG: Print what we are looking for
        print(f"   🔍 FeatureExtractor: Looking for stations {affected_stations} in {len(self.stations)} loaded stations")

        # Node features (10-dim per station)
        nodes = []
        for station in self.stations:
            if station['id'] in affected_stations:
                feature_vec = [
                    1,  # is_affected
                    station.get('platforms', 0),
                    station.get('daily_passengers', 0) / 100000,  # Normalized
                    1 if station.get('is_junction', False) else 0,
                    1 if station.get('zone') == 'core' else 0,
                    station.get('coordinates', [0, 0])[0] / 100,  # Normalized x
                    station.get('coordinates', [0, 0])[1] / 100,  # Normalized y
                    len(station.get('connected_segments', [])),
                    1 if station.get('has_switches', False) else 0,
                    0  # Reserved for sensor health
                ]
                nodes.append({
                    'id': station['id'],
                    'features': feature_vec
                })
        
        # Edge features (segment attributes)
        edges = []
        for segment in self.segments:
            if (segment.get('from_station') in affected_stations or 
                segment.get('to_station') in affected_stations):
                edge_vec = [
                    segment.get('speed_limit', 0) / 200,  # Normalized
                    segment.get('capacity', 0) / 20,
                    1 if segment.get('bidirectional', True) else 0,
                    1 if segment.get('is_critical', False) else 0,
                    segment.get('length_km', 0) / 50  # Normalized
                ]
                edges.append({
                    'from': segment['from_station'],
                    'to': segment['to_station'],
                    'features': edge_vec
                })
        
        # Global context
        global_features = [
            incident.get('network_load_pct', 0) / 100,
            1 if incident.get('is_peak', False) else 0,
            incident.get('hour_of_day', 0) / 24,
            len(affected_stations) / 50
        ]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'global_features': global_features
        }
    
    def extract_lstm_sequence(self, train_id: str, 
                             history_window: int = 10) -> List[List[float]]:
        """
        Extract time-series features for LSTM (Model 2)
        
        Returns:
            sequence: [10, 4] array of [delay, progress, speed, hub_status]
        """
        # This would read from live_status.json in production
        # For now, return mock structure
        sequence = []
        
        for step in range(history_window):
            feature_vec = [
                0,      # delay_min (to be filled from telemetry)
                0.0,    # progress_pct through segment
                120,    # current_speed_limit
                0       # is_at_hub (0 or 1)
            ]
            sequence.append(feature_vec)
        
        return sequence
    
    def extract_semantic_text(self, incident: Dict) -> str:
        """
        Extract natural language description for Semantic Encoder (Model 3)
        
        Returns:
            text: Operator's description of the incident
        """
        # Use semantic_description if available (new format)
        if 'semantic_description' in incident:
            return incident['semantic_description']
        
        # Otherwise, construct from structured data (old format)
        incident_type = incident.get('type', 'unknown').replace('_', ' ').title()
        
        # Handle both old and new location formats
        if 'location' in incident and isinstance(incident['location'], dict):
            location = incident['location'].get('zone', 'unknown')
        else:
            location = incident.get('zone', 'unknown')
        
        # Handle severity (can be int or string)
        severity = incident.get('severity_level', incident.get('severity', 'unknown'))
        
        # Handle weather (new field name: weather_condition)
        weather = incident.get('weather_condition', incident.get('weather', 'clear'))
        
        description = (
            f"{incident_type} at {location} zone. "
            f"Severity: {severity}. Weather: {weather}. "
            f"{incident.get('trains_affected_count', 0)} trains affected."
        )
        
        return description
    
    def extract_conflict_features(self, incident: Dict) -> List[float]:
        """
        Extract features for Conflict Classifier (Model 4)
        
        Returns:
            features: Combined vector for MLP
        """
        # This combines GNN output + LSTM output + Context
        # The actual implementation happens after those models run
        # Here we prepare the "context" part
        
        # Handle weather (new field: weather_condition)
        weather = incident.get('weather_condition', incident.get('weather', 'clear'))
        
        # Handle location (both old and new formats)
        if 'location' in incident and isinstance(incident['location'], dict):
            is_junction = incident['location'].get('is_junction', False)
            station_ids = incident['location'].get('station_ids', [])
        else:
            is_junction = incident.get('is_junction', False)
            station_ids = incident.get('station_ids', [])
        
        context_features = [
            incident.get('network_load_pct', 0) / 100,
            1 if incident.get('is_peak', False) else 0,
            incident.get('hour_of_day', 0) / 24,
            1 if weather == 'rain' else 0,
            1 if weather == 'storm' else 0,
            incident.get('cascade_depth', 0) / 5,
            len(station_ids) / 10,
            1 if is_junction else 0
        ]
        
        return context_features
    
    def extract_outcome_context(self, incident: Dict, 
                               resolution: Dict) -> List[float]:
        """
        Extract features for Outcome Predictor (Model 5)
        
        Returns:
            features: Context + Action vector
        """
        # Handle weather (new field: weather_condition)
        weather = incident.get('weather_condition', incident.get('weather', 'clear'))
        
        # Weather penalty
        weather_penalty = {
            'clear': 0.0,
            'rain': 0.1,
            'storm': 0.3,
            'snow': 0.4
        }.get(weather, 0.0)
        
        # Action complexity
        action_complexity = len(resolution.get('actions', resolution.get('actions_taken', []))) / 10
        
        # Get resolution action (handle different field names)
        action = resolution.get('action', resolution.get('resolution_strategy', '')).lower()
        
        features = [
            incident.get('network_load_pct', 0) / 100,
            1 if incident.get('is_peak', False) else 0,
            weather_penalty,
            action_complexity,
            incident.get('trains_affected_count', 0) / 20,
            1 if 'reroute' in action else 0,
            1 if 'hold' in action or 'dwell' in action else 0,
            1 if 'cancel' in action else 0
        ]
        
        return features

    def extract_all_features(self, incident: Dict) -> Dict[str, Any]:
        """
        Wrapper to extract all features at once.
        Called by integration.py
        """
        # Extract train_id or use default
        train_id = incident.get('train_id', 'T001')
        
        # 1. GNN Features (Topology)
        gnn_data = self.extract_gnn_features(incident)
        # Add num_nodes convenience field for logging
        gnn_data['num_nodes'] = len(gnn_data['nodes'])
        
        # 2. LSTM Features (Temporal)
        lstm_data = self.extract_lstm_sequence(train_id)
        
        # 3. Semantic Text (Content)
        semantic_text = self.extract_semantic_text(incident)
        
        # 4. Conflict Context (Relational)
        conflict_data = self.extract_conflict_features(incident)
        
        return {
            'gnn': {
                'node_features': [n['features'] for n in gnn_data['nodes']],
                'edge_index': [[e['from'], e['to']] for e in gnn_data['edges']], # Simplified for now
                'num_nodes': len(gnn_data['nodes'])
            },
            'lstm': lstm_data,
            'semantic_text': semantic_text,
            'conflict': conflict_data
        }


# Example usage
if __name__ == "__main__":
    """
    Test script output explanation:
    
    When you run this script, it demonstrates feature extraction for all 5 AI models.
    
    Expected Output:
    ================
    === GNN Features ===
    Nodes: 2                    # Number of affected stations (STN_001, STN_002)
                                # NOTE: Only shows affected stations, not all 50 stations!
                                # This is correct - GNN focuses on the incident area
    Edges: 11                   # Number of segments connecting those affected stations
                                # NOTE: Only shows segments connected to affected stations
    
    === LSTM Sequence ===
    Sequence shape: [10, 4]    # 10 time steps, 4 features per step
                               # Features: [delay, progress, speed, hub_status]
    
    === Semantic Text ===
    Description: Signal Failure at core zone. Severity: 4. Weather: clear. 6 trains affected.
                               # Natural language description for semantic encoder
    
    === Conflict Features ===
    Feature vector: [0.85, 1.0, 0.33, 0.0, 0.0, 0.0, 0.2, 1.0]
                               # 8 context features: [load_pct, is_peak, hour, rain, storm, cascade, stations, junction]
    
    Important Notes:
    - Nodes/Edges count only includes AFFECTED stations/segments (from incident location)
    - If you have 50 stations but incident affects 2, you'll see Nodes: 2 (this is correct!)
    - If JSON files don't exist, you'll see warnings and Nodes/Edges will be 0
    - This is a test script using mock data to demonstrate the feature extraction process
    """
    
    # Initialize pipeline
    pipeline = DataFuelPipeline(data_dir="data")
    
    # Mock incident for testing (using NEW format matching actual incident schema)
    test_incident = {
        'type': 'signal_failure',
        'station_ids': ['STN_001', 'STN_002'],  # New format
        'is_junction': True,
        'zone': 'core',
        'severity_level': 4,  # Changed from 'severity': 'high'
        'hour_of_day': 8,
        'is_peak': True,
        'weather_condition': 'clear',  # Changed from 'weather'
        'network_load_pct': 85,
        'trains_affected_count': 6,
        'cascade_depth': 0
    }
    
    # Extract features for each model
    print("=== GNN Features ===")
    gnn_data = pipeline.extract_gnn_features(test_incident)
    print(f"Nodes: {len(gnn_data['nodes'])}")
    print(f"Edges: {len(gnn_data['edges'])}")
    
    print("\n=== LSTM Sequence ===")
    lstm_seq = pipeline.extract_lstm_sequence('T42')
    print(f"Sequence shape: [{len(lstm_seq)}, {len(lstm_seq[0])}]")
    
    print("\n=== Semantic Text ===")
    text = pipeline.extract_semantic_text(test_incident)
    print(f"Description: {text}")
    
    print("\n=== Conflict Features ===")
    conflict_vec = pipeline.extract_conflict_features(test_incident)
    print(f"Feature vector: {conflict_vec}")
    
    # Test with actual incident format
    print("\n=== Testing with Real Incident Format ===")
    real_incident = {
        "incident_id": "test-123",
        "type": "passenger_alarm",
        "station_ids": ["STN_014"],
        "zone": "mid",
        "is_junction": True,
        "severity_level": 5,
        "weather_condition": "snow",
        "network_load_pct": 81,
        "trains_affected_count": 10,
        "cascade_depth": 4,
        "hour_of_day": 7,
        "is_peak": True,
        "semantic_description": "Emergency alarm activated on train E682 at Forest Station."
    }
    
    print("\nReal Incident Semantic Text:")
    print(pipeline.extract_semantic_text(real_incident))