"""
LSTM Telemetry Service
Bridges raw JSON data (live_status.json, incidents.json) with LSTM model.
Integrates with DataFuelPipeline for feature extraction.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryEnricher:
    def __init__(
        self, 
        live_status_path: str = "data/processed/live_status.json",
        segments_path: str = "data/network/segments.json",
        stations_path: str = "data/network/stations.json",
        sequence_length: int = 10
    ):
        """
        Enricher that converts raw JSON into LSTM-ready tensors.
        
        Args:
            live_status_path: Path to real-time network state
            segments_path: Path to segment infrastructure data (for speed limits)
            stations_path: Path to station data (for hub detection)
            sequence_length: Number of timesteps in sequence (default 10)
        """
        self.live_status_path = Path(live_status_path)
        self.segments_path = Path(segments_path)
        self.stations_path = Path(stations_path)
        self.sequence_length = sequence_length
        
        # Data caches
        self._live_cache = None
        self._segments_map = {}
        self._stations_map = {}
        self._hub_stations = set()
        
        # Feature scaling constants (from blueprint)
        self.MAX_DELAY = 60  # minutes (1 hour)
        self.MAX_SPEED = 160  # km/h (main line max)
        
        self._load_infrastructure()
    
    def _load_infrastructure(self):
        """Load segment and station metadata for feature enrichment."""
        try:
            # Load segments
            if self.segments_path.exists():
                with open(self.segments_path, 'r') as f:
                    segments = json.load(f)
                    self._segments_map = {seg['id']: seg for seg in segments}
                logger.info(f"✅ Loaded {len(self._segments_map)} segments")
            
            # Load stations
            if self.stations_path.exists():
                with open(self.stations_path, 'r') as f:
                    stations = json.load(f)
                    self._stations_map = {stn['id']: stn for stn in stations}
                    # Identify hub stations
                    self._hub_stations = {
                        stn['id'] for stn in stations 
                        if stn.get('type') in ['major_hub', 'regional']
                    }
                logger.info(f"✅ Loaded {len(self._stations_map)} stations ({len(self._hub_stations)} hubs)")
        
        except Exception as e:
            logger.warning(f"⚠️ Infrastructure data not loaded: {e}")
    
    def load_live_data(self):
        """Reload live status data (call this periodically in production)."""
        if not self.live_status_path.exists():
            raise FileNotFoundError(f"Live status not found: {self.live_status_path}")
        
        with open(self.live_status_path, 'r') as f:
            self._live_cache = json.load(f)
        
        logger.info(f"✅ Loaded live status: {len(self._live_cache.get('active_trains', []))} active trains")
    
    def get_train_telemetry(self, train_id: str) -> torch.Tensor:
        """
        Extract telemetry for a single train (inference mode).
        
        Returns:
            Tensor of shape [1, sequence_length, 4] ready for LSTM
        """
        if not self._live_cache:
            self.load_live_data()
        
        # Find train in live data
        train_data = next(
            (t for t in self._live_cache.get('active_trains', []) 
             if t['train_id'] == train_id), 
            None
        )
        
        if not train_data:
            logger.warning(f"⚠️ Train {train_id} not found. Returning zero tensor.")
            return torch.zeros((1, self.sequence_length, 4))
        
        # Extract and normalize sequence
        telemetry_window = train_data.get('telemetry_window', [])
        sequence = self._normalize_sequence(telemetry_window)
        
        return torch.tensor([sequence], dtype=torch.float32)
    
    def get_batch_telemetry(self, train_ids: List[str]) -> torch.Tensor:
        """
        Extract telemetry for multiple trains (batch inference).
        
        Returns:
            Tensor of shape [batch_size, sequence_length, 4]
        """
        sequences = []
        for train_id in train_ids:
            seq_tensor = self.get_train_telemetry(train_id)
            sequences.append(seq_tensor.squeeze(0))
        
        return torch.stack(sequences)
    
    def extract_from_incident(self, incident: Dict) -> torch.Tensor:
        """
        Extract telemetry from historical incident data (training mode).
        Uses the 'telemetry_window' or 'history' field from incidents.json.
        
        Args:
            incident: Dictionary containing incident metadata
        
        Returns:
            Tensor of shape [1, sequence_length, 4]
        """
        # Try different possible keys (depends on data generator version)
        telemetry_data = (
            incident.get('telemetry_window') or
            incident.get('history') or
            incident.get('target_train', {}).get('history', [])
        )
        
        if not telemetry_data:
            logger.warning(f"⚠️ No telemetry in incident {incident.get('incident_id')}")
            return torch.zeros((1, self.sequence_length, 4))
        
        sequence = self._normalize_sequence(telemetry_data)
        return torch.tensor([sequence], dtype=torch.float32)
    
    def _normalize_sequence(self, window: List[Dict]) -> List[List[float]]:
        """
        Converts raw JSON dicts into normalized [delay, progress, speed, hub] vectors.
        
        Feature Engineering (Blueprint Section 3.3):
        - Feature 0: Delay (minutes) -> Scaled to [0, 1] via /MAX_DELAY
        - Feature 1: Progress (0.0-1.0) -> Already normalized
        - Feature 2: Speed Limit (km/h) -> Scaled to [0, 1] via /MAX_SPEED
        - Feature 3: Is Hub (binary) -> 0 or 1
        """
        normalized = []
        
        for step in window:
            # Extract raw values
            delay = float(step.get('delay', 0))
            progress = float(step.get('pos', step.get('progress_pct', 0.0)))
            segment_id = step.get('segment_id')
            station_id = step.get('station_id')
            
            # Feature 0: Normalized delay
            normalized_delay = min(delay / self.MAX_DELAY, 1.0)
            
            # Feature 1: Progress (already 0-1)
            normalized_progress = max(0.0, min(progress, 1.0))
            
            # Feature 2: Speed limit from segment data
            speed_limit = self.MAX_SPEED  # Default
            if segment_id and segment_id in self._segments_map:
                speed_limit = self._segments_map[segment_id].get('speed_limit', self.MAX_SPEED)
            normalized_speed = speed_limit / self.MAX_SPEED
            
            # Feature 3: Hub proximity
            is_hub = 1.0 if station_id in self._hub_stations else 0.0
            
            vec = [normalized_delay, normalized_progress, normalized_speed, is_hub]
            normalized.append(vec)
        
        # Handle sequence length mismatch
        if len(normalized) < self.sequence_length:
            # Pad at beginning (train just started)
            padding = [[0.0, 0.0, 0.0, 0.0]] * (self.sequence_length - len(normalized))
            normalized = padding + normalized
        elif len(normalized) > self.sequence_length:
            # Take last N samples
            normalized = normalized[-self.sequence_length:]
        
        return normalized
    
    def extract_lstm_sequence(self, incident_data: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Main integration point with DataFuelPipeline.
        Extracts LSTM features + metadata for training.
        
        Returns:
            (tensor, metadata) where tensor is [1, 10, 4] and metadata contains labels
        """
        tensor = self.extract_from_incident(incident_data)
        
        metadata = {
            'incident_id': incident_data.get('incident_id'),
            'cascade_depth': incident_data.get('cascade_depth', 0),
            'estimated_delay': incident_data.get('estimated_delay_minutes', 0),
            'outcome_score': incident_data.get('outcome_score', 0.0),
            'is_expanding': self._detect_trend(tensor.squeeze(0))
        }
        
        return tensor, metadata
    
    def _detect_trend(self, sequence: torch.Tensor) -> bool:
        """
        Analyzes if delay is expanding (True) or contracting (False).
        Simple heuristic for data validation.
        """
        delays = sequence[:, 0].numpy()  # Extract delay column
        if len(delays) < 3:
            return False
        
        # Compare first half vs second half
        mid = len(delays) // 2
        avg_early = np.mean(delays[:mid])
        avg_late = np.mean(delays[mid:])
        
        return avg_late > avg_early


# Usage Examples
if __name__ == "__main__":
    print("=" * 60)
    print("LSTM Telemetry Service Test")
    print("=" * 60)
    
    enricher = TelemetryEnricher()
    
    # Auto-detect available trains
    try:
        enricher.load_live_data()
        available_trains = [
            t['train_id'] for t in enricher._live_cache.get('active_trains', [])
        ]
        
        if len(available_trains) == 0:
            print("\n⚠️  No active trains found in live_status.json")
            print("   Using mock data for testing...\n")
            available_trains = None
        else:
            print(f"\n✅ Found {len(available_trains)} active trains:")
            for train_id in available_trains[:3]:  # Show first 3
                print(f"   - {train_id}")
            print()
    
    except Exception as e:
        print(f"\n⚠️  Could not load live data: {e}")
        print("   Using mock data for testing...\n")
        available_trains = None
    
    # Test 1: Live data extraction
    print("[Test 1] Live Train Telemetry")
    if available_trains:
        # Use first real train
        test_train = available_trains[0]
        tensor = enricher.get_train_telemetry(test_train)
        print(f"✅ Train {test_train}")
        print(f"   Shape: {tensor.shape}")
        print(f"   First timestep features: {tensor[0, 0].tolist()}")
        print(f"   Last timestep features: {tensor[0, -1].tolist()}")
        
        # Show what each feature means
        print("\n   Feature Breakdown (last timestep):")
        print(f"   [0] Normalized Delay: {tensor[0, -1, 0].item():.3f} (x60 = {tensor[0, -1, 0].item()*60:.1f} min)")
        print(f"   [1] Progress: {tensor[0, -1, 1].item():.3f} (0-1 through segment)")
        print(f"   [2] Speed Limit: {tensor[0, -1, 2].item():.3f} (x160 = {tensor[0, -1, 2].item()*160:.0f} km/h)")
        print(f"   [3] Is Hub: {tensor[0, -1, 3].item():.0f} (0=no, 1=yes)")
    else:
        # Fallback test
        tensor = enricher.get_train_telemetry("EXP_001")
        print(f"⚠️  Using fallback (no real data)")
        print(f"   Shape: {tensor.shape}")
    
    # Test 2: Batch extraction
    print("\n[Test 2] Batch Processing")
    if available_trains and len(available_trains) >= 3:
        batch_trains = available_trains[:3]
        batch = enricher.get_batch_telemetry(batch_trains)
        print(f"✅ Batch shape: {batch.shape}")
        print(f"   Processing trains: {', '.join(batch_trains)}")
    else:
        print("⚠️  Not enough trains for batch test")
    
    # Test 3: Historical incident extraction
    print("\n[Test 3] Training Data Extraction")
    mock_incident = {
        'incident_id': 'TEST_001',
        'telemetry_window': [
            {
                'delay': i*2, 
                'pos': i*0.1, 
                'segment_id': 'SEG_001', 
                'station_id': 'STN_001' if i == 5 else None
            }
            for i in range(10)
        ],
        'cascade_depth': 3,
        'outcome_score': 0.85
    }
    
    tensor, metadata = enricher.extract_lstm_sequence(mock_incident)
    print(f"✅ Tensor shape: {tensor.shape}")
    print(f"   Metadata: {metadata}")
    print(f"   Trend: {'Expanding ⬆️' if metadata['is_expanding'] else 'Contracting ⬇️'}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)