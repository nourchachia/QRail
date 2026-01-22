"""
Neural Rail Conductor - Data Auditor
Verifies compliance with Blueprint Sections 2.6 (Mapping) and 2.7 (Schema).
Ensures the generators produce data ready for the AI models.
"""

import json
from pathlib import Path

def audit_data():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root
    
    print("🔍 Auditing Data Compliance (Sections 2.6 & 2.7)...")
    
    # 1. Audit Stations (Section 2.7 Schema)
    stations_path = data_dir / "data/network/stations.json"
    if stations_path.exists():
        with open(stations_path, 'r') as f:
            stations = json.load(f)
            sample = stations[0]
            required = {"id", "name", "type", "passengers", "coordinates"}
            missing = required - set(sample.keys())
            if not missing:
                print(f"  ✅ stations.json: COMPLIANT. Keys: {list(sample.keys())}")
            else:
                print(f"  ❌ stations.json: NON-COMPLIANT. Missing: {missing}")
                
    # 2. Audit Segments (Section 2.7 Schema)
    segments_path = data_dir / "data/network/segments.json"
    if segments_path.exists():
        with open(segments_path, 'r') as f:
            segments = json.load(f)
            sample = segments[0]
            # Blueprint 2.7 says 'from' and 'to', check Mapping logic
            if "from" in sample and "to" in sample:
                 print(f"  ✅ segments.json: COMPLIANT (Found 'from' and 'to').")
            else:
                 print(f"  ❌ segments.json: NON-COMPLIANT (Missing 'from'/'to'). Keys: {list(sample.keys())}")

    # 3. Audit Intelligence Mapping (Section 2.6)
    # Check if we have the history trace for Model 2 (LSTM)
    train_path = data_dir / "data/processed/incidents/train.json"
    if train_path.exists():
        with open(train_path, 'r') as f:
            incidents = json.load(f)
            sample = incidents[0]
            # Does it have the 10-step window?
            window = sample.get("snapshot", {}).get("telemetry_window_raw", [])
            if len(window) == 10 and len(window[0]) == 4:
                print(f"  ✅ Model 2 (LSTM) Fuel: COMPLIANT. Found [10x4] telemetry window.")
            else:
                print(f"  ❌ Model 2 (LSTM) Fuel: NON-COMPLIANT. Window shape: {len(window)}")

if __name__ == "__main__":
    audit_data()
