"""
Neural Rail Conductor - Data Auditor
Verifies compliance with Blueprint Sections 2.6 (Mapping) and 2.7 (Schema).
Ensures the generators produce data ready for the AI models.
"""

import json
from pathlib import Path

def audit_data():
    # Correctly identify project root from data/data_auditor.py
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root
    
    print("\n🔍 Auditing Data Compliance (Sections 2.6 & 2.7)...")
    print("-" * 50)
    
    # 1. Audit Stations (Section 2.7 Schema)
    stations_path = data_dir / "data/network/stations.json"
    if stations_path.exists():
        with open(stations_path, 'r') as f:
            try:
                stations = json.load(f)
                if not stations:
                    print("  ❌ stations.json: EMPTY FILE")
                else:
                    sample = stations[0]
                    # Blueprint 2.7 expected subset
                    required = {"id", "name", "type", "daily_passengers", "coordinates","zone","platforms", "has_switches", "is_junction", "connected_segments"}
                    missing = required - set(sample.keys())
                    if not missing:
                        print(f"  ✅ stations.json: COMPLIANT. ({len(stations)} stations)")
                    else:
                        print(f"  ❌ stations.json: NON-COMPLIANT. Missing: {missing}")
                        print(f"     Found keys: {list(sample.keys())}")
            except json.JSONDecodeError:
                print("  ❌ stations.json: INVALID JSON FORMAT")
    else:
        print(f"  ❌ stations.json: NOT FOUND at {stations_path.relative_to(data_dir) if data_dir in stations_path.parents else stations_path}")
                
    # 2. Audit Segments (Section 2.7 Schema)
    segments_path = data_dir / "data/network/segments.json"
    if segments_path.exists():
        with open(segments_path, 'r') as f:
            try:
                segments = json.load(f)
                if not segments:
                    print("  ❌ segments.json: EMPTY FILE")
                else:
                    sample = segments[0]
                    # Blueprint 2.7: check 'from' / 'to'
                    if "from_station" in sample and "to_station" in sample:
                         print(f"  ✅ segments.json: COMPLIANT. ({len(segments)} segments)")
                    else:
                         print(f"  ❌ segments.json: NON-COMPLIANT (Missing 'from_station' or 'to_station' fields)")
                         print(f"     Found keys: {list(sample.keys())}")
            except json.JSONDecodeError:
                print("  ❌ segments.json: INVALID JSON FORMAT")
    else:
        print(f"  ❌ segments.json: NOT FOUND")

    # 3. Audit Intelligence Mapping (Section 2.6)
    # Check if we have the history trace for Model 2 (LSTM)
    potential_paths = [
        data_dir / "data/processed/incidents.json", 
    ]
    
    incident_path = next((p for p in potential_paths if p.exists()), None)
            
    if incident_path:
        rel_disp = incident_path.relative_to(project_root) if project_root in incident_path.parents else incident_path.name
        print(f"  📂 Found incidents at: {rel_disp}")
        with open(incident_path, 'r') as f:
            try:
                incidents_raw = json.load(f)
                
                # Support user's dictionary structure with 'train' key
                if isinstance(incidents_raw, dict):
                    incidents_list = incidents_raw.get("train") or incidents_raw.get("incidents") or incidents_raw.get("data")
                    if incidents_list is None:
                        print(f"  ❌ incidents: Found dictionary with keys {list(incidents_raw.keys())} but no 'train' or 'incidents' list found.")
                        return
                else:
                    incidents_list = incidents_raw
            except Exception as e:
                print(f"  ❌ incidents: Error - {str(e)}")
    else:
        print(f"  ❌ incidents: NOT FOUND in suspected locations.")
    print("-" * 50)

if __name__ == "__main__":
    audit_data()
