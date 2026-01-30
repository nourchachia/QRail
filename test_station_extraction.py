"""
Debug: Test station extraction logic on a golden run
"""
import json
import re

# Load one golden run
with open('d:/QRail/data/processed/golden_runs_accidents_enhanced.json', 'r') as f:
    data = json.load(f)
    
golden_runs = data['golden_runs']
test_incident = golden_runs[0]  # INC_001

print("Testing station extraction on INC_001:")
print(f"Incident ID: {test_incident.get('incident_id')}")
print(f"Location: {test_incident.get('location')}")

# Run the extraction logic
station_ids = []
incident = test_incident

if 'station_ids' in incident:
    station_ids = incident['station_ids']
    print(f"Found station_ids directly: {station_ids}")
elif 'location' in incident and isinstance(incident['location'], dict):
    # Check for station_ids array inside location  
    station_ids = incident['location'].get('station_ids', [])
    print(f"Checked location.station_ids: {station_ids}")
    
    # FIX: Golden runs use from_station/to_station format
    if not station_ids:
        from_stn = incident['location'].get('from_station')
        to_stn = incident['location'].get('to_station')
        station_ids = [s for s in [from_stn, to_stn] if s]
        print(f"Extracted from from_station/to_station: {station_ids}")
elif 'location_id' in incident:
    station_ids = [incident['location_id']]
    print(f"Found location_id: {station_ids}")

# Fallback: extract STN_XXX from text
if not station_ids:
    text_to_search = incident.get('log', '') or incident.get('description', '')
    if text_to_search:
        matches = re.findall(r'STN_\d+', text_to_search)
        station_ids = list(set(matches))
        print(f"Regex fallback: {station_ids}")

print(f"\nFINAL station_ids: {station_ids}")
print(f"Expected: ['STN_001', 'STN_002']")
print(f"Match: {set(station_ids) == {'STN_001', 'STN_002'}}")
