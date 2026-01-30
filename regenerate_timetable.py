"""
Regenerate timetable.json with valid routes that match segment connectivity.
This ensures trains can actually travel the routes defined in the timetable.
"""

import json
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import defaultdict, deque
import random
from datetime import datetime, timedelta

# Paths
DATA_DIR = Path(__file__).parent / "data" / "network"
SEGMENTS_FILE = DATA_DIR / "segments.json"
STATIONS_FILE = DATA_DIR / "stations.json"
TIMETABLE_FILE = DATA_DIR / "timetable.json"

# Load data
print("📂 Loading network data...")
with open(SEGMENTS_FILE) as f:
    segments = json.load(f)

with open(STATIONS_FILE) as f:
    stations = json.load(f)

station_ids = {s['id'] for s in stations}
print(f"   ✓ {len(stations)} stations loaded")
print(f"   ✓ {len(segments)} segments loaded")

# Build adjacency graph from segments
print("\n🔗 Building network connectivity graph...")
graph = defaultdict(set)
for segment in segments:
    from_stn = segment['from_station']
    to_stn = segment['to_station']
    
    # Add both directions since segments are bidirectional
    graph[from_stn].add(to_stn)
    if segment['bidirectional']:
        graph[to_stn].add(from_stn)

print(f"   ✓ Graph built with {len(graph)} connected stations")

# Find valid paths between stations using BFS
def find_path(start: str, end: str, max_length: int = 6) -> Optional[List[str]]:
    """Find a valid path from start to end station."""
    if start == end:
        return [start]
    if start not in graph or end not in graph:
        return None
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        if len(path) > max_length:
            continue
        
        for neighbor in graph[current]:
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None

# Define train services with realistic route patterns
print("\n🚆 Generating train services...")
train_services = []

# Group stations by type for realistic routing
major_hubs = [s for s in stations if s.get('type') == 'major_hub']
regional = [s for s in stations if s.get('type') == 'regional']
local = [s for s in stations if s.get('type') == 'local']
minor = [s for s in stations if s.get('type') == 'minor_halt']

print(f"   Station distribution: {len(major_hubs)} hubs, {len(regional)} regional, {len(local)} local, {len(minor)} minor")

# Service templates with realistic names and patterns
service_templates = [
    {
        'prefix': 'EXP',
        'type': 'express',
        'base_stations': major_hubs,
        'count': 12,
        'max_stops': 4,
        'route_pattern': 'Hub-to-Hub routes'
    },
    {
        'prefix': 'REG',
        'type': 'regional',
        'base_stations': regional,
        'count': 10,
        'max_stops': 5,
        'route_pattern': 'Regional connections'
    },
    {
        'prefix': 'LOCAL',
        'type': 'local',
        'base_stations': local,
        'count': 15,
        'max_stops': 6,
        'route_pattern': 'Local stopping service'
    },
    {
        'prefix': 'HALT',
        'type': 'local',
        'base_stations': local + minor,
        'count': 10,
        'max_stops': 4,
        'route_pattern': 'Halt-to-Halt service'
    }
]

train_id_counters = defaultdict(int)
service_count = 0

# Time period multipliers (more trains during peak hours)
PEAK_MORNING = (7, 9)      # 7-9 AM: 1.8x frequency
PEAK_EVENING = (16, 18)    # 4-6 PM: 1.8x frequency
OFF_PEAK = (1, 5)          # 1-5 AM: 0.3x frequency
NIGHT_REDUCTION = (22, 6)  # 10 PM - 6 AM: reduced

def get_time_multiplier(hour: int) -> float:
    """Get frequency multiplier based on time of day."""
    if PEAK_MORNING[0] <= hour < PEAK_MORNING[1]:
        return 8.0  # Heavy morning rush
    elif PEAK_EVENING[0] <= hour < PEAK_EVENING[1]:
        return 8.0  # Heavy evening rush
    elif hour >= 22 or hour < 6:
        return 0.2  # Some trains overnight (ensures 1-3 trains)
    else:
        return 1.0  # Steady daytime baseline

def get_daytype_multiplier(day_type: str) -> float:
    """Get frequency multiplier based on day type."""
    if day_type == 'weekday':
        return 1.0  # Full frequency on weekdays
    elif day_type == 'weekend':
        return 0.7  # Reduced on weekends
    elif day_type == 'holiday':
        return 0.5  # Much reduced on holidays
    return 1.0

for template in service_templates:
    prefix = template['prefix']
    service_type = template['type']
    base_stations = template['base_stations']
    target_count = template['count']
    max_stops = template['max_stops']
    
    # Create trains for each day type with different frequencies
    for day_type in ['weekday', 'weekend', 'holiday']:
        day_multiplier = get_daytype_multiplier(day_type)
        day_target = max(2, int(target_count * day_multiplier))  # At least 2 trains per day type
        
        attempts = 0
        created = 0
        
        while created < day_target and attempts < day_target * 5:
            attempts += 1
            
            # Pick random start and end stations
            start = random.choice(base_stations)
            end = random.choice([s for s in base_stations if s['id'] != start['id']])
            
            # Find path between them
            path = find_path(start['id'], end['id'], max_length=max_stops)
            
            if not path or len(path) < 2:
                continue
            
            # Limit path length
            if len(path) > max_stops:
                path = path[:max_stops]
            
            # Create stops from path
            stops = []
            
            # Pick departure times with weighted density
            hours = list(range(24))
            weights = [get_time_multiplier(h) for h in hours]
            base_time = random.choices(hours, weights=weights, k=1)[0]
            
            current_time_minutes = base_time * 60 + random.randint(0, 30)
            
            for i, station_id in enumerate(path):
                station = next((s for s in stations if s['id'] == station_id), None)
                if not station:
                    break
                
                arrival_minutes = current_time_minutes
                departure_minutes = arrival_minutes + (2 if i < len(path) - 1 else 0)  # 2 min dwell
                
                # Handle times that go past midnight
                arrival_hour = arrival_minutes // 60
                if arrival_hour >= 24:
                    arrival_hour -= 24
                arrival_minute = arrival_minutes % 60
                
                departure_hour = departure_minutes // 60
                if departure_hour >= 24:
                    departure_hour -= 24
                departure_minute = departure_minutes % 60
                
                arrival_time = f"{arrival_hour:02d}:{arrival_minute:02d}"
                departure_time = f"{departure_hour:02d}:{departure_minute:02d}"
                
                stops.append({
                    'station_id': station_id,
                    'station_name': station['name'],
                    'arrival_time': arrival_time,
                    'departure_time': departure_time,
                    'platform': random.randint(1, min(station.get('platforms', 3), 8)),
                    'daytype': day_type
                })
                
                # Add travel time between stations (rough estimate: 25-35 min per segment)
                current_time_minutes += random.randint(25, 35)
            
            if len(stops) >= 2:
                train_id_counters[prefix] += 1
                train_id = f"{prefix}_{train_id_counters[prefix]:03d}"
                
                route_description = " → ".join(path)
                
                service = {
                    'train_id': train_id,
                    'service_type': service_type,
                    'route': f"{template['route_pattern']} ({len(path)} stations)",
                    'stops': stops
                }
                
                train_services.append(service)
                created += 1
                service_count += 1

print(f"   ✓ Created {service_count} train services")

# Save new timetable
print(f"\n💾 Saving regenerated timetable to {TIMETABLE_FILE}...")
with open(TIMETABLE_FILE, 'w') as f:
    json.dump(train_services, f, indent=2)

print(f"   ✓ Timetable saved with {len(train_services)} trains")

# Validate the timetable
print("\n✅ Validation Report:")
print(f"   • Total trains: {len(train_services)}")
print(f"   • Express trains: {sum(1 for t in train_services if t['service_type'] == 'express')}")
print(f"   • Regional trains: {sum(1 for t in train_services if t['service_type'] == 'regional')}")
print(f"   • Local trains: {sum(1 for t in train_services if t['service_type'] == 'local')}")

# Day type distribution
weekday_trains = sum(1 for t in train_services for s in t['stops'] if s['daytype'] == 'weekday')
weekend_trains = sum(1 for t in train_services for s in t['stops'] if s['daytype'] == 'weekend')
holiday_trains = sum(1 for t in train_services for s in t['stops'] if s['daytype'] == 'holiday')

print(f"   • Weekday services: {weekday_trains}")
print(f"   • Weekend services: {weekend_trains}")
print(f"   • Holiday services: {holiday_trains}")

# Peak hour distribution
morning_peak = sum(1 for t in train_services for s in t['stops'] 
                   if 7 <= int(s['arrival_time'].split(':')[0]) < 9)
evening_peak = sum(1 for t in train_services for s in t['stops'] 
                   if 16 <= int(s['arrival_time'].split(':')[0]) < 18)

print(f"   • Morning peak (7-9 AM) arrivals: {morning_peak}")
print(f"   • Evening peak (4-6 PM) arrivals: {evening_peak}")

# Check for any missing segments
print("\n🔍 Checking route validity...")
missing_count = 0
for train in train_services:
    stops = train['stops']
    
    # Get unique stations in order (stops has duplicates for each daytype)
    unique_stops = []
    seen_stn = None
    for stop in stops:
        if stop['station_id'] != seen_stn:
            unique_stops.append(stop['station_id'])
            seen_stn = stop['station_id']
    
    # Check consecutive segments
    for i in range(len(unique_stops) - 1):
        from_stn = unique_stops[i]
        to_stn = unique_stops[i + 1]
        
        if to_stn not in graph[from_stn]:
            print(f"   ⚠️  MISSING PATH: {train['train_id']}: {from_stn} → {to_stn}")
            missing_count += 1

if missing_count == 0:
    print("   ✓ All routes are valid - no missing segment connections!")
else:
    print(f"   ⚠️  Found {missing_count} invalid connections")

print("\n🎉 Timetable regeneration complete!")
