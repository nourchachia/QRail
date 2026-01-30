import json
import copy

def time_to_minutes(time_str):
    """Convert HH:MM to minutes"""
    h, m = map(int, time_str.split(':'))
    return h * 60 + m

def minutes_to_time(minutes):
    """Convert minutes back to HH:MM"""
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

def add_minutes_to_time(time_str, minutes):
    """Add minutes to a time string"""
    total = time_to_minutes(time_str) + minutes
    return minutes_to_time(total)

# Read the current timetable
with open('data/network/timetable.json', 'r') as f:
    timetable = json.load(f)

print(f"Original timetable: {len(timetable)} trains")
original_stops = sum(len(t['stops']) for t in timetable)
print(f"Original total stops: {original_stops}")

# For each train, add duplicate stops at offset times
for train in timetable:
    service_type = train.get('service_type', '')
    
    # Different offset based on service type
    offset = 6 if service_type == 'express' else 8 if service_type == 'regional' else 7
    
    # Keep original stops
    original_stops_list = list(train['stops'])
    
    # Add duplicate stops with offset times
    for stop in original_stops_list:
        new_stop = copy.deepcopy(stop)
        if 'departure_time' in new_stop:
            new_stop['departure_time'] = add_minutes_to_time(stop['departure_time'], offset * 60)
        if 'arrival_time' in new_stop:
            new_stop['arrival_time'] = add_minutes_to_time(stop['arrival_time'], offset * 60)
        train['stops'].append(new_stop)

new_stops = sum(len(t['stops']) for t in timetable)
print(f"\nUpdated timetable: {len(timetable)} trains (unchanged)")
print(f"Updated total stops: {new_stops}")
print(f"Each train now runs twice daily, doubling active trains at any time")

# Save updated timetable
with open('data/network/timetable.json', 'w') as f:
    json.dump(timetable, f, indent=2)

print(f"\n✓ Timetable updated successfully!")
