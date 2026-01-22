"""
Neural Rail Conductor - Live Status Generator

INPUT:  data/network/timetable.json
        data/network/stations.json
        data/network/segments.json

OUTPUT: data/processed/live_status.json  (Real-time train positions & telemetry)

PURPOSE: Generates the "Digital Twin Pulse" - real-time snapshot of all active trains
         with position interpolation, delay curves, and telemetry history.

USAGE:
    python live_status_generator.py
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("data/processed")
NETWORK_DIR = Path("data/network")

STATIONS_FILE = NETWORK_DIR / "stations.json"
SEGMENTS_FILE = NETWORK_DIR / "segments.json"
TIMETABLE_FILE = NETWORK_DIR / "timetable.json"

# Telemetry window configuration
TELEMETRY_WINDOW_SIZE = 10  # Number of historical points
TELEMETRY_WINDOW_MINUTES = 30  # Time span for telemetry buffer

# Signal aspects
SIGNAL_ASPECTS = ["Green", "Amber", "Red"]

# Weather conditions
WEATHER_CONDITIONS = ["clear", "rain", "heavy_rain", "snow", "fog", "storm"]

# Day types
# Day types (must match timetable.json values)
DAY_TYPES = ["weekday", "weekend", "holiday"]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_time(time_str: str) -> int:
    """
    Parse time string (HH:MM) to minutes since midnight.
    """
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def minutes_to_time_str(minutes: int) -> str:
    """
    Convert minutes since midnight to HH:MM string.
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def load_network_data():
    """
    Load network data files.
    """
    print("📍 Loading network data...")
    
    if not STATIONS_FILE.exists():
        raise FileNotFoundError(f"❌ Stations file not found: {STATIONS_FILE}")
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(f"❌ Segments file not found: {SEGMENTS_FILE}")
    if not TIMETABLE_FILE.exists():
        raise FileNotFoundError(f"❌ Timetable file not found: {TIMETABLE_FILE}")
    
    with open(STATIONS_FILE) as f:
        stations = json.load(f)
    
    with open(SEGMENTS_FILE) as f:
        segments = json.load(f)
    
    with open(TIMETABLE_FILE) as f:
        timetable = json.load(f)
    
    print(f"   ✅ Loaded: {len(stations)} stations, {len(segments)} segments, {len(timetable)} train services")
    
    return stations, segments, timetable


def find_segment(from_station_id: str, to_station_id: str, segments: List[Dict]) -> Optional[Dict]:
    """
    Find the segment connecting two stations.
    """
    for seg in segments:
        if (seg["from_station"] == from_station_id and seg["to_station"] == to_station_id) or \
           (seg.get("bidirectional", True) and seg["to_station"] == from_station_id and seg["from_station"] == to_station_id):
            return seg
    return None


# ============================================================================
# TRAIN POSITION CALCULATION
# ============================================================================

def get_train_position_at_time(train_data: Dict, current_time_minutes: int, day_type: str, 
                               segments: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Calculate where a train is at a specific time based on its timetable.
    Returns position data including segment, progress, and delay.
    """
    train_id = train_data["train_id"]
    stops = [s for s in train_data["stops"] if s.get("daytype") == day_type]
    
    if not stops:
        return None
    
    # Sort stops by arrival time
    stops_sorted = sorted(stops, key=lambda s: parse_time(s["arrival_time"]))
    
    # Find which segment/station the train is at
    for i in range(len(stops_sorted)):
        stop = stops_sorted[i]
        arrival_min = parse_time(stop["arrival_time"])
        departure_min = parse_time(stop["departure_time"])
        
        # Check if train is at this station
        if arrival_min <= current_time_minutes <= departure_min:
            # Calculate delay (if any)
            scheduled_arrival = arrival_min
            actual_arrival = current_time_minutes
            delay = max(0, actual_arrival - scheduled_arrival)
            
            return {
                "train_id": train_id,
                "position_type": "station",
                "station_id": stop["station_id"],
                "segment_id": None,
                "from_station": stop["station_id"],
                "to_station": stop["station_id"],
                "progress_pct": 0.0,
                "delay_minutes": delay,
                "scheduled_time": arrival_min,
                "platform": stop.get("platform", 1)
            }
        
        # Check if train is on segment to next station
        if i < len(stops_sorted) - 1:
            next_stop = stops_sorted[i + 1]
            next_arrival_min = parse_time(next_stop["arrival_time"])
            
            if departure_min < current_time_minutes < next_arrival_min:
                # Train is on segment between stop and next_stop
                from_station_id = stop["station_id"]
                to_station_id = next_stop["station_id"]
                
                # Find the segment
                segment = find_segment(from_station_id, to_station_id, segments)
                
                if segment:
                    segment_length = segment["length_km"]
                    scheduled_travel_time = next_arrival_min - departure_min
                    elapsed_time = current_time_minutes - departure_min
                    
                    if scheduled_travel_time > 0:
                        # Calculate progress
                        progress_pct = min(1.0, elapsed_time / scheduled_travel_time)
                        distance_km = progress_pct * segment_length
                        
                        # Calculate delay (compare actual vs scheduled position)
                        # If train is behind schedule, delay increases
                        scheduled_progress = elapsed_time / scheduled_travel_time
                        if scheduled_progress < progress_pct:
                            # Train is ahead of schedule
                            delay = 0
                        else:
                            # Train is behind schedule
                            delay = int((scheduled_progress - progress_pct) * scheduled_travel_time)
                        
                        return {
                            "train_id": train_id,
                            "position_type": "segment",
                            "station_id": None,
                            "segment_id": segment["id"],
                            "from_station": from_station_id,
                            "to_station": to_station_id,
                            "progress_pct": progress_pct,
                            "distance_km": distance_km,
                            "delay_minutes": delay,
                            "scheduled_time": departure_min,
                            "speed_limit": segment.get("speed_limit", 120)
                        }
    
    return None


def is_train_active(train_data: Dict, current_time_minutes: int, day_type: str) -> bool:
    """
    Check if a train is active (has departed but not yet completed its journey).
    """
    stops = [s for s in train_data["stops"] if s.get("daytype") == day_type]
    
    if not stops:
        return False
    
    stops_sorted = sorted(stops, key=lambda s: parse_time(s["arrival_time"]))
    
    # Check if current time is after first departure and before last arrival
    first_departure = parse_time(stops_sorted[0]["departure_time"])
    last_arrival = parse_time(stops_sorted[-1]["arrival_time"])
    
    return first_departure <= current_time_minutes <= last_arrival


def get_active_trains(timetable: List[Dict], current_time_minutes: int, day_type: str, 
                     segments: List[Dict]) -> List[Dict[str, Any]]:
    """
    Get all active trains with their current positions.
    """
    active_trains = []
    
    for train_data in timetable:
        if is_train_active(train_data, current_time_minutes, day_type):
            position = get_train_position_at_time(train_data, current_time_minutes, day_type, segments)
            if position:
                active_trains.append(position)
    
    return active_trains


# ============================================================================
# TELEMETRY WINDOW GENERATION
# ============================================================================

def generate_delay_curve(current_delay: int, window_minutes: int, num_points: int) -> List[int]:
    """
    Generate a delay curve for telemetry window.
    If current delay is 8m, entries at T-30m should be 0m, 2m, 5m, 8m.
    """
    if current_delay == 0:
        return [0] * num_points
    
    # Create a curve that starts at 0 and increases to current_delay
    delays = []
    
    # Use exponential growth curve
    for i in range(num_points):
        # Normalized position (0.0 to 1.0)
        t_norm = i / (num_points - 1) if num_points > 1 else 0
        
        # Exponential curve: delay grows faster near the end
        # Using x^2 curve for smoother growth
        delay = int(current_delay * (t_norm ** 2))
        delays.append(delay)
    
    return delays


def generate_telemetry_window(train_data: Dict, current_time_minutes: int, current_delay: int,
                              current_position: Dict, day_type: str, segments: List[Dict]) -> List[Dict[str, Any]]:
    """
    Generate telemetry buffer with historical position/delay samples.
    Returns list of 10 points for the last 30 minutes.
    """
    telemetry_points = []
    
    # Calculate time intervals
    interval_minutes = TELEMETRY_WINDOW_MINUTES / TELEMETRY_WINDOW_SIZE
    
    # Generate delay curve
    delay_curve = generate_delay_curve(current_delay, TELEMETRY_WINDOW_MINUTES, TELEMETRY_WINDOW_SIZE)
    
    # Generate historical positions
    for i in range(TELEMETRY_WINDOW_SIZE):
        # Time offset (negative, going back in time)
        offset_minutes = -int((TELEMETRY_WINDOW_SIZE - i) * interval_minutes)
        historical_time = current_time_minutes + offset_minutes
        
        # Skip if time is before train started
        stops = [s for s in train_data["stops"] if s.get("daytype") == day_type]
        if not stops:
            continue
        
        stops_sorted = sorted(stops, key=lambda s: parse_time(s["arrival_time"]))
        first_departure = parse_time(stops_sorted[0]["departure_time"])
        
        if historical_time < first_departure:
            continue
        
        # Get position at historical time
        hist_position = get_train_position_at_time(train_data, historical_time, day_type, segments)
        
        if hist_position:
            # Get delay from curve
            hist_delay = delay_curve[i]
            
            # Calculate relative time string
            time_offset = offset_minutes
            time_str = f"{abs(time_offset)}m" if time_offset < 0 else "0m"
            
            telemetry_points.append({
                "t": f"-{abs(time_offset)}m" if time_offset < 0 else "0m",
                "delay": hist_delay,
                "pos": round(hist_position.get("progress_pct", 0.0), 2),
                "segment_id": hist_position.get("segment_id"),
                "station_id": hist_position.get("station_id")
            })
    
    # Sort by time (oldest first)
    telemetry_points.sort(key=lambda x: int(x["t"].replace("-", "").replace("m", "")) if x["t"] != "0m" else 0)
    
    return telemetry_points


# ============================================================================
# SIGNAL ASPECT CALCULATION
# ============================================================================

def calculate_signal_aspect(train_position: Dict, delay: int, network_load: int) -> str:
    """
    Calculate signal aspect based on train status, delay, and network conditions.
    """
    # Red: Severe delay (>15 min) or stopped at station with high delay
    if delay > 15:
        return "Red"
    
    # Red: High network load and any delay
    if network_load > 80 and delay > 5:
        return "Red"
    
    # Amber: Moderate delay (5-15 min) or high network load
    if delay > 5 or network_load > 70:
        return "Amber"
    
    # Green: On time or minor delay
    return "Green"


# ============================================================================
# GLOBAL STATE GENERATION
# ============================================================================

def generate_weather_state() -> Dict[str, Any]:
    """
    Generate current weather conditions.
    """
    weather = random.choice(WEATHER_CONDITIONS)
    
    weather_data = {
        "condition": weather,
        "temperature_c": random.randint(-5, 35),
        "wind_speed_kmh": random.randint(0, 80) if weather in ["storm", "snow"] else random.randint(0, 30),
        "visibility_km": 10.0 if weather == "clear" else random.uniform(0.5, 5.0)
    }
    
    return weather_data


def calculate_network_load(active_trains: List[Dict], total_segments: int) -> int:
    """
    Calculate network load percentage based on active trains and capacity.
    """
    # Count trains on segments
    trains_on_segments = sum(1 for t in active_trains if t.get("position_type") == "segment")
    
    # Estimate load (simplified: assume each segment can handle ~10 trains simultaneously)
    max_capacity = 59
    current_load = len(active_trains)
    
    load_pct = min(100, int((current_load / max_capacity) * 100)) if max_capacity > 0 else 0
    
    return load_pct


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_live_status(snapshot_time: Optional[str] = None, day_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate live status snapshot for a specific time.
    
    Args:
        snapshot_time: ISO timestamp or "HH:MM" format. If None, uses current time.
        day_type: "weekday", "saturday", "sunday", or "holiday". If None, randomly selects.
    """
    # Load network data
    stations, segments, timetable = load_network_data()
    
    # Parse snapshot time
    if snapshot_time:
        if "T" in snapshot_time:
            # ISO format
            dt = datetime.fromisoformat(snapshot_time.replace("Z", "+00:00"))
            current_time_minutes = dt.hour * 60 + dt.minute
            snapshot_datetime = dt
        else:
            # HH:MM format
            current_time_minutes = parse_time(snapshot_time)
            snapshot_datetime = datetime.now().replace(
                hour=current_time_minutes // 60,
                minute=current_time_minutes % 60,
                second=0,
                microsecond=0
            )
    else:
        # Use current time (morning peak for realism)
        snapshot_datetime = datetime.now().replace(hour=8, minute=30, second=0, microsecond=0)
        current_time_minutes = 8 * 60 + 30
    
    # Select day type
    if not day_type:
        day_type = random.choice(DAY_TYPES)
    
    print(f"\n🕐 Generating live status for {snapshot_datetime.strftime('%Y-%m-%d %H:%M:%S')} ({day_type})")
    
    # Get active trains
    active_trains = get_active_trains(timetable, current_time_minutes, day_type, segments)
    print(f"   🚂 Found {len(active_trains)} active trains")
    
    # Generate weather state
    weather = generate_weather_state()
    
    # Calculate network load
    network_load = calculate_network_load(active_trains, len(segments))
    
    # Build active trains data with telemetry
    trains_data = []
    for train_pos in active_trains:
        train_id = train_pos["train_id"]
        
        # Find full train data
        train_data = next((t for t in timetable if t["train_id"] == train_id), None)
        if not train_data:
            continue
        
        current_delay = train_pos.get("delay_minutes", 0)
        
        # Generate telemetry window
        telemetry_window = generate_telemetry_window(
            train_data, current_time_minutes, current_delay, train_pos, day_type, segments
        )
        
        # Calculate signal aspect
        signal_aspect = calculate_signal_aspect(train_pos, current_delay, network_load)
        
        # Build current position
        cur_pos = {}
        if train_pos["position_type"] == "segment":
            cur_pos = {
                "segment": train_pos["segment_id"],
                "pct": round(train_pos["progress_pct"], 2)
            }
        else:
            cur_pos = {
                "station": train_pos["station_id"],
                "platform": train_pos.get("platform", 1)
            }
        
        trains_data.append({
            "train_id": train_id,
            "cur_pos": cur_pos,
            "cur_delay": current_delay,
            "telemetry_window": telemetry_window,
            "signal_aspect": signal_aspect,
            "service_type": train_data.get("service_type", "unknown"),
            "route": train_data.get("route", "unknown")
        })
    
    # Build final status
    live_status = {
        "timestamp": snapshot_datetime.isoformat(),
        "day_type": day_type,
        "active_trains": trains_data,
        "weather": weather,
        "network_load_pct": network_load,
        "total_active_trains": len(trains_data)
    }
    
    return live_status


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main execution: Generate live status snapshot and save to JSON.
    """
    print("\n" + "="*60)
    print("🚂 Neural Rail Conductor - Live Status Generator")
    print("="*60)
    
    # Generate live status (default: 08:30 weekday)
    live_status = generate_live_status()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    output_path = OUTPUT_DIR / "live_status.json"
    
    print(f"\n💾 Saving live status to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(live_status, f, indent=2)
    
    # Print statistics
    print(f"\n✅ Live status generated successfully!")
    print(f"\n📊 Statistics:")
    print(f"   Timestamp: {live_status['timestamp']}")
    print(f"   Active trains: {live_status['total_active_trains']}")
    print(f"   Network load: {live_status['network_load_pct']}%")
    print(f"   Weather: {live_status['weather']['condition']}")
    
    # Signal aspect distribution
    signal_counts = {}
    for train in live_status["active_trains"]:
        aspect = train["signal_aspect"]
        signal_counts[aspect] = signal_counts.get(aspect, 0) + 1
    
    print(f"\n   Signal aspects:")
    for aspect, count in sorted(signal_counts.items()):
        print(f"      {aspect}: {count}")
    
    # Delay distribution
    delays = [t["cur_delay"] for t in live_status["active_trains"]]
    if delays:
        print(f"\n   Delay statistics:")
        print(f"      Average: {np.mean(delays):.1f} minutes")
        print(f"      Max: {max(delays)} minutes")
        print(f"      On-time (0 min): {sum(1 for d in delays if d == 0)} trains")
    
    print("\n✨ Generation complete!")


if __name__ == "__main__":
    main()
