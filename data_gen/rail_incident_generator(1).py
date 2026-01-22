"""
Neural Rail Conductor - Incident Data Generator (UPDATED)

INPUT:  data/network/stations.json  (50 stations with hierarchy)
        data/network/segments.json  (70 segments: Main/Branch/Loop)

OUTPUT: data/processed/incidents.json  (1,000 realistic incidents)

ALIGNMENT: Now generates incidents that match the exact network topology
           from blueprint Part 2 (Pentagon hubs, radius-based zones, etc.)

USAGE:
    python generate_incidents.py
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

TOTAL_INCIDENTS = 1000
TRAIN_SPLIT = 0.8  # 80% training, 20% test

OUTPUT_DIR = Path("data/processed")
NETWORK_DIR = Path("data/network")

# INPUT FILES (Required from teammate)
STATIONS_FILE = NETWORK_DIR / "stations.json"
SEGMENTS_FILE = NETWORK_DIR / "segments.json"
TIMETABLE_FILE = NETWORK_DIR / "timetable.json"

# ============================================================================
# INCIDENT TEMPLATES (5 Archetypes - UPDATED)
# ============================================================================

INCIDENT_TEMPLATES = [
    {
        "type": "signal_failure",
        "archetype": "Bottleneck Cascade",
        "description_template": "Signal system failure at {location_name} ({zone} zone) during {weather} conditions at {time_period}. {junction_detail} Multiple converging routes affected. {severity_impact}",
        "base_severity": 4,
        "preferred_locations": ["major_hub", "junction"],  # Uses real topology
        "cascade_depth_range": (3, 5),
        "requires_switches": True,  # NEW: Must have switches
        "resolution_strategies": [
            "EMERGENCY_SINGLE_LINE_WORKING",
            "REROUTE_VIA_ALTERNATE_JUNCTION",
            "HOLD_ALL_UPSTREAM_TRAINS"
        ]
    },
    {
        "type": "train_breakdown",
        "archetype": "Static Blockage",
        "description_template": "Train {train_id} ({stock_type}) mechanical failure on {segment_name} between {from_station} and {to_station}. {cause}. {weather} weather. {time_period} service.",
        "base_severity": 3,
        "preferred_locations": ["segment", "branch_line"],  # Prefers branch lines
        "cascade_depth_range": (1, 3),
        "stock_types": ["Electric Multiple Unit", "Diesel Unit", "Locomotive-hauled"],
        "resolution_strategies": [
            "RESCUE_TRAIN_DISPATCH",
            "REROUTE_VIA_LOOP",
            "TEMPORARY_BUS_REPLACEMENT"
        ]
    },
    {
        "type": "passenger_alarm",
        "archetype": "Ripple Delay",
        "description_template": "Emergency alarm activated on train {train_id} at {location_name}. {alarm_reason}. {passenger_count} passengers on board. {time_period}.",
        "base_severity": 2,
        "preferred_locations": ["regional", "local", "major_hub"],
        "cascade_depth_range": (1, 2),
        "alarm_reasons": [
            "Medical emergency reported",
            "Suspicious package investigation",
            "Door obstruction",
            "Passenger altercation"
        ],
        "resolution_strategies": [
            "MEDICAL_TEAM_DISPATCH",
            "SECURITY_RESPONSE",
            "EXTEND_DWELL_TIME",
            "CONTINUE_WITH_CAUTION"
        ]
    },
    {
        "type": "severe_weather",
        "archetype": "Global Slowdown",
        "description_template": "Severe {weather} event across {zone} zone. Network-wide speed restrictions implemented. {weather_details}. Affecting {affected_count} services. {time_period}.",
        "base_severity": 4,
        "preferred_locations": ["network_wide", "zone_wide"],
        "cascade_depth_range": (4, 5),
        "weather_types": {
            "heavy_rain": "Flooding risk on low-lying sections",
            "snow": "Ice accumulation on overhead wires",
            "storm": "Wind speed exceeding safe limits for high-sided stock",
            "fog": "Visibility below signaling safety threshold"
        },
        "resolution_strategies": [
            "IMPLEMENT_EMERGENCY_TIMETABLE",
            "REDUCE_ALL_SPEED_LIMITS",
            "CANCEL_NON_ESSENTIAL_SERVICES",
            "INCREASE_HEADWAYS_NETWORK_WIDE"
        ]
    },
    {
        "type": "infrastructure_fault",
        "archetype": "Dead End Closure",
        "description_template": "Critical infrastructure failure: {fault_type} at {location_name}. {technical_details}. Segment {segment_id} closed for emergency works. {time_period}.",
        "base_severity": 5,
        "preferred_locations": ["critical_segment", "main_line", "junction"],
        "cascade_depth_range": (2, 4),
        "fault_types": {
            "track_buckle": "Rail temperature exceeds design limits",
            "switch_failure": "Points mechanism jammed, manual operation not possible",
            "overhead_wire_damage": "Pantograph contact lost, arc damage detected",
            "platform_edge_failure": "Structural integrity compromised",
            "signal_box_fire": "Control systems offline, backup failover initiated"
        },
        "resolution_strategies": [
            "COMPLETE_SEGMENT_CLOSURE",
            "ACTIVATE_CONTINGENCY_ROUTING",
            "TERMINATE_SERVICES_AT_INTERMEDIATE_STATION",
            "IMPLEMENT_BUS_BRIDGE"
        ]
    }
]

# ============================================================================
# ENHANCED CONTEXTUAL VARIABLES
# ============================================================================

WEATHER_CONDITIONS = {
    "clear": {"severity_mod": 0, "speed_impact": 1.0, "visibility_km": 10.0},
    "rain": {"severity_mod": 1, "speed_impact": 0.85, "visibility_km": 5.0},
    "heavy_rain": {"severity_mod": 2, "speed_impact": 0.7, "visibility_km": 2.0},
    "snow": {"severity_mod": 2, "speed_impact": 0.6, "visibility_km": 3.0},
    "fog": {"severity_mod": 1, "speed_impact": 0.75, "visibility_km": 0.5},
    "storm": {"severity_mod": 3, "speed_impact": 0.5, "visibility_km": 4.0}
}

TIME_PERIODS = {
    "early_morning": {"hours": range(5, 7), "is_peak": False, "load_pct": 30, "label": "early morning service"},
    "morning_peak": {"hours": range(7, 10), "is_peak": True, "load_pct": 90, "label": "morning peak"},
    "midday": {"hours": range(10, 16), "is_peak": False, "load_pct": 50, "label": "off-peak"},
    "evening_peak": {"hours": range(16, 20), "is_peak": True, "load_pct": 95, "label": "evening rush"},
    "night": {"hours": range(20, 24), "is_peak": False, "load_pct": 25, "label": "evening service"},
    "late_night": {"hours": range(0, 5), "is_peak": False, "load_pct": 10, "label": "overnight maintenance window"}
}

DAY_TYPES = ["weekday", "saturday", "sunday", "holiday"]

TRAIN_ID_PREFIXES = ["T", "E", "R", "L"]  # T=Terminal, E=Express, R=Regional, L=Local

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Incident:
    incident_id: str
    type: str
    archetype: str
    semantic_description: str
    
    # Location (Enhanced with real topology)
    location_id: str
    location_name: str
    location_type: str  # major_hub, regional, local, segment, network_wide
    station_ids: List[str]
    segment_id: str
    zone: str  # core, mid, peripheral
    is_junction: bool
    has_switches: bool
    
    # Temporal
    timestamp: str
    hour_of_day: int
    time_of_day: str
    day_type: str
    is_peak: bool
    
    # Environmental (Enhanced)
    weather_condition: str
    temperature_c: int
    visibility_km: float
    wind_speed_kmh: int
    
    # Operational
    severity_level: int  # 1-5
    network_load_pct: int
    trains_affected_count: int
    cascade_depth: int
    estimated_delay_minutes: int
    
    # Technical details (NEW)
    affected_platforms: List[int]
    track_circuits_affected: List[str]
    rolling_stock_type: Optional[str]
    
    # Train positions (NEW - timetable-based)
    affected_trains: List[Dict[str, Any]]  # List of trains at incident location/time
    train_positions_at_incident: Dict[str, Dict[str, Any]]  # train_id -> position details
    
    # Resolution
    resolution_code: str
    resolution_strategy: str
    actions_taken: List[Dict[str, Any]]
    solution_rating: Dict[str, Any]  # NEW: Physical feasibility and effectiveness rating
    
    # Outcome
    actual_delay_minutes: int
    passenger_satisfaction: str  # Low, Medium, High
    outcome_score: float  # 0.0 to 1.0
    operator_rating: str  # thumbs_up, neutral, thumbs_down
    
    # Context (NEW - for Gemini/LLM enrichment)
    maintenance_notes: str
    operator_logs: str
    
    # Metadata
    is_golden_run: bool
    similar_incident_ids: List[str]
    created_at: str

# ============================================================================
# TIMETABLE LOADING & TRAIN POSITION CALCULATION
# ============================================================================

def load_timetable():
    """
    Load timetable data.
    """
    if not TIMETABLE_FILE.exists():
        raise FileNotFoundError(
            f"❌ Timetable file not found: {TIMETABLE_FILE}\n"
            f"   Required for timetable-conforming incident generation"
        )
    
    print(f"✅ Loading timetable from: {TIMETABLE_FILE}")
    with open(TIMETABLE_FILE) as f:
        timetable = json.load(f)
    
    print(f"   📊 Timetable loaded: {len(timetable)} train services")
    return timetable


def parse_time(time_str: str) -> int:
    """
    Parse time string (HH:MM) to minutes since midnight.
    """
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def time_to_minutes(hour: int, minute: int = 0) -> int:
    """
    Convert hour and minute to minutes since midnight.
    """
    return hour * 60 + minute


def get_train_position_at_time(train_data: Dict, incident_time_minutes: int, day_type: str, segments: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Calculate where a train is at a specific time based on its timetable.
    Returns: {
        "train_id": str,
        "position_type": "station" | "segment",
        "station_id": str (if at station),
        "segment_id": str (if on segment),
        "from_station": str,
        "to_station": str,
        "progress_pct": float (0.0-1.0, if on segment),
        "distance_km": float (if on segment)
    }
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
        if arrival_min <= incident_time_minutes <= departure_min:
            return {
                "train_id": train_id,
                "position_type": "station",
                "station_id": stop["station_id"],
                "segment_id": None,
                "from_station": stop["station_id"],
                "to_station": stop["station_id"],
                "progress_pct": 0.0,
                "distance_km": 0.0,
                "platform": stop.get("platform", 1)
            }
        
        # Check if train is on segment to next station
        if i < len(stops_sorted) - 1:
            next_stop = stops_sorted[i + 1]
            next_arrival_min = parse_time(next_stop["arrival_time"])
            
            if departure_min < incident_time_minutes < next_arrival_min:
                # Train is on segment between stop and next_stop
                from_station_id = stop["station_id"]
                to_station_id = next_stop["station_id"]
                
                # Find the segment
                segment = None
                for seg in segments:
                    if (seg["from_station"] == from_station_id and seg["to_station"] == to_station_id) or \
                       (seg.get("bidirectional", True) and seg["to_station"] == from_station_id and seg["from_station"] == to_station_id):
                        segment = seg
                        break
                
                if segment:
                    segment_length = segment["length_km"]
                    travel_time_minutes = next_arrival_min - departure_min
                    elapsed_time = incident_time_minutes - departure_min
                    
                    if travel_time_minutes > 0:
                        progress_pct = min(1.0, elapsed_time / travel_time_minutes)
                        distance_km = progress_pct * segment_length
                        
                        return {
                            "train_id": train_id,
                            "position_type": "segment",
                            "station_id": None,
                            "segment_id": segment["id"],
                            "from_station": from_station_id,
                            "to_station": to_station_id,
                            "progress_pct": progress_pct,
                            "distance_km": distance_km,
                            "speed_limit": segment.get("speed_limit", 120)
                        }
    
    return None


def find_trains_at_location(timetable: List[Dict], incident_time_minutes: int, day_type: str, 
                           location: Dict, segments: List[Dict]) -> List[Dict[str, Any]]:
    """
    Find all trains at a given location (station or segment) at a specific time.
    """
    trains_at_location = []
    
    for train_data in timetable:
        position = get_train_position_at_time(train_data, incident_time_minutes, day_type, segments)
        
        if not position:
            continue
        
        # Check if train is at the incident location
        is_at_location = False
        
        if location["location_type"] == "segment" and position["position_type"] == "segment":
            # Check if train is on the affected segment
            if position["segment_id"] == location["segment_id"]:
                is_at_location = True
        
        elif location["location_type"] in ["major_hub", "regional", "local", "junction"]:
            # Check if train is at the affected station
            if position["position_type"] == "station" and position["station_id"] in location["station_ids"]:
                is_at_location = True
        
        elif location["location_type"] == "network_wide":
            # Network-wide incidents affect all trains
            is_at_location = True
        
        if is_at_location:
            trains_at_location.append({
                **position,
                "service_type": train_data.get("service_type", "unknown"),
                "route": train_data.get("route", "unknown")
            })
    
    return trains_at_location


# ============================================================================
# NETWORK LOADING & ANALYSIS
# ============================================================================

def load_network_data():
    """
    Load and validate station/segment data with enhanced topology awareness.
    """
    
    if not STATIONS_FILE.exists():
        raise FileNotFoundError(
            f"❌ Network file not found: {STATIONS_FILE}\n"
            f"   Your teammate needs to generate this first using generate_network.py"
        )
    
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"❌ Network file not found: {SEGMENTS_FILE}\n"
            f"   Your teammate needs to generate this first using generate_network.py"
        )
    
    print(f"✅ Loading stations from: {STATIONS_FILE}")
    with open(STATIONS_FILE) as f:
        stations = json.load(f)
    
    print(f"✅ Loading segments from: {SEGMENTS_FILE}")
    with open(SEGMENTS_FILE) as f:
        segments = json.load(f)
    
    # Validation
    if len(stations) == 0:
        raise ValueError("stations.json is empty!")
    if len(segments) == 0:
        raise ValueError("segments.json is empty!")
    
    print(f"   📊 Network loaded: {len(stations)} stations, {len(segments)} segments")
    
    # Enhanced topology analysis
    network_stats = analyze_network_topology(stations, segments)
    
    return stations, segments, network_stats


def analyze_network_topology(stations, segments):
    """
    Analyze network to identify critical nodes, junctions, and segment types.
    This enables realistic incident placement.
    """
    stats = {
        "major_hubs": [],
        "regional_stations": [],
        "local_stations": [],
        "minor_halts": [],
        "junctions": [],
        "main_line_segments": [],
        "branch_line_segments": [],
        "loop_segments": [],
        "critical_segments": [],
        "stations_by_zone": {"core": [], "mid": [], "peripheral": []}
    }
    
    # Count connections per station
    connections = {}
    for segment in segments:
        from_stn = segment.get("from_station")
        to_stn = segment.get("to_station")
        
        connections[from_stn] = connections.get(from_stn, 0) + 1
        if segment.get("bidirectional", True):
            connections[to_stn] = connections.get(to_stn, 0) + 1
    
    # Categorize stations
    for station in stations:
        stn_type = station.get("type", "local")
        zone = station.get("zone", "mid")
        
        # By type
        if stn_type == "major_hub":
            stats["major_hubs"].append(station)
        elif stn_type == "regional":
            stats["regional_stations"].append(station)
        elif stn_type == "local":
            stats["local_stations"].append(station)
        elif stn_type == "minor_halt":
            stats["minor_halts"].append(station)
        
        # By zone
        stats["stations_by_zone"][zone].append(station)
        
        # Identify junctions (3+ connections OR major_hub)
        if connections.get(station["id"], 0) >= 3 or stn_type == "major_hub":
            stats["junctions"].append(station)
    
    # Categorize segments
    for segment in segments:
        seg_type = segment.get("track_type", "branch")
        is_critical = segment.get("is_critical", False)
        
        if seg_type == "main_line":
            stats["main_line_segments"].append(segment)
        elif seg_type == "loop":
            stats["loop_segments"].append(segment)
        else:
            stats["branch_line_segments"].append(segment)
        
        if is_critical:
            stats["critical_segments"].append(segment)
    
    # Print summary
    print(f"   🔀 Identified {len(stats['junctions'])} junctions")
    print(f"   🏢 Major hubs: {len(stats['major_hubs'])}")
    print(f"   🚉 Regional: {len(stats['regional_stations'])}")
    print(f"   🚏 Local: {len(stats['local_stations'])}")
    print(f"   ⚠️  Critical segments: {len(stats['critical_segments'])}")
    
    return stats


# ============================================================================
# SMART LOCATION SELECTION
# ============================================================================

def select_incident_location(template, stations, segments, network_stats):
    """
    Intelligently select location based on incident type and real network topology.
    Uses the preferred_locations from template to match realistic scenarios.
    """
    preferred = template["preferred_locations"]
    
    # CASE 1: Network-wide incidents (weather events)
    if "network_wide" in preferred:
        zone = random.choice(["core", "mid", "peripheral"])
        affected = random.sample(
            network_stats["stations_by_zone"][zone],
            min(10, len(network_stats["stations_by_zone"][zone]))
        )
        
        return {
            "location_id": f"ZONE_{zone.upper()}",
            "location_name": f"{zone.capitalize()} Zone Network",
            "location_type": "network_wide",
            "station_ids": [s["id"] for s in affected],
            "segment_id": "",
            "zone": zone,
            "is_junction": False,
            "has_switches": False,
            "stations": affected
        }
    
    # CASE 2: Junction/Hub incidents (signal failures)
    if "major_hub" in preferred or "junction" in preferred:
        if network_stats["junctions"]:
            station = random.choice(network_stats["junctions"])
        else:
            station = random.choice(network_stats["major_hubs"] or stations)
        
        # Find connected segments
        connected_segs = [
            seg for seg in segments
            if seg.get("from_station") == station["id"] or seg.get("to_station") == station["id"]
        ]
        
        return {
            "location_id": station["id"],
            "location_name": station["name"],
            "location_type": station.get("type", "junction"),
            "station_ids": [station["id"]],
            "segment_id": connected_segs[0]["id"] if connected_segs else "",
            "zone": station.get("zone", "core"),
            "is_junction": True,
            "has_switches": station.get("has_switches", True),
            "stations": [station]
        }
    
    # CASE 3: Segment incidents (breakdowns, infrastructure faults)
    if "segment" in preferred or "branch_line" in preferred or "critical_segment" in preferred:
        # Prefer critical segments for infrastructure faults
        if "critical_segment" in preferred and network_stats["critical_segments"]:
            segment = random.choice(network_stats["critical_segments"])
        elif "main_line" in preferred and network_stats["main_line_segments"]:
            segment = random.choice(network_stats["main_line_segments"])
        elif "branch_line" in preferred and network_stats["branch_line_segments"]:
            segment = random.choice(network_stats["branch_line_segments"])
        else:
            segment = random.choice(segments)
        
        # Get connected stations
        from_station = next((s for s in stations if s["id"] == segment["from_station"]), None)
        to_station = next((s for s in stations if s["id"] == segment["to_station"]), None)
        
        return {
            "location_id": segment["id"],
            "location_name": f"Segment {segment['id']}",
            "location_type": "segment",
            "station_ids": [segment["from_station"], segment["to_station"]],
            "segment_id": segment["id"],
            "zone": from_station.get("zone", "mid") if from_station else "mid",
            "is_junction": False,
            "has_switches": segment.get("has_switches", False),
            "stations": [s for s in [from_station, to_station] if s]
        }
    
    # CASE 4: Regional/Local stations (passenger alarms)
    if "regional" in preferred or "local" in preferred:
        station_pool = network_stats["regional_stations"] + network_stats["local_stations"]
        if not station_pool:
            station_pool = stations
        
        station = random.choice(station_pool)
        
        return {
            "location_id": station["id"],
            "location_name": station["name"],
            "location_type": station.get("type", "regional"),
            "station_ids": [station["id"]],
            "segment_id": "",
            "zone": station.get("zone", "mid"),
            "is_junction": False,
            "has_switches": station.get("has_switches", False),
            "stations": [station]
        }
    
    # Fallback
    station = random.choice(stations)
    return {
        "location_id": station["id"],
        "location_name": station["name"],
        "location_type": station.get("type", "local"),
        "station_ids": [station["id"]],
        "segment_id": "",
        "zone": station.get("zone", "mid"),
        "is_junction": False,
        "has_switches": station.get("has_switches", False),
        "stations": [station]
    }


# ============================================================================
# ENHANCED DESCRIPTION GENERATION
# ============================================================================

def generate_semantic_description(template, location, context):
    """
    Generate realistic, human-like incident descriptions using the template.
    """
    description = template["description_template"]
    
    # Fill in placeholders
    replacements = {
        "location_name": location["location_name"],
        "zone": location["zone"],
        "weather": context["weather"],
        "time_period": context["time_label"],
        "train_id": f"{random.choice(TRAIN_ID_PREFIXES)}{random.randint(100, 999)}",
        "segment_name": location["location_name"],
        "segment_id": location["segment_id"],
        "from_station": location["stations"][0]["name"] if location["stations"] else "Unknown",
        "to_station": location["stations"][1]["name"] if len(location["stations"]) > 1 else "Terminal"
    }
    
    # Type-specific details
    if template["type"] == "signal_failure":
        replacements["junction_detail"] = f"{len(location['station_ids'])} converging routes." if location["is_junction"] else ""
        replacements["severity_impact"] = f"Estimated {context['trains_affected']} trains affected."
    
    elif template["type"] == "train_breakdown":
        replacements["stock_type"] = random.choice(template.get("stock_types", ["Unknown"]))
        replacements["cause"] = f"Reported {random.choice(['traction motor failure', 'brake system fault', 'door malfunction', 'pantograph damage'])}"
    
    elif template["type"] == "passenger_alarm":
        replacements["alarm_reason"] = random.choice(template.get("alarm_reasons", ["Unknown reason"]))
        replacements["passenger_count"] = random.randint(50, 800)
    
    elif template["type"] == "severe_weather":
        weather_type = context["weather"]
        replacements["weather_details"] = template.get("weather_types", {}).get(weather_type, "Adverse conditions")
        replacements["affected_count"] = random.randint(5, 20)
    
    elif template["type"] == "infrastructure_fault":
        fault = random.choice(list(template.get("fault_types", {}).keys()))
        replacements["fault_type"] = fault
        replacements["technical_details"] = template["fault_types"][fault]
    
    # Apply replacements
    for key, value in replacements.items():
        description = description.replace(f"{{{key}}}", str(value))
    
    return description


# ============================================================================
# PHYSICALLY FEASIBLE RESOLUTION GENERATION
# ============================================================================

def calculate_travel_time(segment: Dict, speed_limit: int, weather_speed_mod: float) -> float:
    """
    Calculate travel time in minutes for a segment given speed limit and weather.
    """
    length_km = segment["length_km"]
    effective_speed = speed_limit * weather_speed_mod
    if effective_speed <= 0:
        return float('inf')
    return (length_km / effective_speed) * 60  # minutes


def find_alternative_route(from_station_id: str, to_station_id: str, blocked_segment_id: str,
                          segments: List[Dict], stations: List[Dict]) -> Optional[List[str]]:
    """
    Find an alternative route avoiding the blocked segment.
    Returns list of segment IDs or None if no route found.
    """
    # Simple BFS to find alternative route
    from collections import deque
    
    queue = deque([(from_station_id, [])])
    visited = {from_station_id}
    
    while queue:
        current_station, path = queue.popleft()
        
        if current_station == to_station_id:
            return path
        
        # Check all segments from current station
        for seg in segments:
            if seg["id"] == blocked_segment_id:
                continue
            
            next_station = None
            if seg["from_station"] == current_station:
                next_station = seg["to_station"]
            elif seg.get("bidirectional", True) and seg["to_station"] == current_station:
                next_station = seg["from_station"]
            
            if next_station and next_station not in visited:
                visited.add(next_station)
                queue.append((next_station, path + [seg["id"]]))
    
    return None


def generate_physically_feasible_resolution(incident_type: str, location: Dict, affected_trains: List[Dict],
                                           segments: List[Dict], stations: List[Dict], 
                                           weather_speed_mod: float, resolution_strategy: str) -> Dict[str, Any]:
    """
    Generate a physically feasible resolution considering:
    - Train positions and speeds
    - Segment distances and speed limits
    - Weather impacts
    - Physical constraints (can't teleport trains, must respect speeds)
    """
    actions = []
    feasibility_score = 1.0
    physical_constraints = []
    
    if resolution_strategy == "REROUTE_VIA_ALTERNATE_JUNCTION" or "REROUTE" in resolution_strategy:
        # Check if rerouting is physically possible
        if location["location_type"] == "segment" and location["segment_id"]:
            blocked_seg = next((s for s in segments if s["id"] == location["segment_id"]), None)
            if blocked_seg:
                from_stn = blocked_seg["from_station"]
                to_stn = blocked_seg["to_station"]
                
                # Find alternative route
                alt_route = find_alternative_route(from_stn, to_stn, location["segment_id"], segments, stations)
                
                if alt_route:
                    # Calculate total travel time for alternative route
                    total_time = 0
                    total_distance = 0
                    for seg_id in alt_route:
                        seg = next((s for s in segments if s["id"] == seg_id), None)
                        if seg:
                            total_time += calculate_travel_time(seg, seg.get("speed_limit", 120), weather_speed_mod)
                            total_distance += seg["length_km"]
                    
                    # Compare to original route
                    original_time = calculate_travel_time(blocked_seg, blocked_seg.get("speed_limit", 120), weather_speed_mod)
                    delay_penalty = max(0, total_time - original_time)
                    
                    actions.append({
                        "action": "reroute_trains",
                        "train_count": len(affected_trains),
                        "via_segments": alt_route,
                        "additional_travel_time_minutes": round(delay_penalty, 1),
                        "additional_distance_km": round(total_distance - blocked_seg["length_km"], 1),
                        "feasible": True
                    })
                    
                    # Reduce feasibility if delay is too high
                    if delay_penalty > 30:
                        feasibility_score *= 0.7
                        physical_constraints.append("Alternative route adds significant delay")
                    elif delay_penalty > 15:
                        feasibility_score *= 0.85
                else:
                    feasibility_score *= 0.3
                    physical_constraints.append("No alternative route available")
                    actions.append({
                        "action": "reroute_trains",
                        "feasible": False,
                        "reason": "No alternative route found"
                    })
    
    elif resolution_strategy == "RESCUE_TRAIN_DISPATCH":
        # Check if rescue train can reach breakdown location
        if affected_trains:
            breakdown_train = affected_trains[0]
            if breakdown_train.get("position_type") == "segment":
                # Find nearest station with available rescue train
                segment = next((s for s in segments if s["id"] == breakdown_train.get("segment_id")), None)
                if segment:
                    # Estimate time for rescue train to reach
                    # Assume rescue train at nearest major hub
                    nearest_station = breakdown_train.get("from_station")
                    rescue_time = random.randint(20, 45)  # Realistic rescue dispatch time
                    
                    actions.append({
                        "action": "dispatch_rescue_train",
                        "from_station": nearest_station,
                        "estimated_arrival_minutes": rescue_time,
                        "feasible": True
                    })
                    
                    if rescue_time > 40:
                        feasibility_score *= 0.8
                        physical_constraints.append("Rescue train response time is high")
    
    elif resolution_strategy == "HOLD_ALL_UPSTREAM_TRAINS":
        # Check capacity at holding station
        if location["station_ids"]:
            holding_station_id = location["station_ids"][0]
            station = next((s for s in stations if s["id"] == holding_station_id), None)
            if station:
                platform_count = station.get("platforms", 2)
                trains_to_hold = len(affected_trains) + random.randint(1, 3)
                
                if trains_to_hold > platform_count * 2:  # Can hold 2 trains per platform
                    feasibility_score *= 0.6
                    physical_constraints.append(f"Insufficient platform capacity ({platform_count} platforms for {trains_to_hold} trains)")
                
                actions.append({
                    "action": "hold_upstream_trains",
                    "station_id": holding_station_id,
                    "trains_held": trains_to_hold,
                    "platform_capacity": platform_count,
                    "estimated_hold_duration_minutes": random.randint(5, 20),
                    "feasible": trains_to_hold <= platform_count * 2
                })
    
    elif resolution_strategy == "IMPLEMENT_EMERGENCY_TIMETABLE" or "REDUCE_ALL_SPEED_LIMITS" in resolution_strategy:
        # Network-wide speed reduction is always physically possible
        speed_reduction_pct = random.randint(20, 40)
        actions.append({
            "action": "reduce_speed_limits",
            "reduction_percent": speed_reduction_pct,
            "network_wide": True,
            "feasible": True
        })
        # Slight feasibility reduction due to cascading delays
        feasibility_score *= 0.9
    
    elif resolution_strategy == "CANCEL_NON_ESSENTIAL_SERVICES":
        # Cancellation is always physically possible
        cancel_count = min(len(affected_trains), random.randint(1, 3))
        actions.append({
            "action": "cancel_services",
            "count": cancel_count,
            "train_ids": [t["train_id"] for t in affected_trains[:cancel_count]],
            "feasible": True
        })
    
    else:
        # Default action
        actions.append({
            "action": "monitor_and_assess",
            "duration_minutes": 5,
            "feasible": True
        })
    
    return {
        "actions": actions,
        "feasibility_score": round(feasibility_score, 3),
        "physical_constraints": physical_constraints,
        "is_physically_possible": feasibility_score > 0.5
    }


def rate_solution(resolution_data: Dict, estimated_delay: int, trains_affected: int, 
                 is_peak: bool, severity: int) -> Dict[str, Any]:
    """
    Rate a solution based on:
    - Physical feasibility
    - Delay impact
    - Number of trains affected
    - Peak hour considerations
    - Overall effectiveness
    """
    feasibility = resolution_data["feasibility_score"]
    
    # Calculate delay impact score (lower delay = better)
    delay_score = 1.0
    if estimated_delay > 60:
        delay_score = 0.5
    elif estimated_delay > 30:
        delay_score = 0.7
    elif estimated_delay > 15:
        delay_score = 0.85
    
    # Calculate passenger impact score
    passenger_score = 1.0
    if trains_affected > 10:
        passenger_score = 0.6
    elif trains_affected > 5:
        passenger_score = 0.75
    elif trains_affected > 2:
        passenger_score = 0.9
    
    # Peak hour penalty
    peak_penalty = 0.9 if is_peak else 1.0
    
    # Severity consideration
    severity_mod = 1.0 - (severity - 1) * 0.1
    
    # Overall rating
    overall_score = feasibility * delay_score * passenger_score * peak_penalty * severity_mod
    overall_score = max(0.0, min(1.0, overall_score))
    
    # Rating category
    if overall_score >= 0.8:
        rating_category = "Excellent"
    elif overall_score >= 0.6:
        rating_category = "Good"
    elif overall_score >= 0.4:
        rating_category = "Acceptable"
    else:
        rating_category = "Poor"
    
    return {
        "overall_score": round(overall_score, 3),
        "feasibility_score": feasibility,
        "delay_score": round(delay_score, 3),
        "passenger_impact_score": round(passenger_score, 3),
        "rating_category": rating_category,
        "constraints": resolution_data["physical_constraints"],
        "is_recommended": overall_score >= 0.6
    }


# ============================================================================
# INCIDENT GENERATION (MAIN)
# ============================================================================

def generate_incident(stations, segments, network_stats, timetable, is_golden=False) -> Incident:
    """
    Generate a single realistic incident based on real network topology.
    """
    
    # Select template
    template = random.choice(INCIDENT_TEMPLATES)
    
    # Temporal context
    time_period_name = random.choice(list(TIME_PERIODS.keys()))
    time_period = TIME_PERIODS[time_period_name]
    hour = random.choice(list(time_period["hours"]))
    minute = random.randint(0, 59)
    is_peak = time_period["is_peak"]
    day_type = random.choice(DAY_TYPES)
    incident_time_minutes = time_to_minutes(hour, minute)
    
    # Environmental context
    weather = random.choice(list(WEATHER_CONDITIONS.keys()))
    weather_data = WEATHER_CONDITIONS[weather]
    temperature = random.randint(-5, 35)
    visibility = weather_data["visibility_km"]
    wind_speed = random.randint(0, 80) if weather in ["storm", "snow"] else random.randint(0, 30)
    
    # Location (smart selection)
    location = select_incident_location(template, stations, segments, network_stats)
    
    # Find trains at incident location/time (NEW - timetable-based)
    affected_trains = find_trains_at_location(timetable, incident_time_minutes, day_type, location, segments)
    
    # If no trains found, try to find trains nearby or adjust time slightly
    if not affected_trains and location["location_type"] != "network_wide":
        # Try ±15 minutes
        for offset in [-15, -10, -5, 5, 10, 15]:
            adjusted_time = incident_time_minutes + offset
            if 0 <= adjusted_time < 1440:  # Within same day
                affected_trains = find_trains_at_location(timetable, adjusted_time, day_type, location, segments)
                if affected_trains:
                    # Adjust incident time slightly
                    incident_time_minutes = adjusted_time
                    hour = adjusted_time // 60
                    minute = adjusted_time % 60
                    break
    
    # Build train positions dictionary
    train_positions = {}
    for train in affected_trains:
        train_positions[train["train_id"]] = {
            "position_type": train["position_type"],
            "station_id": train.get("station_id"),
            "segment_id": train.get("segment_id"),
            "progress_pct": train.get("progress_pct", 0.0),
            "distance_km": train.get("distance_km", 0.0)
        }
    
    # If still no trains, create a minimal set for network-wide incidents
    if not affected_trains and location["location_type"] == "network_wide":
        # Sample some trains from timetable
        sample_trains = random.sample(timetable, min(5, len(timetable)))
        for train_data in sample_trains:
            pos = get_train_position_at_time(train_data, incident_time_minutes, day_type, segments)
            if pos:
                affected_trains.append({
                    **pos,
                    "service_type": train_data.get("service_type", "unknown"),
                    "route": train_data.get("route", "unknown")
                })
                train_positions[train_data["train_id"]] = {
                    "position_type": pos["position_type"],
                    "station_id": pos.get("station_id"),
                    "segment_id": pos.get("segment_id"),
                    "progress_pct": pos.get("progress_pct", 0.0),
                    "distance_km": pos.get("distance_km", 0.0)
                }
    
    # Severity
    severity = template["base_severity"] + weather_data["severity_mod"]
    if is_peak:
        severity += 1
    if location["is_junction"]:
        severity += 1
    severity = max(1, min(5, severity))
    
    # Network load
    network_load = time_period["load_pct"] + random.randint(-10, 10)
    network_load = max(10, min(100, network_load))
    
    # Cascade calculation
    min_cascade, max_cascade = template["cascade_depth_range"]
    cascade_depth = random.randint(min_cascade, max_cascade)
    if is_peak:
        cascade_depth += 1
    if location["is_junction"]:
        cascade_depth += 1
    cascade_depth = min(cascade_depth, 5)
    
    # Affected trains (use actual count from timetable, with minimum)
    trains_affected = max(len(affected_trains), random.randint(1, cascade_depth * 3))
    
    # Delay calculation
    base_delay = severity * 15
    cascade_mult = 1 + (cascade_depth * 0.3)
    peak_mult = 1.5 if is_peak else 1.0
    estimated_delay = int(base_delay * cascade_mult * peak_mult / weather_data["speed_impact"])
    estimated_delay += random.randint(-10, 20)
    
    # Context for description
    context = {
        "weather": weather,
        "time_label": time_period["label"],
        "trains_affected": trains_affected
    }
    
    # Generate description (use actual train IDs if available)
    context["trains_affected"] = trains_affected
    if affected_trains:
        context["train_id"] = affected_trains[0]["train_id"]
    description = generate_semantic_description(template, location, context)
    
    # Resolution (NEW - physically feasible)
    resolution_strategy = random.choice(template["resolution_strategies"])
    resolution_code = f"{resolution_strategy}_{uuid.uuid4().hex[:8]}"
    
    # Generate physically feasible resolution
    weather_speed_mod = weather_data["speed_impact"]
    resolution_data = generate_physically_feasible_resolution(
        template["type"], location, affected_trains, segments, stations, 
        weather_speed_mod, resolution_strategy
    )
    actions = resolution_data["actions"]
    
    # Rate the solution
    solution_rating = rate_solution(resolution_data, estimated_delay, trains_affected, is_peak, severity)
    
    # Outcome
    if is_golden:
        actual_delay = int(estimated_delay * random.uniform(0.3, 0.6))
        outcome_score = random.uniform(0.9, 1.0)
        satisfaction = "High"
        rating = "thumbs_up"
    else:
        actual_delay = int(estimated_delay * random.uniform(0.6, 1.4))
        delay_ratio = actual_delay / max(estimated_delay, 1)
        
        if delay_ratio < 0.7:
            outcome_score = random.uniform(0.8, 0.95)
            satisfaction = "High"
            rating = "thumbs_up"
        elif delay_ratio < 1.2:
            outcome_score = random.uniform(0.6, 0.8)
            satisfaction = "Medium"
            rating = "neutral"
        else:
            outcome_score = random.uniform(0.3, 0.6)
            satisfaction = "Low"
            rating = "thumbs_down"
    
    # Technical details
    affected_platforms = []
    if location["stations"]:
        for station in location["stations"]:
            platform_count = station.get("platforms", 2)
            affected_platforms.extend(random.sample(range(1, platform_count + 1), random.randint(1, min(3, platform_count))))
    
    track_circuits = [f"TC_{random.randint(100, 999)}" for _ in range(random.randint(1, 4))]
    
    rolling_stock = random.choice(template.get("stock_types", [None])) if "stock_types" in template else None
    
    # Operator context (for Gemini enrichment later)
    maintenance_notes = random.choice([
        "No scheduled maintenance in area",
        "Track inspection completed 3 days ago - all clear",
        "Known minor signal intermittency reported last week",
        "Recent overhead line maintenance - monitoring required"
    ])
    
    operator_logs = random.choice([
        f"Alex: Initial report received at {hour:02d}:{random.randint(0, 59):02d}",
        f"Multiple driver reports of {template['type'].replace('_', ' ')}",
        f"Control room notified by automated system alert"
    ])
    
    # Create incident
    incident = Incident(
        incident_id=str(uuid.uuid4()),
        type=template["type"],
        archetype=template["archetype"],
        semantic_description=description,
        
        location_id=location["location_id"],
        location_name=location["location_name"],
        location_type=location["location_type"],
        station_ids=location["station_ids"],
        segment_id=location["segment_id"],
        zone=location["zone"],
        is_junction=location["is_junction"],
        has_switches=location["has_switches"],
        
        timestamp=(datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0) - timedelta(days=random.randint(1, 365))).isoformat(),
        hour_of_day=hour,
        time_of_day=time_period_name,
        day_type=day_type,
        is_peak=is_peak,
        
        weather_condition=weather,
        temperature_c=temperature,
        visibility_km=round(visibility, 2),
        wind_speed_kmh=wind_speed,
        
        severity_level=severity,
        network_load_pct=network_load,
        trains_affected_count=trains_affected,
        cascade_depth=cascade_depth,
        estimated_delay_minutes=estimated_delay,
        
        affected_platforms=affected_platforms,
        track_circuits_affected=track_circuits,
        rolling_stock_type=rolling_stock,
        
        affected_trains=[{
            "train_id": t["train_id"],
            "service_type": t.get("service_type", "unknown"),
            "position_type": t["position_type"],
            "location": t.get("station_id") or t.get("segment_id", "unknown")
        } for t in affected_trains],
        train_positions_at_incident=train_positions,
        
        resolution_code=resolution_code,
        resolution_strategy=resolution_strategy,
        actions_taken=actions,
        solution_rating=solution_rating,
        
        actual_delay_minutes=actual_delay,
        passenger_satisfaction=satisfaction,
        outcome_score=round(outcome_score, 3),
        operator_rating=rating,
        
        maintenance_notes=maintenance_notes,
        operator_logs=operator_logs,
        
        is_golden_run=is_golden,
        similar_incident_ids=[],
        created_at=datetime.now().isoformat()
    )
    
    return incident


# ============================================================================
# MAIN
# ============================================================================
def main():
    """
    Main execution: Load network, generate incidents, save to JSON.
    """
    
    # Load network topology
    print("\n📍 Loading network data...")
    try:
        stations, segments, network_stats = load_network_data()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n⚠️  STOPPING: Cannot generate incidents without network files.")
        print("   Run generate_network.py first!")
        return
    
    # Load timetable
    print("\n🚂 Loading timetable data...")
    try:
        timetable = load_timetable()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n⚠️  STOPPING: Cannot generate timetable-conforming incidents without timetable file.")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate incidents
    print(f"\n🎲 Generating {TOTAL_INCIDENTS} incidents based on real network topology...")
    incidents = []
    
    for i in range(TOTAL_INCIDENTS):
        incidents.append(generate_incident(stations, segments, network_stats, timetable, is_golden=False))
        
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/{TOTAL_INCIDENTS}")
    
    # Shuffle and split
    random.shuffle(incidents)
    split_idx = int(TOTAL_INCIDENTS * TRAIN_SPLIT)
    train_incidents = incidents[:split_idx]
    test_incidents = incidents[split_idx:]
    
    # Save to JSON
    print(f"\n💾 Saving data...")
    print(f"   Training set: {len(train_incidents)} incidents")
    print(f"   Test set: {len(test_incidents)} incidents")
    
    output_path = OUTPUT_DIR / "incidents.json"
    
    with open(output_path, 'w') as f:
        json.dump({
            "metadata": {
                "total_incidents": TOTAL_INCIDENTS,
                "train_count": len(train_incidents),
                "test_count": len(test_incidents),
                "generated_at": datetime.now().isoformat(),
                "network_stats": {
                    "stations": len(stations),
                    "segments": len(segments),
                    "junctions": len(network_stats.get("junctions", [])),
                    "train_services": len(timetable)
                }
            },
            "train": [asdict(inc) for inc in train_incidents],
            "test": [asdict(inc) for inc in test_incidents]
        }, f, indent=2)
    
    # Print statistics
    print(f"\n✅ Saved to {output_path}")
    print("\n📊 Dataset Statistics:")
    print(f"   Average severity: {np.mean([inc.severity_level for inc in incidents]):.2f}")
    print(f"   Average outcome score: {np.mean([inc.outcome_score for inc in incidents]):.2f}")
    
    print("\n   Incident type distribution:")
    type_counts = {}
    for inc in incidents:
        type_counts[inc.type] = type_counts.get(inc.type, 0) + 1
    for itype, count in sorted(type_counts.items()):
        print(f"      {itype}: {count} ({count/TOTAL_INCIDENTS*100:.1f}%)")
    
    print("\n   Location type distribution:")
    location_counts = {}
    for inc in incidents:
        loc_type = "junction" if inc.is_junction else inc.location_type
        location_counts[loc_type] = location_counts.get(loc_type, 0) + 1
    for loc_type, count in sorted(location_counts.items()):
        print(f"      {loc_type}: {count} ({count/TOTAL_INCIDENTS*100:.1f}%)")
    
    print("\n✨ Generation complete!")
    print("   Ready for embeddings & Qdrant upload")
    print(f"\n📈 Solution Rating Statistics:")
    ratings = [inc.solution_rating.get("overall_score", 0) for inc in incidents if hasattr(inc, 'solution_rating')]
    if ratings:
        print(f"   Average solution rating: {np.mean(ratings):.3f}")
        print(f"   Excellent solutions (≥0.8): {sum(1 for r in ratings if r >= 0.8)} ({sum(1 for r in ratings if r >= 0.8)/len(ratings)*100:.1f}%)")
        print(f"   Feasible solutions (≥0.5): {sum(1 for r in ratings if r >= 0.5)} ({sum(1 for r in ratings if r >= 0.5)/len(ratings)*100:.1f}%)")


if __name__ == "__main__":
    main()


