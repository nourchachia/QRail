"""
Neural Rail Conductor - Network Topology Generator
Generates a synthetic rail network with 50 stations and 70 segments
optimized for demonstrating the 5 incident archetypes.
"""

import json
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Station:
    """Represents a railway station node"""
    id: str
    name: str
    type: str  # 'major_hub', 'regional', 'local', 'minor_halt'
    zone: str  # 'core', 'mid', 'peripheral'
    platforms: int
    daily_passengers: int
    coordinates: Tuple[float, float]  # (x, y) in 0-100 space
    connected_segments: List[str]
    has_switches: bool
    is_junction: bool
    
    def to_dict(self):
        d = asdict(self)
        d['connected_segments'] = []  # Will populate after segments
        return d


@dataclass
class Segment:
    """Represents a track segment (edge between stations)"""
    id: str
    from_station: str
    to_station: str
    length_km: float
    speed_limit: int  # km/h
    capacity: int  # trains per hour
    bidirectional: bool
    track_type: str  # 'main_line', 'branch', 'loop', 'siding'
    has_switches: bool
    is_critical: bool  # Part of backbone infrastructure
    electrified: bool
    
    def to_dict(self):
        return asdict(self)


class NetworkGenerator:
    """Generates a synthetic rail network"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.stations: List[Station] = []
        self.segments: List[Segment] = []
        self.station_lookup: Dict[str, Station] = {}
        
    def generate(self) -> Tuple[List[Dict], List[Dict]]:
        """Main generation pipeline"""
        print("🚄 Generating Rail Network...")
        
        # Step 1: Create station hierarchy
        self._generate_stations()
        
        # Step 2: Connect with backbone
        self._create_main_line()
        
        # Step 3: Add branch lines
        self._create_branch_lines()
        
        # Step 4: Add loop/redundancy
        self._create_loop_connections()
        
        # Step 5: Update station metadata
        self._finalize_stations()
        
        # Step 6: Validate network
        self._validate_network()
        
        print(f"✅ Generated {len(self.stations)} stations, {len(self.segments)} segments")
        
        return (
            [s.to_dict() for s in self.stations],
            [seg.to_dict() for seg in self.segments]
        )
    
    def _generate_stations(self):
        """Create 50 stations with realistic hierarchy"""
        
        # === CORE ZONE (5 Major Hubs) ===
        hub_names = ["Central Station", "North Terminal", "South Junction", 
                     "East Exchange", "West Hub"]
        
        for i, name in enumerate(hub_names):
            angle = (i / 5) * 2 * math.pi
            x = 50 + 20 * math.cos(angle)
            y = 50 + 20 * math.sin(angle)
            
            self.stations.append(Station(
                id=f"STN_{i+1:03d}",
                name=name,
                type="major_hub",
                zone="core",
                platforms=random.randint(8, 12),
                daily_passengers=random.randint(80000, 150000),
                coordinates=(x, y),
                connected_segments=[],
                has_switches=True,
                is_junction=True
            ))
        
        # === MID ZONE (15 Regional Stations) ===
        regional_prefixes = ["Park", "Grove", "Heights", "Bridge", "Valley", 
                            "Hill", "Lake", "River", "Forest", "Meadow",
                            "Garden", "Plaza", "Square", "Market", "Cross"]
        
        for i in range(15):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(30, 50)
            x = 50 + radius * math.cos(angle)
            y = 50 + radius * math.sin(angle)
            
            self.stations.append(Station(
                id=f"STN_{i+6:03d}",
                name=f"{regional_prefixes[i]} Station",
                type="regional",
                zone="mid",
                platforms=random.randint(4, 6),
                daily_passengers=random.randint(20000, 60000),
                coordinates=(x, y),
                connected_segments=[],
                has_switches=i < 5,  # First 5 are junctions
                is_junction=i < 5
            ))
        
        # === PERIPHERAL ZONE (30 Local/Minor Stations) ===
        local_count = 20
        minor_count = 10
        
        for i in range(local_count):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(55, 75)
            x = 50 + radius * math.cos(angle)
            y = 50 + radius * math.sin(angle)
            
            self.stations.append(Station(
                id=f"STN_{i+21:03d}",
                name=f"Local Stop {chr(65 + i)}",
                type="local",
                zone="peripheral",
                platforms=random.randint(2, 3),
                daily_passengers=random.randint(5000, 18000),
                coordinates=(x, y),
                connected_segments=[],
                has_switches=False,
                is_junction=False
            ))
        
        for i in range(minor_count):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(78, 95)
            x = 50 + radius * math.cos(angle)
            y = 50 + radius * math.sin(angle)
            
            self.stations.append(Station(
                id=f"STN_{i+41:03d}",
                name=f"Halt {i+1}",
                type="minor_halt",
                zone="peripheral",
                platforms=1,
                daily_passengers=random.randint(500, 4000),
                coordinates=(x, y),
                connected_segments=[],
                has_switches=False,
                is_junction=False
            ))
        
        # Create lookup
        self.station_lookup = {s.id: s for s in self.stations}
    
    def _create_main_line(self):
        """Create high-speed backbone connecting all hubs (ring topology)"""
        hubs = [s for s in self.stations if s.type == "major_hub"]
        
        for i in range(len(hubs)):
            from_hub = hubs[i]
            to_hub = hubs[(i + 1) % len(hubs)]
            
            distance = self._euclidean_distance(from_hub, to_hub)
            
            seg = Segment(
                id=f"SEG_{len(self.segments)+1:03d}",
                from_station=from_hub.id,
                to_station=to_hub.id,
                length_km=round(distance * 2.5, 1),  # Scale factor
                speed_limit=160,  # High-speed
                capacity=20,
                bidirectional=True,
                track_type="main_line",
                has_switches=True,
                is_critical=True,
                electrified=True
            )
            self.segments.append(seg)
    
    def _create_branch_lines(self):
        """Connect regional and local stations to hubs"""
        hubs = [s for s in self.stations if s.type == "major_hub"]
        regionals = [s for s in self.stations if s.type == "regional"]
        locals_minors = [s for s in self.stations if s.type in ["local", "minor_halt"]]
        
        # Connect each regional to nearest hub
        for regional in regionals:
            nearest_hub = min(hubs, key=lambda h: self._euclidean_distance(regional, h))
            distance = self._euclidean_distance(regional, nearest_hub)
            
            seg = Segment(
                id=f"SEG_{len(self.segments)+1:03d}",
                from_station=nearest_hub.id,
                to_station=regional.id,
                length_km=round(distance * 2.0, 1),
                speed_limit=120,
                capacity=12,
                bidirectional=True,
                track_type="branch",
                has_switches=regional.has_switches,
                is_critical=False,
                electrified=True
            )
            self.segments.append(seg)
        
        # Connect locals/minors to nearest regional or hub
        for station in locals_minors:
            candidates = regionals + hubs
            nearest = min(candidates, key=lambda s: self._euclidean_distance(station, s))
            distance = self._euclidean_distance(station, nearest)
            
            seg = Segment(
                id=f"SEG_{len(self.segments)+1:03d}",
                from_station=nearest.id,
                to_station=station.id,
                length_km=round(distance * 1.5, 1),
                speed_limit=80 if station.type == "local" else 60,
                capacity=8 if station.type == "local" else 4,
                bidirectional=True,
                track_type="branch",
                has_switches=False,
                is_critical=False,
                electrified=station.type != "minor_halt"
            )
            self.segments.append(seg)
    
    def _create_loop_connections(self):
        """Add redundant paths (loops) for rerouting scenarios"""
        # Connect some regional stations to each other
        regionals = [s for s in self.stations if s.type == "regional"]
        
        loop_count = 0
        max_loops = 15
        
        for i, reg1 in enumerate(regionals):
            if loop_count >= max_loops:
                break
                
            for reg2 in regionals[i+1:]:
                distance = self._euclidean_distance(reg1, reg2)
                
                # Only connect if reasonably close and not too many segments
                if 15 < distance < 40 and loop_count < max_loops:
                    seg = Segment(
                        id=f"SEG_{len(self.segments)+1:03d}",
                        from_station=reg1.id,
                        to_station=reg2.id,
                        length_km=round(distance * 2.2, 1),
                        speed_limit=100,
                        capacity=10,
                        bidirectional=True,
                        track_type="loop",
                        has_switches=True,
                        is_critical=False,
                        electrified=True
                    )
                    self.segments.append(seg)
                    loop_count += 1
                    
                    if loop_count >= max_loops:
                        break
    
    def _finalize_stations(self):
        """Update station metadata with connected segments"""
        for seg in self.segments:
            # Add to from_station
            from_stn = self.station_lookup[seg.from_station]
            from_stn.connected_segments.append(seg.id)
            
            # Add to to_station
            to_stn = self.station_lookup[seg.to_station]
            to_stn.connected_segments.append(seg.id)
    
    def _validate_network(self):
        """Ensure network meets requirements"""
        issues = []
        
        # Check station counts
        type_counts = {}
        for s in self.stations:
            type_counts[s.type] = type_counts.get(s.type, 0) + 1
        
        print(f"\n📊 Network Statistics:")
        print(f"  Stations by type: {type_counts}")
        print(f"  Total segments: {len(self.segments)}")
        
        segment_types = {}
        for seg in self.segments:
            segment_types[seg.track_type] = segment_types.get(seg.track_type, 0) + 1
        print(f"  Segments by type: {segment_types}")
        
        # Check connectivity
        isolated = [s for s in self.stations if len(s.connected_segments) == 0]
        if isolated:
            issues.append(f"⚠️  {len(isolated)} isolated stations: {[s.id for s in isolated]}")
        
        # Check junctions
        junctions = [s for s in self.stations if s.is_junction]
        print(f"  Junctions: {len(junctions)}")
        
        if issues:
            print("\n⚠️  Validation Issues:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✅ Network validation passed!")
    
    def _euclidean_distance(self, s1: Station, s2: Station) -> float:
        """Calculate distance between two stations"""
        x1, y1 = s1.coordinates
        x2, y2 = s2.coordinates
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def save_to_json(self, output_dir: str = "data/network"):
        """Save network to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stations_data, segments_data = self.generate()
        
        # Save stations
        stations_file = output_path / "stations.json"
        with open(stations_file, 'w') as f:
            json.dump(stations_data, f, indent=2)
        print(f"💾 Saved {len(stations_data)} stations to {stations_file}")
        
        # Save segments
        segments_file = output_path / "segments.json"
        with open(segments_file, 'w') as f:
            json.dump(segments_data, f, indent=2)
        print(f"💾 Saved {len(segments_data)} segments to {segments_file}")
        
        return stations_file, segments_file


def main():
    """Generate and save network"""
    generator = NetworkGenerator(seed=42)
    generator.save_to_json()
    
    print("\n🎉 Network generation complete!")
    print("\nNext steps:")
    print("  1. Review data/network/stations.json")
    print("  2. Review data/network/segments.json")
    print("  3. Run generate_incidents.py to create operational scenarios")


if __name__ == "__main__":
    main()

"""
Network Characteristics - Detailed Specification
📊 Network Overview
Total Nodes: 50 stations
Total Edges: ~70 segments
Topology: Hub-and-spoke with ring backbone + redundant loops
Coordinate System: 2D plane (0-100, 0-100)

🏢 Station Hierarchy (50 Total)
Major Hubs (5 stations)

IDs: STN_001 to STN_005
Names: Central Station, North Terminal, South Junction, East Exchange, West Hub
Type: "major_hub"
Zone: "core"
Platforms: 8-12
Daily Passengers: 80,000-150,000
Coordinates: Arranged in pentagon pattern around center (50, 50) with radius ~20
Properties:

has_switches: true
is_junction: true
Connected by high-speed main line (ring topology)



Regional Stations (15 stations)

IDs: STN_006 to STN_020
Names: Park Station, Grove Station, Heights Station, etc.
Type: "regional"
Zone: "mid"
Platforms: 4-6
Daily Passengers: 20,000-60,000
Coordinates: Scattered at radius 30-50 from center
Properties:

First 5 have is_junction: true (potential bottlenecks)
Connected to nearest hub via branch lines



Local Stops (20 stations)

IDs: STN_021 to STN_040
Names: Local Stop A, Local Stop B, etc.
Type: "local"
Zone: "peripheral"
Platforms: 2-3
Daily Passengers: 5,000-18,000
Coordinates: Radius 55-75 from center
Properties:

has_switches: false
Connected to regional stations or hubs



Minor Halts (10 stations)

IDs: STN_041 to STN_050
Names: Halt 1, Halt 2, etc.
Type: "minor_halt"
Zone: "peripheral"
Platforms: 1
Daily Passengers: 500-4,000
Coordinates: Radius 78-95 (outermost)
Properties:

electrified: false (diesel-only)
Lowest priority in network




🛤️ Segment Types (70 Total)
Main Line (5 segments)

Track Type: "main_line"
Connects: Hub → Hub (ring)
Length: 40-80 km
Speed Limit: 160 km/h (high-speed)
Capacity: 20 trains/hour
Properties:

bidirectional: true
is_critical: true
electrified: true
has_switches: true



Branch Lines (~35 segments)

Track Type: "branch"
Connects: Hub → Regional, Regional → Local
Length: 15-60 km
Speed Limit: 80-120 km/h
Capacity: 8-12 trains/hour
Properties:

bidirectional: true
is_critical: false
Some have switches at junctions



Loop Connections (~15 segments)

Track Type: "loop"
Connects: Regional ↔ Regional (alternate routes)
Length: 25-70 km
Speed Limit: 100 km/h
Capacity: 10 trains/hour
Purpose: Enable rerouting during incidents

Sidings (~15 segments)

Track Type: "siding" (if added)
Connects: Local ↔ Minor Halt
Length: 10-35 km
Speed Limit: 60 km/h
Capacity: 4 trains/hour


📤 Output Files
data/network/stations.json
json[
  {
    "id": "STN_001",
    "name": "Central Station",
    "type": "major_hub",
    "zone": "core",
    "platforms": 10,
    "daily_passengers": 125000,
    "coordinates": [70.0, 50.0],
    "connected_segments": ["SEG_001", "SEG_005", "SEG_012"],
    "has_switches": true,
    "is_junction": true
  },
  // ... 49 more stations
]
Data Types:

id: string (format: STN_XXX)
name: string
type: enum ("major_hub", "regional", "local", "minor_halt")
zone: enum ("core", "mid", "peripheral")
platforms: integer (1-12)
daily_passengers: integer (500-150000)
coordinates: array of 2 floats [x, y]
connected_segments: array of strings (segment IDs)
has_switches: boolean
is_junction: boolean

data/network/segments.json
json[
{
    "id": "SEG_001",
    "from_station": "STN_001",
    "to_station": "STN_002",
    "length_km": 52.3,
    "speed_limit": 160,
    "capacity": 20,
    "bidirectional": true,
    "track_type": "main_line",
    "has_switches": true,
    "is_critical": true,
    "electrified": true
},
  // ... ~69 more segments
]
Data Types:

id: string (format: SEG_XXX)
from_station: string (station ID)
to_station: string (station ID)
length_km: float (10.0-80.0)
speed_limit: integer (60-160 km/h)
capacity: integer (4-20 trains/hour)
bidirectional: boolean
track_type: enum ("main_line", "branch", "loop", "siding")
has_switches: boolean
is_critical: boolean
electrified: boolean


🎯 Network Properties Optimized For:

Signal Failure at Junction: 10 junction nodes with high connectivity
Train Breakdown: Multiple branch lines with single points of failure
Passenger Alarm: High-traffic stations (hubs + regionals)
Severe Weather: Network-wide topology with varying speed limits
Infrastructure Fault: Loop connections enable rerouting demonstrations

The network is designed to be realistic yet controllable for demo purposes, with clear hierarchies and intentional bottlenecks that showcase AI decision-making.
"""