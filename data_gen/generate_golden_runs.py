#!/usr/bin/env python3
"""
Golden Runs Generator for Rail Network Accidents

- Reads network data from data/network/{stations.json, segments.json, timetable.json}
- Generates N realistic accident incidents that reference actual segments and trains
- Optionally attaches best-practice solution templates per accident type

Usage:
  python data_gen/generate_golden_runs.py \
    --count 50 \
    --start 2025-06-01 \
    --end 2026-05-31 \
    --with-solutions \
    --out data/processed/golden_runs_accidents_generated.json

Notes:
- Affected trains are selected from services that stop at either end of the segment.
- Coordinates are interpolated along the segment between station coordinates.
- Metrics and severity are sampled from distributions per accident type.
"""
import argparse
import json
import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

RNG = random.Random(42)

ACCIDENT_TYPES = [
    "derailment",
    "collision",
    "signal_failure",
    "track_defect",
    "track_obstruction",
    "electrical_failure",
    "brake_failure",
    "wheel_bearing_failure",
    "switch_malfunction",
    "electrical_fire",
]

WEATHER = ["clear", "rainy", "foggy", "windy", "storm"]

# Best-practice solutions per type (same schema used in processed file)
SOLUTION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    'collision': {
        'description': 'Implementation of predictive collision avoidance system (PCAS) with automatic braking at 80% warning threshold, installation of advanced signaling with 3-section blocks instead of 2-section, retraining of all drivers on emergency procedures',
        'implementation_time_hours': 72,
        'cost_estimate': 850000,
        'implementation_steps': [
            'Install automatic train protection (ATP) system on all main lines',
            'Upgrade signal boxes with enhanced detection equipment',
            'Reduce train spacing by implementing 3-aspect signaling',
            'Conduct emergency brake testing on all trains',
            'Provide driver retraining on PCAS operation',
            'Perform system validation with test trains'
        ],
        'preventive_measures': [
            'Deploy GPS-based real-time train tracking system',
            'Establish mandatory speed limits before junctions (80 km/h)',
            'Implement continuous audible warning system',
            'Create backup communication protocols between signallers'
        ],
        'estimated_prevention_cost': 1200000
    },
    'derailment': {
        'description': 'Complete track realignment with ultrasonic inspection, replacement of affected rail sections, wheel condition assessment and locomotive overhaul',
        'implementation_time_hours': 48,
        'cost_estimate': 480000,
        'implementation_steps': [
            'Clear debris and secure accident scene',
            'Perform detailed track geometry survey',
            'Remove and replace damaged rail sections',
            'Conduct ultrasonic flaw detection on adjacent track',
            'Test wheel integrity of all locomotives',
            'Perform dynamic testing before service resumption'
        ],
        'preventive_measures': [
            'Increase track inspection frequency to weekly for main lines',
            'Implement accelerometer-based condition monitoring system',
            'Establish speed reduction zones in wet conditions',
            'Upgrade drainage systems to prevent water accumulation'
        ],
        'estimated_prevention_cost': 250000
    },
    'signal_failure': {
        'description': 'Installation of solid-state signal control system with redundant backup power supplies and manual signal operation protocols',
        'implementation_time_hours': 36,
        'cost_estimate': 620000,
        'implementation_steps': [
            'Activate manual signal override procedures',
            'Perform immediate inspection of signal electronics',
            'Replace faulty signal modules with redundant units',
            'Install uninterruptible power supplies (UPS) for signal boxes',
            'Conduct signal system diagnostics',
            'Test all signal aspects before resuming service'
        ],
        'preventive_measures': [
            'Implement dual-channel signaling with automatic failover',
            'Establish quarterly preventive maintenance schedule',
            'Deploy condition monitoring on all critical signals',
            'Create standby signal technician roster for emergencies'
        ],
        'estimated_prevention_cost': 890000
    },
    'track_obstruction': {
        'description': 'Immediate debris removal and track inspection, installation of track-side obstacle detection cameras, establishment of rapid response cleanup team',
        'implementation_time_hours': 2,
        'cost_estimate': 85000,
        'implementation_steps': [
            'Deploy rapid response team with specialized track clearing equipment',
            'Clear debris and vegetation from track',
            'Perform visual and tactile track inspection for damage',
            'Verify track geometry and alignment',
            'Test train passage at reduced speed before resumption'
        ],
        'preventive_measures': [
            'Install automated obstacle detection system using LIDAR',
            'Implement quarterly vegetation clearance program',
            'Establish buffer zones with fencing along sensitive sections',
            'Deploy drones for monthly track-side condition surveillance'
        ],
        'estimated_prevention_cost': 340000
    },
    'electrical_failure': {
        'description': 'Replacement of damaged catenary sections with reinforced design, installation of redundant feed points, implementation of automatic sectioning to isolate faults',
        'implementation_time_hours': 40,
        'cost_estimate': 720000,
        'implementation_steps': [
            'De-energize affected sections of catenary',
            'Inspect support structures for damage',
            'Replace damaged catenary wire sections',
            'Upgrade insulator specifications for wind resistance',
            'Install additional mechanical supports at 50m intervals',
            'Test electrical load capacity before resumption'
        ],
        'preventive_measures': [
            'Upgrade to high-strength catenary wire rated for 150 km/h+ winds',
            'Install automated weather monitoring with wind speed sensors',
            'Establish speed restrictions when winds exceed 80 km/h',
            'Implement redundant power feed from secondary substations'
        ],
        'estimated_prevention_cost': 950000
    },
    'wheel_bearing_failure': {
        'description': 'Replacement of all wheel bearings with sealed high-grade units, installation of temperature monitoring sensors, establishment of predictive maintenance program',
        'implementation_time_hours': 24,
        'cost_estimate': 420000,
        'implementation_steps': [
            'Remove train from service and lift on maintenance jacks',
            'Extract and inspect all wheel bearing assemblies',
            'Replace all bearings with sealed high-grade units',
            'Install temperature sensors on critical bearing points',
            'Perform dynamic balancing of all wheel sets',
            'Test train on track at increasing speeds'
        ],
        'preventive_measures': [
            'Deploy wireless temperature sensors on all trains',
            'Establish mandatory bearing replacement at 500,000 km',
            'Implement daily bearing temperature checks during peak season',
            'Create predictive model using bearing temperature trends'
        ],
        'estimated_prevention_cost': 580000
    },
    'switch_malfunction': {
        'description': 'Complete switch mechanism replacement with modern hydraulic actuator system, installation of redundant position detection, implementation of automated switch lubrication',
        'implementation_time_hours': 18,
        'cost_estimate': 340000,
        'implementation_steps': [
            'Install mechanical lock-out/tag-out (LOTO) procedures',
            'Remove faulty switch mechanism',
            'Install new hydraulic-actuated switch system',
            'Add redundant position sensors for fail-safe operation',
            'Install automated lubrication system for moving parts',
            'Perform dynamic switch tests with live trains'
        ],
        'preventive_measures': [
            'Replace all switches >20 years old with modern designs',
            'Implement automatic lubrication on all switches',
            'Deploy condition monitoring on switch wear patterns',
            'Establish weekly switch operation verification tests'
        ],
        'estimated_prevention_cost': 520000
    },
    'track_defect': {
        'description': 'Rail replacement with modern continuous welded rail (CWR), ultrasonic inspection of entire segment, installation of thermal monitoring systems',
        'implementation_time_hours': 44,
        'cost_estimate': 680000,
        'implementation_steps': [
            'Perform comprehensive ultrasonic scan of entire track section',
            'Mark all defective sections exceeding threshold',
            'Replace rail sections with high-strength CWR',
            'Install stress relief joints at critical points',
            'Inspect all welds with eddy current testing',
            'Perform thermal load testing under peak conditions'
        ],
        'preventive_measures': [
            'Implement automated rail flaw detection vehicle surveys monthly',
            'Deploy thermal imaging to detect stress-induced weak points',
            'Establish speed restrictions during temperature extremes',
            'Upgrade to premium rail grade in high-stress areas'
        ],
        'estimated_prevention_cost': 750000
    },
    'brake_failure': {
        'description': 'Complete brake system overhaul with redundancy emphasis, installation of secondary pneumatic braking system, automatic brake effectiveness testing protocol',
        'implementation_time_hours': 32,
        'cost_estimate': 680000,
        'implementation_steps': [
            'Conduct comprehensive brake system inspection on all cars',
            'Replace faulty brake components and cylinders',
            'Install secondary pneumatic brake system independent of primary',
            'Install load-sensing valve for automatic brake force adjustment',
            'Implement automatic brake test before each service',
            'Establish dynamic braking performance validation'
        ],
        'preventive_measures': [
            'Deploy predictive brake wear monitoring on all trains',
            'Establish mandatory brake system inspection every 2 weeks',
            'Implement automatic brake testing before departure',
            'Create alerting system for brake degradation trends'
        ],
        'estimated_prevention_cost': 920000
    },
    'electrical_fire': {
        'description': 'Replacement of all wiring with fire-resistant insulation, installation of automatic fire detection and suppression systems',
        'implementation_time_hours': 40,
        'cost_estimate': 580000,
        'implementation_steps': [
            'Remove and inspect all electrical wiring and connectors',
            'Replace wiring with polyimide-insulated cables rated for 200°C',
            'Install thermal sensors throughout motor compartment',
            'Add automatic fire suppression using CO2 system',
            'Upgrade ventilation fans and air intake filters',
            'Perform electrical load tests to verify system integrity'
        ],
        'preventive_measures': [
            'Implement monthly electrical compartment inspections',
            'Deploy thermal imaging cameras during maintenance',
            'Establish maximum operating temperature limits with automatic shutoff',
            'Install automatic electrical isolation switches on fault detection'
        ],
        'estimated_prevention_cost': 720000
    }
}


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=50, help='Number of incidents to generate')
    p.add_argument('--start', type=str, default='2025-06-01', help='Start date (YYYY-MM-DD)')
    p.add_argument('--end', type=str, default='2026-05-31', help='End date (YYYY-MM-DD)')
    p.add_argument('--out', type=str, default='data/processed/golden_runs_accidents_generated.json', help='Output JSON path')
    p.add_argument('--with-solutions', action='store_true', help='Attach solution templates to incidents')
    return p.parse_args()


def pick_severity(acc_type: str) -> str:
    weights = {
        'derailment': ("critical", 0.5),
        'collision': ("critical", 0.6),
        'signal_failure': ("medium", 0.7),
        'track_defect': ("high", 0.5),
        'track_obstruction': ("medium", 0.7),
        'electrical_failure': ("high", 0.6),
        'brake_failure': ("high", 0.6),
        'wheel_bearing_failure': ("medium", 0.6),
        'switch_malfunction': ("medium", 0.7),
        'electrical_fire': ("high", 0.6),
    }
    base, prob = weights.get(acc_type, ("medium", 0.6))
    if RNG.random() < prob:
        return base
    return 'critical' if base == 'high' and RNG.random() < 0.3 else ('high' if base == 'medium' and RNG.random() < 0.4 else base)


def rand_ts(start: datetime, end: datetime) -> str:
    delta = end - start
    seconds = RNG.randint(0, int(delta.total_seconds()))
    ts = start + timedelta(seconds=seconds)
    return ts.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')


def interp(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    return (round(a[0] + (b[0] - a[0]) * t, 2), round(a[1] + (b[1] - a[1]) * t, 2))


def build_station_maps(stations: List[Dict[str, Any]]):
    by_id = {s['id']: s for s in stations}
    coords = {s['id']: (float(s['coordinates'][0]), float(s['coordinates'][1])) for s in stations}
    return by_id, coords


def normalize_timetable(timetable: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Return train_id -> sequence of station_ids (weekday first occurrence order)."""
    trains: Dict[str, List[str]] = {}
    for svc in timetable:
        tid = svc.get('train_id')
        stops = svc.get('stops', [])
        # Keep first occurrence per station following listed order, prefer weekday entries.
        seen = set()
        ordered = []
        for st in stops:
            if st.get('daytype') != 'weekday':
                continue
            sid = st.get('station_id')
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        # If no weekday entries, fallback to first occurrences regardless of daytype
        if not ordered:
            seen = set()
            for st in stops:
                sid = st.get('station_id')
                if sid and sid not in seen:
                    seen.add(sid)
                    ordered.append(sid)
        trains[tid] = ordered
    return trains


def trains_touching_segment(trains_map: Dict[str, List[str]], from_st: str, to_st: str, limit: int = 2) -> List[str]:
    candidates = []
    for tid, seq in trains_map.items():
        if from_st in seq or to_st in seq:
            candidates.append(tid)
    RNG.shuffle(candidates)
    return candidates[:max(1, limit)]


def impact_minutes_by_severity(sev: str) -> int:
    return {
        'critical': RNG.randint(180, 360),
        'high': RNG.randint(90, 240),
        'medium': RNG.randint(30, 120)
    }[sev]


def metrics_for(acc_type: str, sev: str, is_critical_seg: bool) -> Dict[str, Any]:
    base = {
        'critical': (35, (12, 24), (4, 8)),
        'high': (20, (8, 16), (2, 6)),
        'medium': (8, (3, 8), (0, 3))
    }[sev]
    ndp = base[0] + (10 if is_critical_seg else 0)
    tl = RNG.randint(*base[1])
    tc = RNG.randint(*base[2])
    return {
        'network_disruption_percentage': min(ndp, 95),
        'trains_delayed': tl,
        'trains_cancelled': tc,
        'alternative_routes_available': 0 if is_critical_seg else RNG.randint(0, 2),
        'recovery_time_hours': round(impact_minutes_by_severity(sev) / 60.0, 1)
    }


def description_for(acc_type: str, affected: List[str], segment_id: str) -> str:
    if acc_type == 'collision' and len(affected) >= 2:
        return f"Train {affected[0]} collision with {affected[1]} on segment {segment_id}"
    if acc_type == 'derailment' and affected:
        return f"Train {affected[0]} derailed due to track issue on {segment_id}"
    if acc_type == 'signal_failure':
        return f"Signal failure on {segment_id} causing route conflicts"
    if acc_type == 'track_defect':
        return f"Detected rail defect on {segment_id} affecting services"
    if acc_type == 'track_obstruction':
        return f"Obstruction reported on {segment_id} requiring immediate clearance"
    if acc_type == 'electrical_failure':
        return f"Overhead catenary failure on {segment_id} disrupting power supply"
    if acc_type == 'brake_failure' and affected:
        return f"Brake failure reported on train {affected[0]} near {segment_id}"
    if acc_type == 'wheel_bearing_failure' and affected:
        return f"Elevated wheel bearing temperatures on train {affected[0]} near {segment_id}"
    if acc_type == 'switch_malfunction':
        return f"Switch malfunction on {segment_id} causing routing delays"
    if acc_type == 'electrical_fire' and affected:
        return f"Electrical fire contained on train {affected[0]} near {segment_id}"
    return f"Incident of type {acc_type} on {segment_id}"


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(__file__))
    stations = load_json(os.path.join(root, 'data', 'network', 'stations.json'))
    segments = load_json(os.path.join(root, 'data', 'network', 'segments.json'))
    timetable = load_json(os.path.join(root, 'data', 'network', 'timetable.json'))

    station_by_id, coords = build_station_maps(stations)
    trains_map = normalize_timetable(timetable)

    start_dt = datetime.strptime(args.start, '%Y-%m-%d')
    end_dt = datetime.strptime(args.end, '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1)

    incidents: List[Dict[str, Any]] = []

    for idx in range(1, args.count + 1):
        seg = RNG.choice(segments)
        seg_id = seg['id']
        from_st = seg['from_station']
        to_st = seg['to_station']
        from_xy = coords[from_st]
        to_xy = coords[to_st]
        t = RNG.random()
        x, y = interp(from_xy, to_xy, t)

        acc_type = RNG.choice(ACCIDENT_TYPES)
        sev = pick_severity(acc_type)
        ts = rand_ts(start_dt, end_dt)

        affected = trains_touching_segment(trains_map, from_st, to_st, limit=2)

        speed_limit = seg.get('speed_limit', 100)
        speed_at = max(10, int(RNG.normalvariate(speed_limit * 0.85, speed_limit * 0.1)))

        impact_minutes = impact_minutes_by_severity(sev)
        metrics = metrics_for(acc_type, sev, bool(seg.get('is_critical')))

        inc = {
            'incident_id': f"INC_{idx:03d}",
            'timestamp': ts,
            'location': {
                'segment_id': seg_id,
                'from_station': from_st,
                'to_station': to_st,
                'coordinates': [round(x, 2), round(y, 2)],
            },
            'accident_type': acc_type,
            'severity': sev,
            'description': description_for(acc_type, affected, seg_id),
            'affected_trains': affected,
            'impact_duration_minutes': impact_minutes,
            'casualties': RNG.randint(0, 20) if sev in ('high', 'critical') and acc_type in ('derailment', 'collision', 'electrical_fire') else RNG.randint(0, 5),
            'affected_passengers': RNG.randint(50, 800),
            'weather_conditions': RNG.choice(WEATHER),
            'speed_at_incident': speed_at,
            'golden_run_metrics': metrics,
        }
        if args.with_solutions and acc_type in SOLUTION_TEMPLATES:
            sol = dict(SOLUTION_TEMPLATES[acc_type])
            sol['reference_accident_types'] = [acc_type]
            inc['solution'] = sol

        incidents.append(inc)

    os.makedirs(os.path.dirname(os.path.join(root, args.out)), exist_ok=True)
    with open(os.path.join(root, args.out), 'w', encoding='utf-8') as f:
        json.dump(incidents, f, indent=2)

    print(f"Generated {len(incidents)} incidents -> {args.out}")


if __name__ == '__main__':
    main()
