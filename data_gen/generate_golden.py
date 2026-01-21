import json
import random

# Set seed for reproducibility so the same "random" runs are generated every time
random.seed(42)

# Define network entities based on the provided generate_network.py structure
hubs = [
    {"id": "STN_001", "name": "Central Station"},
    {"id": "STN_002", "name": "North Terminal"},
    {"id": "STN_003", "name": "South Junction"},
    {"id": "STN_004", "name": "East Exchange"},
    {"id": "STN_005", "name": "West Hub"}
]

regionals = [
    {"id": "STN_006", "name": "Park Station"},
    {"id": "STN_007", "name": "Grove Station"},
    {"id": "STN_008", "name": "Heights Station"},
    {"id": "STN_009", "name": "Bridge Station"},
    {"id": "STN_010", "name": "Valley Station"},
    {"id": "STN_011", "name": "Hill Station"},
    {"id": "STN_012", "name": "Lake Station"},
    {"id": "STN_013", "name": "River Station"},
    {"id": "STN_014", "name": "Forest Station"},
    {"id": "STN_015", "name": "Meadow Station"}
]

# Basic infrastructure elements
segments = ["A", "B", "C", "D", "E"]
platforms = [f"Platform {i}" for i in range(1, 13)]
train_prefixes = ["T1", "T2", "T3", "R1", "R2", "E1", "E2", "L1", "L2", "X1"]

# Define templates for different incident scenarios and their perfect resolutions
resolution_templates = [
    {
        "incident": "Total signal loss at {hub}",
        "time": "Peak AM",
        "action": "Rerouted all {prefix} prefix trains to {platform} via Segment {seg}",
        "code": "REROUTE_{hub_code}_{platform_code}_VIA_{seg}",
        "delay_base": 15,
        "satisfaction": "High"
    },
    {
        "incident": "Switch malfunction at {hub}",
        "time": "Peak PM",
        "action": "Isolated faulty switch, directed {prefix} trains to alternate {platform}",
        "code": "ISOLATE_SWITCH_{hub_code}_{platform_code}",
        "delay_base": 10,
        "satisfaction": "High"
    },
    {
        "incident": "Platform equipment failure at {regional}",
        "time": "Midday",
        "action": "Bypassed {regional}, {prefix} trains skip-stopped to {hub}",
        "code": "BYPASS_{regional_code}_TO_{hub_code}",
        "delay_base": 8,
        "satisfaction": "High"
    },
    {
        "incident": "Track obstruction on Segment {seg} between {hub} and {regional}",
        "time": "Early morning",
        "action": "Activated alternate loop route via {regional2}",
        "code": "LOOP_VIA_{regional2_code}",
        "delay_base": 15,
        "satisfaction": "High"
    },
    {
        "incident": "Power system fault at {regional}",
        "time": "Late evening",
        "action": "Diverted {prefix} services to diesel routing via {regional2}",
        "code": "DIESEL_ROUTE_{regional_code}_VIA_{regional2_code}",
        "delay_base": 8,
        "satisfaction": "High"
    },
    {
        "incident": "Communication system loss at {hub}",
        "time": "Peak AM",
        "action": "Manual signal authorization, reduced capacity on {platform}",
        "code": "MANUAL_AUTH_{hub_code}_{platform_code}",
        "delay_base": 12,
        "satisfaction": "High"
    },
    {
        "incident": "Signal degradation on main line Segment {seg}",
        "time": "Midday",
        "action": "Reduced speed limit to 80 km/h, spacing adjustments on all {prefix} trains",
        "code": "REDUCE_SPEED_{seg}_ALL_{prefix}",
        "delay_base": 7,
        "satisfaction": "High"
    },
    {
        "incident": "Interlocking system fault at {hub}",
        "time": "Peak PM",
        "action": "Emergency manual operations, consolidated arrivals to {platform}",
        "code": "MANUAL_OPS_{hub_code}_CONSOL_{platform_code}",
        "delay_base": 20,
        "satisfaction": "High"
    },
    {
        "incident": "Track maintenance emergency at {regional}",
        "time": "Early morning",
        "action": "Single-track operation, alternating {prefix} trains every 15 min",
        "code": "SINGLE_TRACK_{regional_code}_ALT_{prefix}",
        "delay_base": 11,
        "satisfaction": "High"
    },
    {
        "incident": "Platform overcrowding at {hub}",
        "time": "Peak AM",
        "action": "Split arrivals between {platform} and {platform2}, extra {prefix} service deployed",
        "code": "SPLIT_ARRIVALS_{hub_code}_{platform_code}_{platform2_code}",
        "delay_base": 5,
        "satisfaction": "High"
    }
]

golden_runs = []

# Generate 50 incidents
for i in range(50):
    # Rotate through templates to ensure even coverage
    template = resolution_templates[i % len(resolution_templates)]
    
    # Select random entities for this specific run
    hub = random.choice(hubs)
    regional = random.choice(regionals)
    regional2 = random.choice([r for r in regionals if r != regional])
    seg_id = random.choice(segments)
    platform = random.choice(platforms)
    platform2 = random.choice([p for p in platforms if p != platform])
    prefix = random.choice(train_prefixes)
    
    # Generate short codes for the resolution string
    hub_code = hub["name"].split()[0].upper()[:4]
    regional_code = regional["name"].split()[0].upper()[:4]
    regional2_code = regional2["name"].split()[0].upper()[:4]
    platform_code = f"P{platform.split()[-1]}"
    platform2_code = f"P{platform2.split()[-1]}"
    
    # Construct the semantic description
    description = template["incident"].format(
        hub=hub["name"],
        regional=regional["name"],
        regional2=regional2["name"],
        seg=seg_id,
        prefix=prefix
    ) + ". " + template["time"] + ". " + template["action"].format(
        hub=hub["name"],
        regional=regional["name"],
        regional2=regional2["name"],
        seg=seg_id,
        platform=platform,
        platform2=platform2,
        prefix=prefix
    )
    
    # Construct the resolution code
    resolution_code = template["code"].format(
        hub_code=hub_code,
        regional_code=regional_code,
        regional2_code=regional2_code,
        seg=seg_id,
        platform_code=platform_code,
        platform2_code=platform2_code,
        prefix=prefix
    )
    
    # Create the record
    golden_run = {
        "incident_id": f"inc_golden_{i+1:02d}",
        "semantic_description": description,
        "resolution_code": resolution_code,
        "outcome_metrics": {
            "delay_minutes": random.randint(template["delay_base"] - 2, template["delay_base"] + 2),
            "passenger_satisfaction": template["satisfaction"]
        },
        "is_golden": True
    }
    
    golden_runs.append(golden_run)

# Output results to a file
with open('golden_runs.json', 'w') as f:
    json.dump(golden_runs, f, indent=2)

print(f"Generated {len(golden_runs)} golden runs.")
