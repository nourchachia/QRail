import json
import os

def generate_static_data():
    # Base paths relative to this script location (data_gen/)
    # script is in c:\Users\ASUS\Desktop\projects2025\QRail\data_gen
    # root is ..
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    stations_path = os.path.join(project_root, "data", "network", "stations.json")
    segments_path = os.path.join(project_root, "data", "network", "segments.json")
    
    # Output to frontend js folder
    output_path = os.path.join(project_root, "src", "frontend", "js", "static-data.js")

    print(f"Reading {stations_path}...")
    with open(stations_path, 'r', encoding='utf-8') as f:
        stations = json.load(f)

    print(f"Reading {segments_path}...")
    with open(segments_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    js_content = f"""/**
 * STATIC NETWORK DATA
 * Generated from data/network/stations.json and segments.json
 * Guarantees correct topology without relying on API or random generation.
 */

window.staticNetworkData = {{
    stations: {json.dumps(stations, indent=2)},
    segments: {json.dumps(segments, indent=2)}
}};

console.log('✅ Static network data loaded:', window.staticNetworkData.stations.length, 'stations');
"""

    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print("Done!")

if __name__ == "__main__":
    generate_static_data()
