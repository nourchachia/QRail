"""
Check what's actually in Qdrant - are golden runs there? What do their descriptions look like?
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

print("="*70)
print("INVESTIGATING QDRANT CONTENTS")
print("="*70)

# 1. Check total points
info = client.get_collection("operational_memory")
print(f"\nTotal points in collection: {info.points_count}")

# 2. Check how many golden runs
print("\n" + "-"*70)
print("Checking for Golden Runs...")
print("-"*70)

golden_filter = Filter(
    must=[
        FieldCondition(
            key='is_golden_run',
            match=MatchValue(value=True)
        )
    ]
)

golden_results = client.scroll(
    collection_name='operational_memory',
    scroll_filter=golden_filter,
    limit=10
)

golden_count = len(golden_results[0])
print(f"\nGolden runs found in first 10: {golden_count}")

if golden_results[0]:
    print("\nFirst 3 Golden Runs:")
    for i, point in enumerate(golden_results[0][:3], 1):
        payload = point.payload
        print(f"\n{i}. ID: {payload.get('incident_id', 'NO ID')}")
        print(f"   Type: {payload.get('accident_type', 'unknown')}")
        print(f"   Description: {payload.get('description', 'NO DESC')[:100]}...")
        print(f"   Stations: {payload.get('station_ids', [])}")
        print(f"   Is Golden: {payload.get('is_golden_run')}")
else:
    print("\nNO GOLDEN RUNS FOUND!")
    print("This is the problem - uploader didn't mark them as golden")

# 3. Get ANY random incidents to see what they look like
print("\n" + "-"*70)
print("Sample of ANY incidents in database:")
print("-"*70)

random_results = client.scroll(
    collection_name='operational_memory',
    limit=5
)

for i, point in enumerate(random_results[0][:5], 1):
    payload = point.payload
    print(f"\n{i}. ID: {payload.get('incident_id', 'NO ID')}")
    print(f"   Type: {payload.get('accident_type', 'unknown')}")  
    print(f"   Golden: {payload.get('is_golden_run', False)}")
    print(f"   Desc: {payload.get('description', 'NO DESC')[:80]}...")

print("\n" + "="*70)
