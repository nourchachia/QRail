import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Suppress emoji print errors on Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_API_KEY")

if not url or not key:
    print("ERROR: QDRANT_URL or QDRANT_API_KEY not in .env")
    exit(1)

client = QdrantClient(url=url, api_key=key)

print(f"Connected to Qdrant: {url[:40]}...")
print("\nSearching for INC_003 (signal failure golden run)...\n")

# Search for INC_003
result = client.scroll(
    collection_name='operational_memory',
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key='incident_id',
                match=MatchValue(value='INC_003')
            )
        ]
    ),
    limit=1
)

if result[0]:
    print('SUCCESS: Found INC_003 in Qdrant!')
    point = result[0][0]
    print(f'\nDetails:')
    print(f'  Incident ID: {point.payload.get("incident_id")}')
    print(f'  Description: {point.payload.get("description", "N/A")[:100]}')
    print(f'  Type: {point.payload.get("accident_type", "N/A")}')
    print(f'  Is Golden: {point.payload.get("is_golden_run")}')
    print(f'  Stations: {point.payload.get("station_ids", [])}')
else:
    print('ERROR: INC_003 NOT in Qdrant database!')
    print('\nFix: Run this command to upload golden runs:')
    print('  python src/backend/uploader.py')

# Check total golden runs
print("\n" + "="*50)
print("Checking total golden runs in database...")
golden_result = client.scroll(
    collection_name='operational_memory',
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key='is_golden_run',
                match=MatchValue(value=True)
            )
        ]
    ),
    limit=100
)

golden_count = len(golden_result[0])
print(f"Total golden runs found: {golden_count}")

if golden_count < 50:
    print(f"WARNING: Expected 50 golden runs, found only {golden_count}")
    print("Re-run uploader to sync all golden runs")
