from src.backend.search_engine import NeuralSearcher
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize searcher
searcher = NeuralSearcher()

print('Checking for INC_003 in Qdrant...')

# Check if client is connected
if not searcher.client:
    print('❌ Qdrant client not connected!')
    print('Check QDRANT_URL and QDRANT_API_KEY in .env')
    exit(1)

# Search for INC_003
from qdrant_client.models import Filter, FieldCondition, MatchValue

result = searcher.client.scroll(
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
    print('✅ Found INC_003 in Qdrant!')
    point = result[0][0]
    print(f'   ID: {point.payload.get("incident_id")}')
    print(f'   Description: {point.payload.get("description", "N/A")[:80]}...')
    print(f'   Is Golden: {point.payload.get("is_golden_run")}')
else:
    print('❌ INC_003 NOT in Qdrant!')
    print('   Need to run: python src/backend/uploader.py')

