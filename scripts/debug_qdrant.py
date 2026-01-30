
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv()

url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=url, api_key=key)
client.set_model("sentence-transformers/all-MiniLM-L6-v2")

print("Connected to Qdrant")

try:
    collection_info = client.get_collection(collection_name="operational_memory")
    print(f"Collection: operational_memory")
    print(f"Status: {collection_info.status}")
    print(f"Points count: {collection_info.points_count}")
    print(f"Vectors count: {collection_info.vectors_count}")
            
except Exception as e:
    print(f"Error: {e}")
