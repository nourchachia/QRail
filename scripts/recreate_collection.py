"""
Recreate Qdrant Collection with FastEmbed Naming
This script deletes the old collection and creates a new one compatible with FastEmbed.
"""
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Import Qdrant
from qdrant_client import QdrantClient

print("=" * 70)
print("🔄 Recreating Qdrant Collection with FastEmbed Naming")
print("=" * 70)

# Connect to Qdrant
url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_API_KEY")

if not url or not key:
    print("❌ QDRANT_URL or QDRANT_API_KEY not found in .env")
    sys.exit(1)

client = QdrantClient(url=url, api_key=key)
print(f"✅ Connected to Qdrant Cloud")

# Delete old collection
try:
    client.delete_collection(collection_name="operational_memory")
    print("✅ Deleted old 'operational_memory' collection")
except Exception as e:
    print(f"ℹ️ Collection doesn't exist or couldn't be deleted: {e}")

# Now init the new collection (database.py will create it with the new schema)
from src.backend.database import StorageManager

storage = StorageManager(qdrant_url=url, qdrant_api_key=key)
success = storage.init_operational_memory()

if success:
    print("\n✅ New collection created with FastEmbed-compatible naming!")
    print("\n📌 NEXT STEP:")
    print("   Run: python src/backend/uploader.py")
    print("   This will populate the new collection with incident data.")
else:
    print("\n❌ Failed to create new collection")
    sys.exit(1)

print("=" * 70)
