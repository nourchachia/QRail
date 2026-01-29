import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.backend.database import StorageManager

def reinit_qdrant():
    print("🔄 Re-initializing Qdrant Collection...")
    
    # Load env vars
    load_dotenv()
    
    # Initialize storage
    storage = StorageManager()
    
    if not storage.client:
        print("❌ Qdrant client connection failed.")
        return
    
    collection_name = "operational_memory"
    
    # Check existence
    collections = storage.client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if exists:
        print(f"🗑️ Deleting existing collection '{collection_name}'...")
        storage.client.delete_collection(collection_name)
    
    # Recreate
    print(f"✨ Creating fresh collection '{collection_name}'...")
    success = storage.init_operational_memory()
    
    if success:
        print("✅ Qdrant Collection Successfully Re-initialized!")
    else:
        print("❌ Failed to re-initialize collection.")

if __name__ == "__main__":
    reinit_qdrant()
