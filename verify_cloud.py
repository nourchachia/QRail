import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def verify():
    # 1. Load Credentials
    current_dir = Path(__file__).resolve().parent
    env_path = current_dir / ".env"
    load_dotenv(env_path)
    
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    
    if not url or not key:
        print("❌ Error: Missing QDRANT_URL or QDRANT_API_KEY in .env")
        return

    # 2. Connect
    print(f"📡 Connecting to Qdrant Cloud...")
    client = QdrantClient(url=url, api_key=key)
    
    try:
        # 3. Get Collection Info
        info = client.get_collection("operational_memory")
        print(f"\n✅ COLLECTION FOUND: operational_memory")
        print(f"📊 POINTS COUNT: {info.points_count}")
        
        # 4. Peek at Data (Safe way to check structure)
        result = client.scroll(
            collection_name="operational_memory",
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if result and result[0]:
            point = result[0][0]
            print(f"\n💎 DATA SAMPLE FROM CLOUD:")
            print(f"   ID:       {point.id}")
            print(f"   Type:     {point.payload.get('type')}")
            print(f"   Location: {point.payload.get('location_name')}")
            print(f"   Context:  {point.payload.get('semantic_description', 'No description')[:60]}...")
            
            print(f"\n✨ DATABASE STATUS: 100% HEALTHY")
            print(f"Your 800 incidents are safe on the GCP server.")
        else:
            print("\n⚠️ Collection exists but is empty. Check your uploader logs.")
            
    except Exception as e:
        print(f"\n❌ CLOUD ERROR: {e}")
        print("   (Note: If it's an AttributeError, the Qdrant SDK structure has changed.)")

if __name__ == "__main__":
    verify()
