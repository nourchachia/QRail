"""
Query Black Swan Incidents from Qdrant
---------------------------------------
Retrieves all incidents that were automatically flagged as black swans.

Usage:
    python src/models/query_black_swans.py
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.backend.database import StorageManager
from qdrant_client.models import Filter, FieldCondition, MatchValue

def query_black_swans():
    """Retrieve all black swan incidents from Qdrant."""
    
    print("[INFO] Connecting to Qdrant...")
    storage = StorageManager()
    
    if not storage.client:
        print("[ERROR] Qdrant client not available!")
        return
    
    try:
        # Query all black swan incidents
        results = storage.client.scroll(
            collection_name="operational_memory",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="detected_as_black_swan",
                        match=MatchValue(value=True)
                    )
                ]
            ),
            limit=100
        )
        
        black_swans = results[0]
        
        print(f"\n{'='*70}")
        print(f"BLACK SWAN INCIDENTS DETECTED: {len(black_swans)}")
        print(f"{'='*70}\n")
        
        if not black_swans:
            print("No black swan incidents found yet.")
            print("Run incident analysis to detect anomalies.")
            return
        
        # Display each black swan
        for idx, point in enumerate(black_swans, 1):
            payload = point.payload
            
            print(f"{idx}. Incident ID: {point.id}")
            print(f"   Text: {payload.get('incident_text', 'N/A')[:80]}...")
            print(f"   Anomaly Score: {payload.get('anomaly_score', 'N/A'):.4f}")
            print(f"   Detected: {payload.get('detection_timestamp', 'N/A')}")
            print(f"   Type: {payload.get('incident_type', 'unknown')}")
            print(f"   Severity: {payload.get('severity', 'unknown')}")
            print(f"   Requires Review: {payload.get('requires_manual_review', False)}")
            print()
        
        # Statistics
        print(f"{'='*70}")
        print(f"STATISTICS:")
        print(f"{'='*70}")
        
        scores = [p.payload.get('anomaly_score', 0) for p in black_swans]
        if scores:
            print(f"Average Anomaly Score: {sum(scores)/len(scores):.4f}")
            print(f"Min Score: {min(scores):.4f}")
            print(f"Max Score: {max(scores):.4f}")
        
        types = {}
        for p in black_swans:
            inc_type = p.payload.get('incident_type', 'unknown')
            types[inc_type] = types.get(inc_type, 0) + 1
        
        print(f"\nIncident Types:")
        for inc_type, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  - {inc_type}: {count}")
        
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    query_black_swans()
