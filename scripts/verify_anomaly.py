"""
Verify Anomaly Detector
-----------------------
Tests the integrated anomaly detector with:
1. A normal incident (should be normal)
2. A crazy/black swan incident (should be anomaly)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backend.integration import IncidentPipeline

def test_anomaly_detector():
    print("[INFO] Initializing Pipeline...")
    pipeline = IncidentPipeline()
    
    if not pipeline.anomaly_detector:
        print("[ERROR] Anomaly Detector NOT loaded! Check checkpoints/anomaly_detector/model.pkl")
        return

    # Case 1: Normal Incident (similar to training data)
    normal_text = """
    Signal failure at North Terminal. 
    Delays expected on main line. 
    Technicians dispatched to fix interlocking issue.
    """
    
    print("\n📋 Testing Normal Incident...")
    res_normal = pipeline.process(normal_text)
    anom_normal = res_normal.get('anomaly')
    
    print(f"   Result: {anom_normal}")
    if anom_normal and not anom_normal['is_anomaly']:
        print("   [PASS] CORRECT: Identified as Normal")
    else:
        print("   [FAIL] UNEXPECTED: Identified as Anomaly (or None)")

    # Case 2: Black Swan Incident (Space Aliens / Godzilla / Something wild)
    # The semantic embedding should be very different
    abnormal_text = """
    Reports of unidentified aerial phenomenon hovering over West Hub.
    Electromagnetic interference causing all trains to levitate.
    Tracks melting from plasma discharge.
    """
    
    print("\n📋 Testing Black Swan Incident...")
    res_abnormal = pipeline.process(abnormal_text)
    anom_abnormal = res_abnormal.get('anomaly')
    
    print(f"   Result: {anom_abnormal}")
    if anom_abnormal and anom_abnormal['is_anomaly']:
        print("   [PASS] CORRECT: Identified as Anomaly")
    else:
        print("   [FAIL] UNEXPECTED: Identified as Normal (or None)")

if __name__ == "__main__":
    test_anomaly_detector()
