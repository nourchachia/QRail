"""
Test the backend directly to see if anomaly is returned
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json

# Test directly calling the pipeline
from src.backend.integration import IncidentPipeline

print("Initializing pipeline...")
pipeline = IncidentPipeline()

# Test with a normal incident
print("\n" + "=" * 70)
print("TEST 1: Normal Incident")
print("=" * 70)
normal_text = "Signal failure at Central Station during morning peak causing 20-minute delay."
result = pipeline.process(normal_text)

print(f"\nAnomaly field: {result.get('anomaly')}")
if result.get('anomaly'):
    print(f"  - is_anomaly: {result['anomaly'].get('is_anomaly')}")
    print(f"  - anomaly_score: {result['anomaly'].get('anomaly_score')}")
    print(f"  - severity: {result['anomaly'].get('severity')}")
    print(f"  - confidence: {result['anomaly'].get('confidence')}")
else:
    print("  ERROR: No anomaly field in result!")

# Test with an anomalous incident
print("\n" + "=" * 70)
print("TEST 2: Anomalous Incident")
print("=" * 70)
anomaly_text = "Giant crystalline entities have emerged from the tunnels causing blue light anomalies."
result2 = pipeline.process(anomaly_text)

print(f"\nAnomaly field: {result2.get('anomaly')}")
if result2.get('anomaly'):
    print(f"  - is_anomaly: {result2['anomaly'].get('is_anomaly')}")
    print(f"  - anomaly_score: {result2['anomaly'].get('anomaly_score')}")
    print(f"  - severity: {result2['anomaly'].get('severity')}")
    print(f"  - confidence: {result2['anomaly'].get('confidence')}")
else:
    print("  ERROR: No anomaly field in result!")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("✓ Anomaly detection working at backend level" if result.get('anomaly') and result2.get('anomaly') else "✗ Anomaly detection NOT working")
