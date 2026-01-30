"""
Test anomaly detection via API endpoint
"""
import requests
import json
import time

API_URL = "http://localhost:8002/api/analyze"

# Wait for server to be ready
print("Waiting for server to be ready...")
for i in range(30):
    try:
        response = requests.post(API_URL, json={"text": "test"}, timeout=5)
        if response.status_code in [200, 422]:  # 422 means validation error, but server is up
            print("✓ Server is ready!")
            break
    except:
        print(f"  Attempt {i+1}/30 - Retrying in 2 seconds...")
        time.sleep(2)
else:
    print("✗ Server not responding after 60 seconds")
    exit(1)

print("\n" + "=" * 70)
print("TEST 1: Normal Incident")
print("=" * 70)

normal_text = "Signal failure at Central Station during morning peak."
response = requests.post(API_URL, json={"text": normal_text})
data = response.json()

print(f"Response status: {response.status_code}")
print(f"Anomaly field exists: {'anomaly' in data}")
if 'anomaly' in data:
    print(f"Anomaly data: {json.dumps(data['anomaly'], indent=2)}")
else:
    print("ERROR: No anomaly field in response!")
    print(f"Available fields: {list(data.keys())}")

print("\n" + "=" * 70)
print("TEST 2: Anomalous Incident")
print("=" * 70)

anomaly_text = "Alien attack on the rail network with giant crystalline entities."
response2 = requests.post(API_URL, json={"text": anomaly_text})
data2 = response2.json()

print(f"Response status: {response2.status_code}")
print(f"Anomaly field exists: {'anomaly' in data2}")
if 'anomaly' in data2:
    print(f"Anomaly data: {json.dumps(data2['anomaly'], indent=2)}")
    if data2['anomaly'] and data2['anomaly'].get('is_anomaly'):
        print("\n✓ ANOMALY DETECTED!")
    else:
        print("\n✗ Not flagged as anomaly (might be normal pattern)")
else:
    print("ERROR: No anomaly field in response!")
    print(f"Available fields: {list(data2.keys())}")
