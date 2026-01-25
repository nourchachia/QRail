import requests
import json
import time

BASE_URL = "http://localhost:8001"

# Helper to print table without external dependencies
def print_table(rows, headers):
    if not rows:
        print("No results to display.")
        return
        
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    header_fmt = " | ".join([f"{{:<{w}}}" for w in widths])
    separator = "-+-".join(["-" * w for w in widths])
    
    print("\n" + "="*len(separator))
    print(header_fmt.format(*headers))
    print(separator)
    for row in rows:
        print(header_fmt.format(*row))
    print("="*len(separator))

def test_endpoint(method, endpoint, payload=None, description=""):
    print(f"\n🧪 Testing {method} {endpoint}: {description}")
    start = time.time()
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}")
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", json=payload)
        
        duration = time.time() - start
        status = "✅ PASS" if response.status_code == 200 else "❌ FAIL"
        
        # Check specific content
        try:
            data = response.json()
        except:
            data = {}
            
        details = ""
        
        if endpoint == "/api/search":
            count = len(data.get("similar_incidents", []))
            details = f"Found {count} results"
            if count == 0: status = "⚠️ WARN (Empty)"
            # PRINT RESULTS
            print(f"   👇 RESPONSE PREVIEW:")
            print(json.dumps(data, indent=2))
            
        elif endpoint == "/api/analyze":
            emb = len(data.get("embeddings", {}).get("semantic", []))
            details = f"Embeddings gen: {emb}-dim"
            # PRINT RESULTS
            print(f"   👇 RESPONSE PREVIEW:")
            # Truncate long vectors for measuring readability
            display_data = data.copy()
            if "embeddings" in display_data:
                display_data["embeddings"] = {k: f"{len(v)}-dim vector" for k,v in display_data["embeddings"].items()}
            print(json.dumps(display_data, indent=2))
            
        elif endpoint == "/api/golden-runs":
            count = len(data.get("golden_runs", []))
            details = f"Returned {count} golden runs"
            if count == 0: status = "❌ FAIL (Use uploader)"
            
        print(f"   {status} | {duration:.2f}s | {details}")
        return [method, endpoint, status, f"{duration:.2f}s", details]
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return [method, endpoint, "❌ ERROR", "0s", str(e)]

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
results = []

print("="*60)
print("🚀 STARTING COMPREHENSIVE API TEST SUITE")
print("="*60)

# 1. Basic Health
results.append(test_endpoint("GET", "/", description="Health Check"))

# 2. Static Data
results.append(test_endpoint("GET", "/api/stations", description="Get Stations"))
results.append(test_endpoint("GET", "/api/segments", description="Get Segments"))

# 3. Live Data
results.append(test_endpoint("GET", "/api/network/status", description="Live Network Status"))

# 4. Golden Runs
results.append(test_endpoint("GET", "/api/golden-runs", description="Golden Runs Knowledge Base"))

# 5. Debug Endpoints
results.append(test_endpoint("GET", "/api/debug/models-status", description="Model Status"))
results.append(test_endpoint("GET", "/api/debug/qdrant-status", description="Qdrant Status"))

# 6. Search - Case A: General Query
results.append(test_endpoint("POST", "/api/search", {
    "query_text": "Signal failure during peak hours",
    "limit": 5
}, "Search: General Query"))

# 7. Search - Case B: Specific Location (Should trigger GNN?)
results.append(test_endpoint("POST", "/api/search", {
    "query_text": "Derailment at Central Station STN_001",
    "limit": 5
}, "Search: Location Specific"))

# 9. Search - Case C: Specific Golden Run (Should find a Golden Result)
# Searching for a known Golden Run description (from golden_runs_accidents.json)
results.append(test_endpoint("POST", "/api/search", {
    "query_text": "Derailment due to excessive speed at curved track section",
    "limit": 3
}, "Search: Target Golden Run"))

# 10. Search - Case D: Weather Impact
results.append(test_endpoint("POST", "/api/search", {
    "query_text": "Signal failure caused by heavy snow and freezing temperatures",
    "limit": 3
}, "Search: Weather Specific"))

# 11. Analyze - Case B: Complex Cascading Delay
results.append(test_endpoint("POST", "/api/analyze", {
    "text": "Train EXP_013 stalled at Segment 45 causing 20min delay to following regional trains under heavy rain" 
}, "Analyze: Complex Scenario"))

# Print Final Summary
print_table(results, headers=["Method", "Endpoint", "Status", "Time", "Details"])
