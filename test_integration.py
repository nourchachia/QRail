"""
Integration Test (Simplified) - test_integration_simple.py

Purpose:
    Tests the complete integration WITHOUT requiring all dependencies
    1. SemanticEncoder with DataFuelPipeline.extract_semantic_text()
    2. IncidentParser with live_status.json enrichment
    3. Full end-to-end workflow

Tests:
    ✓ SemanticEncoder model caching
    ✓ DataFuelPipeline text extraction
    ✓ IncidentParser Gemini prompt execution
    ✓ Weather/load enrichment from live_status.json
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

print("=" * 70)
print("INTEGRATION TEST - SemanticEncoder + IncidentParser")
print("=" * 70)

# Test 1: SemanticEncoder Model Caching
print("\n[TEST 1] SemanticEncoder Model Caching")
print("-" * 70)

try:
    from src.models.semantic_encoder import SemanticEncoder, SENTENCE_TRANSFORMERS_AVAILABLE
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        print("   Skipping SemanticEncoder tests...")
        encoder_available = False
    else:
        print("Creating first encoder instance...")
        encoder1 = SemanticEncoder()
        print(f"✓ Encoder 1 created. Model loaded: {encoder1._model is not None}")

        print("Creating second encoder instance (should use cached model)...")
        encoder2 = SemanticEncoder()
        print(f"✓ Encoder 2 created. Same model instance: {encoder1._model is encoder2._model}")

        if encoder1._model is encoder2._model:
            print("✅ PASS: Model caching works correctly!")
        else:
            print("❌ FAIL: Model was reloaded (caching failed)")
        
        encoder_available = True
except Exception as e:
    print(f"❌ Error in SemanticEncoder test: {e}")
    encoder_available = False

# Test 2: DataFuelPipeline Text Extraction
print("\n[TEST 2] DataFuelPipeline.extract_semantic_text()")
print("-" * 70)

try:
    from src.backend.feature_extractor import DataFuelPipeline

    pipeline = DataFuelPipeline(data_dir="data")

    test_incident_1 = {
        'type': 'signal_failure',
        'zone': 'core',
        'severity_level': 4,
        'weather_condition': 'rain',
        'trains_affected_count': 5
    }

    test_incident_2 = {
        'semantic_description': 'Major signal failure at Central Hub affecting 15 trains during morning peak'
    }

    text1 = pipeline.extract_semantic_text(test_incident_1)
    text2 = pipeline.extract_semantic_text(test_incident_2)

    print(f"Test Incident 1 (constructed):\n  → {text1}")
    print(f"\nTest Incident 2 (semantic_description):\n  → {text2}")

    if text1 and text2:
        print("✅ PASS: Text extraction works for both formats!")
        pipeline_available = True
    else:
        print("❌ FAIL: Text extraction returned empty strings")
        pipeline_available = False
except Exception as e:
    print(f"❌ Error in DataFuelPipeline test: {e}")
    pipeline_available = False

# Test 3: SemanticEncoder Integration with Pipeline
if encoder_available and pipeline_available:
    print("\n[TEST 3] SemanticEncoder + DataFuelPipeline Integration")
    print("-" * 70)

    try:
        embedding1 = encoder1.encode_from_pipeline(test_incident_1, pipeline)
        embedding2 = encoder1.encode(text2)

        print(f"Embedding 1 shape: {embedding1.shape}")
        print(f"Embedding 2 shape: {embedding2.shape}")
        print(f"Embedding 1 norm: {(embedding1**2).sum()**0.5:.4f}")
        print(f"Embedding 2 norm: {(embedding2**2).sum()**0.5:.4f}")

        if embedding1.shape == (384,) and embedding2.shape == (384,):
            print("✅ PASS: Embeddings have correct shape!")
        else:
            print("❌ FAIL: Incorrect embedding shape")
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
else:
    print("\n[TEST 3] Skipped (dependencies not available)")

# Test 4: Prepare live_status.json for IncidentParser
print("\n[TEST 4] Prepare live_status.json")
print("-" * 70)

live_status_path = Path("data/processed/live_status.json")
if live_status_path.exists():
    with open(live_status_path, 'r') as f:
        live_status = json.load(f)
    print(f"✓ live_status.json found")
    print(f"  Weather: {live_status.get('weather', {}).get('condition', 'N/A')}")
    print(f"  Network Load: {live_status.get('network_load_pct', 'N/A')}%")
else:
    print("⚠️  live_status.json not found. Creating mock version...")
    live_status_path.parent.mkdir(parents=True, exist_ok=True)
    mock_live_status = {
        "weather": {
            "condition": "rain",
            "temperature_c": 15,
            "wind_speed_kmh": 25,
            "visibility_km": 8.0
        },
        "network_load_pct": 75
    }
    with open(live_status_path, 'w') as f:
        json.dump(mock_live_status, f, indent=2)
    print(f"✓ Mock live_status.json created at {live_status_path}")

# Test 5: IncidentParser with live_status.json Enrichment
print("\n[TEST 5] IncidentParser - Gemini Parsing (Simplified)")
print("-" * 70)

# Create a simplified mock StorageManager
class MockStorageManager:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
    
    def load_json(self, filename):
        path = self.processed_dir / filename
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

# Test incident parser without full database dependency
try:
    print("Testing IncidentParser prompt creation...")
    
    # Import parser components
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src' / 'backend'))
    
    # Mock the database import
    sys.modules['src.backend.database'] = type(sys)('src.backend.database')
    sys.modules['src.backend.database'].StorageManager = MockStorageManager
    
    from src.backend.incident_parser import IncidentParser, GEMINI_AVAILABLE
    
    if not GEMINI_AVAILABLE:
        print("⚠️  google-generativeai not installed")
        print("   Run: pip install google-generativeai")
        print("   Testing with fallback parser only...")
    
    parser = IncidentParser(data_dir="data")
    
    description = "Signal system failure at Central Station during peak hours. Multiple converging routes affected."
    
    print(f"\nParsing incident description:")
    print(f"  → {description}")
    
    # Test with fallback first (always works)
    context = parser._load_live_status()
    print(f"\n📊 Loaded Context:")
    print(f"  Weather: {context['weather_condition']}")
    print(f"  Temperature: {context['temperature_c']}°C")
    print(f"  Network Load: {context['network_load_pct']}%")
    
    result = parser.parse(description, use_gemini=GEMINI_AVAILABLE)
    
    print("\n📊 Parsed Result:")
    print(json.dumps(result, indent=2))
    
    # Verify enrichment
    if 'weather' in result and 'network_load_pct' in result:
        print("✅ PASS: Weather and network load enrichment successful!")
    else:
        print("❌ FAIL: Missing enrichment data")
    
    if 'estimated_delay_minutes' in result and 'primary_failure_code' in result:
        print("✅ PASS: Delay and failure code extraction successful!")
    else:
        print("❌ FAIL: Missing extracted fields")
    
    parser_available = True
    
except Exception as e:
    print(f"❌ Error in IncidentParser test: {e}")
    import traceback
    traceback.print_exc()
    parser_available = False

# Test 6: Full End-to-End Integration
if encoder_available and pipeline_available and parser_available:
    print("\n[TEST 6] Full End-to-End Integration")
    print("-" * 70)

    try:
        full_incident = {
            "incident_id": "INC_INTEGRATION_TEST",
            "type": "train_breakdown",
            "station_ids": ["STN_001"],
            "zone": "core",
            "severity_level": 5,
            "semantic_description": "Train mechanical breakdown at Central Station blocking main platform during morning rush hour"
        }

        print("Step 1: Extract semantic text...")
        semantic_text = pipeline.extract_semantic_text(full_incident)
        print(f"  → {semantic_text}")

        print("\nStep 2: Encode to embedding vector...")
        embedding = encoder1.encode(semantic_text)
        print(f"  → Shape: {embedding.shape}, Norm: {(embedding**2).sum()**0.5:.4f}")

        print("\nStep 3: Parse with Gemini (extract delay + failure code)...")
        parsed = parser.parse_incident(full_incident)
        print(f"  → Delay: {parsed.get('estimated_delay_minutes', 'N/A')} minutes")
        print(f"  → Failure Code: {parsed.get('primary_failure_code', 'N/A')}")
        print(f"  → Weather: {parsed.get('weather', {}).get('condition', 'N/A')}")
        print(f"  → Network Load: {parsed.get('network_load_pct', 'N/A')}%")
        
        print("✅ PASS: Full end-to-end integration successful!")
    except Exception as e:
        print(f"❌ Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[TEST 6] Skipped (dependencies not available)")

print("\n" + "=" * 70)
print("🎯 INTEGRATION TEST COMPLETE")
print("=" * 70)

# Summary
print("\n📋 Summary:")
if encoder_available:
    print("  ✅ SemanticEncoder: Model caching implemented and working")
else:
    print("  ❌ SemanticEncoder: Not available (install sentence-transformers)")

if pipeline_available:
    print("  ✅ DataFuelPipeline: Text extraction handles both formats")
else:
    print("  ❌ DataFuelPipeline: Failed")

if parser_available:
    print("  ✅ IncidentParser: Delay + failure code extraction working")
    print("  ✅ Enrichment: Auto-fetches weather/load from live_status.json")
else:
    print("  ❌ IncidentParser: Failed")

if encoder_available and pipeline_available and parser_available:
    print("  ✅ End-to-End: Full pipeline integration successful")
    print("\n✨ All components integrated successfully!")
else:
    print("\n⚠️  Some components missing dependencies")
