# Semantic Encoder (Model 3) Implementation

## Overview
Successfully implemented the Semantic Encoder and Incident Parser for the QRail system.

## Files Created

### 1. `src/models/semantic_encoder.py`
- **SemanticEncoder class**: Converts text to 384-dim embeddings using SentenceTransformer
- **Features**:
  - Model caching (singleton pattern) - prevents reloading `all-MiniLM-L6-v2` on every call
  - Thread-safe implementation
  - Integration with `DataFuelPipeline.extract_semantic_text()`
  - Batch processing support
  - Graceful error handling

### 2. `src/backend/incident_parser.py`
- **IncidentParser class**: Parses incident descriptions using Gemini AI
- **Features**:
  - Precise extraction of `estimated_delay_minutes` and `primary_failure_code`
  - Auto-fetches weather/load from `live_status.json` via StorageManager
  - Gemini prompt engineering for structured output
  - Fallback rule-based parsing when Gemini unavailable

### 3. Updated Files
- `src/models/__init__.py`: Added SemanticEncoder export
- `requirements.txt`: Added dependencies (sentence-transformers, google-generativeai)

## Installation

```bash
pip install sentence-transformers>=2.2.0
pip install google-generativeai>=0.3.0  # Optional, for Gemini parsing
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Semantic Encoder

```python
from src.models.semantic_encoder import SemanticEncoder
from src.backend.feature_extractor import DataFuelPipeline

# Initialize (model loads and caches automatically)
encoder = SemanticEncoder()

# Single text encoding
text = "Signal failure at Central Station during peak hours"
embedding = encoder.encode(text)  # Returns (384,) numpy array

# Batch encoding
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = encoder.encode_batch(texts)  # Returns (3, 384) numpy array

# Integration with DataFuelPipeline
pipeline = DataFuelPipeline(data_dir="data")
incident = {...}  # Your incident dictionary
embedding = encoder.encode_from_pipeline(incident)
```

### Incident Parser

```python
from src.backend.incident_parser import IncidentParser

# Initialize parser
parser = IncidentParser(data_dir="data", api_key="your-gemini-key")

# Parse description
description = "Signal system failure at Central Station during peak hours"
result = parser.parse(description)
# Returns: {
#   "estimated_delay_minutes": 45,
#   "primary_failure_code": "SIGNAL_FAIL",
#   "confidence": 0.95,
#   "reasoning": "...",
#   "weather": {...},  # From live_status.json
#   "network_load_pct": 85  # From live_status.json
# }

# Parse and enrich incident
incident = {
    "incident_id": "INC_001",
    "semantic_description": description
}
enriched = parser.parse_incident(incident)
```

## Integration with DataFuelPipeline

The semantic encoder integrates seamlessly with the existing pipeline:

```python
from src.backend.feature_extractor import DataFuelPipeline
from src.models.semantic_encoder import SemanticEncoder

pipeline = DataFuelPipeline(data_dir="data")
encoder = SemanticEncoder()

# Extract semantic text (existing method)
text = pipeline.extract_semantic_text(incident)

# Encode to embedding (new)
embedding = encoder.encode(text)
```

## Model Details

- **Model**: `all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Normalization**: L2-normalized by default
- **Performance**: ~10,000 sentences/second on CPU
- **Memory**: ~80MB model size
- **Caching**: Model loaded once, reused for all subsequent calls

## Testing

Run test scripts:
```bash
python src/models/semantic_encoder.py
python src/backend/incident_parser.py
```

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Set Gemini API key** (optional): Set `GEMINI_API_KEY` environment variable
3. **Test the implementations**: Run the test scripts
4. **Integrate into pipeline**: Use in your incident processing workflow

## Notes

- Model caching significantly improves performance (no reload on every call)
- Singleton pattern ensures only one model instance in memory
- Thread-safe for concurrent processing
- Graceful fallback if dependencies not installed
- IncidentParser auto-enriches with weather/load from `live_status.json`
