r"""
================================================================================
QRail Neural Rail Conductor - FastAPI Backend
================================================================================

🎯 WHAT IS THIS FILE?
----------------------
This is the REST API layer that connects your frontend to the AI backend.

📊 WHY ALL IN ONE FILE?
-----------------------
For a hackathon/demo project, having everything in main.py is INTENTIONAL:
✅ Easier to read and understand the complete flow
✅ Simpler deployment (one file to deploy)
✅ Faster development (no hunting through multiple files)

For production, you'd split into:
- routes.py (endpoint definitions)
- schemas.py (Pydantic models)
- middleware.py (CORS, auth, logging)

But for QRail demo, this unified approach is cleaner.

🔗 MODELS 4 & 5 INTEGRATION
----------------------------
Q: "Does this code include Model 4/5?"
A: YES! It calls them via integration.py

HOW IT WORKS NOW (before Models 4/5 ready):
- integration.py tries to load Models 4/5
- If not found, returns empty defaults
- API still works, just with partial results

WHEN MODELS 4/5 READY:
- Integration.py automatically uses them
- API code needs ZERO changes (maybe 2 lines for cleaner syntax)
- Everything just works

🚀 HOW TO RUN THIS
------------------
STEP 1: Open terminal
STEP 2: cd C:\Users\ASUS\Desktop\QRail
STEP 3: python src/api/main.py
STEP 4: Open http://localhost:8002/docs

⚠️ IMPORTANT: The server must be RUNNING before you visit the link!

================================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import random
from datetime import datetime
from pathlib import Path

# =====================================================================
# === STEP 1: Setup Project Root Path ===
# =====================================================================
# WHY: Need to import from src/ directory
# WHAT: Add project root to Python path
# HOW: Calculate root and append to sys.path
# BEFORE: Python can't find src.backend modules
# AFTER: Can import integration.py and other modules

project_root = str(Path(__file__).parent.parent.parent)
import sys
if project_root not in sys.path:
    sys.path.append(project_root)

# =====================================================================
# === STEP 2: Import AI Pipeline ===
# =====================================================================
# WHY: This is the "brain" that processes incidents
# WHAT: Import IncidentPipeline from integration.py
# HOW: IncidentPipeline already handles Models 1-5
# BEFORE: No AI processing capability
# AFTER: Can analyze incidents end-to-end
#
# 📌 NOTE ON MODELS 4/5:
# - IncidentPipeline tries to load all 5 models
# - If Model 4/5 not found: returns empty conflicts/recommendations
# - If Model 4/5 ready: returns AI predictions
# - This API doesn't care which case - it works either way!

from src.backend.integration import IncidentPipeline

# =====================================================================
# === STEP 3: Create FastAPI Application ===
# =====================================================================
# WHY: FastAPI provides REST API framework
# WHAT: Initialize app with metadata
# HOW: Auto-generates docs at /docs
# BEFORE: No API server
# AFTER: RESTful API ready to receive requests

app = FastAPI(
    title="QRail Neural Rail Conductor API",
    description="AI-powered railway incident management system",
    version="1.0.0"
)

# =====================================================================
# === STEP 4: Add CORS Middleware ===
# =====================================================================
# WHY: Frontend (React) runs on different port, need CORS
# WHAT: Allow requests from localhost:3000 and localhost:5173
# HOW: FastAPI CORSMiddleware handles this automatically
# BEFORE: Browser blocks API calls due to CORS policy
# AFTER: Frontend can call API without errors
#
# 📌 PORTS EXPLAINED:
# - 3000: Default Create React App port
# - 5173: Default Vite (modern React) port
# - 8000: This API server port

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# === STEP 4.5: Serve Static Data Files (For Frontend) ===
# =====================================================================
# WHY: Frontend needs to fetch timetable.json, segments.json dynamically
# WHAT: Mount data/network directory as /static
# HOW: FastAPI StaticFiles serves files over HTTP
# BEFORE: Frontend gets 404 when fetching /static/timetable.json
# AFTER: Frontend can fetch http://localhost:8002/static/timetable.json

from fastapi.staticfiles import StaticFiles

# Mount data directory for frontend to access
data_static_path = os.path.join(project_root, "data", "network")
if os.path.exists(data_static_path):
    app.mount("/static", StaticFiles(directory=data_static_path), name="static")
    print(f"📁 Static files mounted: {data_static_path}")
else:
    print(f"⚠️ Warning: Data directory not found at {data_static_path}")

# =====================================================================
# === STEP 5: Initialize AI Pipeline (HAPPENS ONCE ON STARTUP) ===
# =====================================================================
# WHY: Loading models is slow - do it once, not per request
# WHAT: Create IncidentPipeline instance
# HOW: Loads all models, connects to Qdrant
# BEFORE: Server starts but models not loaded
# AFTER: All AI components ready to use
#
# 📌 WHAT LOADS HERE:
# - Model 1 (GNN): ✅ Always loaded
# - Model 2 (LSTM): ✅ Always loaded
# - Model 3 (Semantic): ✅ Always loaded
# - Model 4 (Conflict): ⚠️ Loaded if file exists, else None
# - Model 5 (Outcome): ⚠️ Loaded if file exists, else None
# - Qdrant client: ✅ Connects to cloud

print("🚄 Initializing QRail AI Pipeline...")
print("   This may take 30 seconds on first run...")
# Use absolute path for data_dir to avoid "file not found" errors
abs_data_dir = os.path.join(project_root, "data")
pipeline = IncidentPipeline(data_dir=abs_data_dir)
print("✅ Pipeline ready!")


# =====================================================================
# === REQUEST/RESPONSE MODELS (Type Safety with Pydantic) ===
# =====================================================================
# WHY: Type validation and auto-generated docs
# WHAT: Define the shape of requests and responses
# HOW: Pydantic models enforce types
# BEFORE: Unvalidated JSON, easy to make mistakes
# AFTER: Auto-validated, auto-documented API

class IncidentAnalysisRequest(BaseModel):
    """
    📥 INPUT: What the frontend sends to /api/analyze
    
    WHY: Operator needs to describe incident in natural language
    WHAT: Just the raw text description
    HOW: Frontend captures from textarea, sends as JSON
    BEFORE: Operator types incident description in UI
    AFTER: This model validates the request
    """
    text: str  # Raw incident description
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Signal failure at Central Station during morning peak. 5 trains affected."
            }
        }


class IncidentAnalysisResponse(BaseModel):
    """
    📤 OUTPUT: What we send back to frontend from /api/analyze
    
    WHY: Frontend needs structured data to display
    WHAT: Complete analysis with all AI results
    HOW: JSON with nested objects
    BEFORE: AI pipeline processes the incident
    AFTER: Frontend displays this data to operator
    
    📌 FIELDS EXPLAINED:
    - raw_text: Echo of what user sent
    - parsed: Structured incident data (from Gemini)
    - embeddings: 512-dim vector (Models 1+2+3)
    - similar_incidents: Top 5 matches from Qdrant
    - conflicts: 8 probabilities from Model 4 (or empty if not ready)
    - recommendations: Ranked resolutions from Model 5 (or generic if not ready)
    """
    raw_text: str
    parsed: Dict
    embeddings: Dict
    similar_incidents: List[Dict]
    conflicts: Dict
    recommendations: List[Dict]


class SearchRequest(BaseModel):
    """
    📥 INPUT: For /api/search endpoint
    
    WHY: Sometimes want to search without full analysis
    WHAT: Just query text and result limit
    HOW: Frontend "Search Similar" button
    BEFORE: User clicks search
    AFTER: Gets similar historical incidents
    """
    query_text: str
    limit: Optional[int] = 5


# =====================================================================
# === ENDPOINT 1: Root / Health Check ===
# =====================================================================
@app.get("/")
def root():
    """
    🏥 HEALTH CHECK
    
    WHY: Need to verify API is running
    WHAT: Simple status endpoint
    HOW: Returns JSON with service info
    BEFORE: Don't know if server is alive
    AFTER: 200 OK means we're running
    
    INPUT: None
    OUTPUT: {"status": "running", "service": "...", "version": "..."}
    """
    return {
        "status": "running",
        "service": "QRail Neural Rail Conductor API",
        "version": "1.0.0",
        "docs": "Visit /docs for interactive API documentation"
    }


# =====================================================================
# === ENDPOINT 2: Main Incident Analysis (⭐ MOST IMPORTANT) ===
# =====================================================================
@app.post("/api/analyze", response_model=IncidentAnalysisResponse)
def analyze_incident(request: IncidentAnalysisRequest):
    """
    ⭐ MAIN ENDPOINT: Complete AI-powered incident analysis
    
    === WHY ===
    This is the core functionality - operator describes incident, gets AI recommendations
    
    === WHAT ===
    Runs the complete AI pipeline (6 steps):
    1. Parse text (Gemini)
    2. Extract features
    3. Generate embeddings (Models 1,2,3)
    4. Search similar incidents (Qdrant)
    5. Predict conflicts (Model 4)
    6. Rank resolutions (Model 5)
    
    === HOW ===
    Calls integration.py which orchestrates all models
    
    === BEFORE ===
    User submitted incident via frontend form
    
    === AFTER ===
    Frontend displays:
    - Parsed incident details
    - Similar past incidents
    - Predicted conflicts with probabilities
    - Recommended resolutions ranked by success probability
    
    === INPUT ===
    {
        "text": "Signal failure at Central Station during morning peak hours. 
                 Heavy rain. 5 trains affected with cascade delays."
    }
    
    === OUTPUT ===
    {
        "raw_text": "Signal failure at...",
        "parsed": {
            "primary_failure_code": "SIGNAL_FAILURE",
            "estimated_delay_minutes": 25,
            "weather_condition": "rain",
            "is_peak": true
        },
        "embeddings": {
            "semantic": [0.1, 0.2, ...],  # 384-dim
            "structural": [0.3, ...],      # 64-dim
            "temporal": [0.5, ...]         # 64-dim
        },
        "similar_incidents": [
            {"incident_id": "abc123", "score": 0.95, "is_golden": true},
            ...
        ],
        "conflicts": {
            "headway_violation": 0.85,
            "platform_oversubscription": 0.23,
            ...
        },
        "recommendations": [
            {"strategy": "HOLD_UPSTREAM", "confidence": 0.92},
            ...
        ]
    }
    
    === 📌 MODELS 4/5 INTEGRATION ===
    RIGHT NOW (before Models 4/5 ready):
    - conflicts: {} (empty dict)
    - recommendations: [] (empty list or generic templates)
    
    WHEN MODELS 4/5 READY:
    - conflicts: {8 probabilities from Model 4}
    - recommendations: [AI-ranked resolutions from Model 5]
    
    CODE CHANGES NEEDED: ZERO! (maybe 2 lines for cleaner syntax)
    Integration.py automatically detects when models become available.
    """
    try:
        # === Call the brain (integration.py) ===
        # WHY: This is where all the AI magic happens
        # WHAT: Runs all 6 pipeline steps
        # HOW: integration.py orchestrates Models 1-5 + Qdrant
        # BEFORE: Have raw text
        # AFTER: Have complete AI analysis
        
        result = pipeline.process(request.text)
        
        # === Return the complete analysis ===
        # WHY: Frontend needs all this data
        # WHAT: Structured JSON response
        # HOW: Pydantic validates the format
        # BEFORE: Data is in Python dict
        # AFTER: JSON sent to frontend
        
        return {
            "raw_text": result['raw_text'],
            "parsed": result['parsed'],
            "embeddings": result['embeddings'],
            "similar_incidents": result['similar_incidents'],
            
            # 📌 These fields work NOW even if Models 4/5 not ready
            # Integration.py returns empty/default values gracefully
            "conflicts": result.get('conflicts', {}),
            "recommendations": result.get('recommendations', [])
        }
        
    except Exception as e:
        # WHY: Need graceful error handling
        # WHAT: Return 500 error with details
        # HOW: FastAPI HTTPException
        # BEFORE: Request fails
        # AFTER: Frontend knows why it failed
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# === ENDPOINT 3: Search Similar Incidents ===
# =====================================================================
@app.post("/api/search")
def search_similar(request: SearchRequest):
    """
    🔍 SEARCH ONLY (lighter than full analysis)
    
    === WHY ===
    Sometimes just want to find similar cases without full AI analysis
    
    === WHAT ===
    Generate embeddings → Query Qdrant → Return matches
    
    === HOW ===
    Use pipeline but only take the search results
    
    === BEFORE ===
    User clicked "Find Similar" button
    
    === AFTER ===
    Frontend displays list of similar historical incidents
    
    === INPUT ===
    {
        "query_text": "Signal failure during peak hour",
        "limit": 5
    }
    
    === OUTPUT ===
    {
        "similar_incidents": [
            {
                "incident_id": "abc123",
                "score": 0.95,
                "is_golden": true,
                "summary": "Signal failure at Central, resolved in 15min"
            },
            ...
        ]
    }
    
    === MODELS USED ===
    - Models 1,2,3: Generate embeddings
    - Qdrant: Vector similarity search
    - NOT Models 4,5: Don't need conflicts/outcomes for search
    """
    try:
        # Quick search without full pipeline
        result = pipeline.process(request.query_text)
        
        return {
            "similar_incidents": result['similar_incidents'][:request.limit]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# === ENDPOINT 4: Get All Stations ===
# =====================================================================
@app.get("/api/stations")
def get_stations():
    """
    🚉 NETWORK DATA: All 50 stations
    
    === WHY ===
    Frontend needs station data to draw network map
    
    === WHAT ===
    Returns all stations from stations.json
    
    === HOW ===
    Reads from data/network/stations.json
    
    === BEFORE ===
    Frontend loads (e.g., map component mounts)
    
    === AFTER ===
    Frontend draws stations on D3/Leaflet map
    
    === INPUT ===
    None
    
    === OUTPUT ===
    {
        "stations": [
            {
                "id": "STN_001",
                "name": "Central Station",
                "type": "major_hub",
                "platforms": 8,
                "coordinates": [51.5, -0.1],
                ...
            },
            ...
        ],
        "count": 50
    }
    
    === MODELS USED ===
    None - this is static data
    """
    try:
        stations = pipeline.storage.get_stations()
        return {"stations": stations, "count": len(stations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# === ENDPOINT 5: Get All Segments ===
# =====================================================================
@app.get("/api/segments")
def get_segments():
    """
    🛤️ NETWORK DATA: All 70 track segments
    
    === WHY ===
    Frontend needs segment data to draw track lines
    
    === WHAT ===
    Returns all segments from segments.json
    
    === HOW ===
    Reads from data/network/segments.json
    
    === BEFORE ===
    Frontend map component needs track lines
    
    === AFTER ===
    Frontend draws segments connecting stations
    
    === INPUT ===
    None
    
    === OUTPUT ===
    {
        "segments": [
            {
                "id": "SEG_001",
                "from_station": "STN_001",
                "to_station": "STN_002",
                "length_km": 5.2,
                "speed_limit": 120,
                "bidirectional": true,
                ...
            },
            ...
        ],
        "count": 70
    }
    
    === MODELS USED ===
    None - static data
    """
    try:
        segments = pipeline.storage.get_segments()
        return {"segments": segments, "count": len(segments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# === ENDPOINT 6: Get Live Network Status ===
# =====================================================================
@app.get("/api/network/status")
def get_network_status():
    """
    📡 LIVE DATA: Real-time network state
    
    === WHY ===
    Dashboard needs current network status
    
    === WHAT ===
    Returns live_status.json (real-time telemetry)
    
    === HOW ===
    Reads from data/processed/live_status.json
    (Generated by live_status_generator.py)
    
    === BEFORE ===
    Frontend dashboard refreshes (every 30 seconds)
    
    === AFTER ===
    Frontend updates live indicators, train positions
    
    === INPUT ===
    None
    
    === OUTPUT ===
    {
        "timestamp": "2024-01-24T22:00:00",
        "network_load_pct": 85,
        "active_incidents": 3,
        "trains_on_network": 45,
        "stations_status": [...],
        ...
    }
    
    === MODELS USED ===
    None - just returns current telemetry
    """
    try:
        status = pipeline.storage.get_live_status()
        if not status:
            # Generate temporary mock status if file is missing
            status = {
                "timestamp": datetime.now().isoformat(),
                "network_load_pct": random.randint(10, 85),
                "active_incidents": 0,
                "weather": {"condition": "clear", "temperature_c": 22}
            }
        else:
            # Add dynamic fluctuations to existing data
            status['timestamp'] = datetime.now().isoformat()
            
            # Fluctuate network load by +/- 5%
            base_load = status.get('network_load_pct', 50)
            status['network_load_pct'] = max(5, min(95, base_load + random.randint(-5, 5)))
            
            # Randomly change weather 5% of the time
            if random.random() < 0.05:
                conditions = ["clear", "rainy", "heavy_rain", "snow", "foggy"]
                status['weather']['condition'] = random.choice(conditions)
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# === ENDPOINT 7: Get Golden Runs ===
# =====================================================================
@app.get("/api/golden-runs")
def get_golden_runs():
    """
    ⭐ BEST PRACTICES: 50 perfect resolution examples
    
    === WHY ===
    Model 5 learns from these, operators can reference them
    
    === WHAT ===
    Returns golden_runs.json (curated perfect resolutions)
    
    === HOW ===
    Reads from data/processed/golden_runs.json
    
    === BEFORE ===
    Frontend "Best Practices" tab opened
    
    === AFTER ===
    Frontend displays exemplary incident resolutions
    
    === INPUT ===
    None
    
    === OUTPUT ===
    {
        "golden_runs": [
            {
                "incident_id": "golden_001",
                "outcome_score": 0.98,
                "resolution_strategy": "...",
                "why_perfect": "...",
                ...
            },
            ...
        ],
        "count": 50
    }
    
    === MODELS USED ===
    Model 5 uses these internally for training/ranking
    This endpoint just returns the raw data
    """
    try:
        golden = pipeline.storage.get_golden_runs()
        return {"golden_runs": golden, "count": len(golden)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# === DEBUGGING ENDPOINTS (Optional - for development) ===
# =====================================================================

@app.get("/api/debug/qdrant-status")
def check_qdrant():
    """
    🔧 DEBUG: Check Qdrant connection
    
    === WHY ===
    Need to verify Qdrant is working and has data
    
    === WHAT ===
    Query Qdrant for collection stats
    
    === HOW ===
    Direct Qdrant client call
    
    === BEFORE ===
    Developer suspects Qdrant issue
    
    === AFTER ===
    Shows connection status and point count
    """
    try:
        if not pipeline.storage.client:
            return {"status": "disconnected", "message": "Qdrant client not initialized"}
        
        info = pipeline.storage.client.get_collection("operational_memory")
        
        # Fix: Qdrant's CollectionInfo uses different attributes depending on version
        return {
            "status": "connected",
            "collection": "operational_memory",
            "points_count": info.points_count,
            "vectors_count": getattr(info, 'vectors_count', getattr(info, 'vector_count', 'N/A')),
            "message": "Qdrant is working correctly"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/debug/models-status")
def check_models():
    """
    🔧 DEBUG: Check which models are loaded
    
    === WHY ===
    Need to know which AI models are available
    
    === WHAT ===
    Check if each model is loaded (not None)
    
    === HOW ===
    Check pipeline attributes
    
    === BEFORE ===
    Developer wonders why some features don't work
    
    === AFTER ===
    Shows exactly which models are loaded
    
    === 📌 INTERPRETING RESULTS ===
    - gnn_encoder: true → Model 1 ready
    - lstm_encoder: true → Model 2 ready
    - semantic_encoder: true → Model 3 ready
    - conflict_classifier: false → Model 4 NOT ready (your teammates building)
    - outcome_predictor: false → Model 5 NOT ready (your teammates building)
    - qdrant_connected: true → Can search incidents
    """
    return {
        "models_loaded": {
            "model_1_gnn": pipeline.gnn_encoder is not None,
            "model_2_lstm": pipeline.lstm_encoder is not None,
            "model_3_semantic": pipeline.semantic_encoder is not None,
            "model_4_conflict": pipeline.conflict_classifier is not None,
            "model_5_outcome": pipeline.outcome_predictor is not None,
        },
        "infrastructure": {
            "qdrant_connected": pipeline.storage.client is not None,
            "database_ready": pipeline.storage is not None
        },
        "message": "false = not ready yet, true = loaded and ready"
    }


# =====================================================================
# === ENDPOINT 8: Close Incident (LIVE LEARNING - Part B) ===
# =====================================================================

class IncidentClosureRequest(BaseModel):
    """
    📥 INPUT: Operator closes an incident with results
    
    WHY: Add real operational outcomes to knowledge base
    WHAT: Incident + resolution + actual outcome
    HOW: Frontend "Close Incident" button
    """
    incident_description: str
    resolution_strategy: str
    operator_rating: str  # "good", "neutral", "bad"
    actual_outcome_score: float  # 0.0 to 1.0
    actual_delay_minutes: Optional[int] = None
    notes: Optional[str] = ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "incident_description": "Signal failure at Central Station resolved by holding upstream trains",
                "resolution_strategy": "HOLD_UPSTREAM_TRAINS",
                "operator_rating": "good",
                "actual_outcome_score": 0.92,
                "actual_delay_minutes": 15,
                "notes": "Resolution worked better than expected"
            }
        }


@app.post("/api/incident/close")
def close_incident(request: IncidentClosureRequest):
    """
    📝 LIVE LEARNING: Add real incident outcome to knowledge base
    
    === WHY ===
    When operator closes an incident, we want to:
    1. Make it immediately searchable (upload to Qdrant)
    2. Use it for weekly Model 5 retraining
    
    === WHAT ===
    Two-phase update:
    - IMMEDIATE: Upload to Qdrant (searchable now!)
    - WEEKLY: Included in next Model 5 training batch
    
    === HOW ===
    1. Generate embeddings for the incident
    2. Upload to Qdrant → immediately searchable
    3. Append to incidents.json → used in next training run
    
    === BEFORE ===
    Operator clicks "Close Incident" in UI
    
    === AFTER ===
    - New incident appears in similarity search results instantly
    - Next week's Model 5 training includes this outcome
    
    === INPUT ===
    {
        "incident_description": "Signal failure at Central...",
        "resolution_strategy": "HOLD_UPSTREAM_TRAINS",
        "operator_rating": "good",
        "actual_outcome_score": 0.92,
        "actual_delay_minutes": 15
    }
    
    === OUTPUT ===
    {
        "status": "success",
        "message": "Incident added to knowledge base",
        "incident_id": "abc123",
        "searchable": true,
        "training_queue": true
    }
    
    === MODELS USED ===
    - Models 1,2,3: Generate embeddings immediately
    - Model 5: Will use this data in next weekly training
    """
    try:
        import json
        import uuid
        from datetime import datetime
        
        # === Step 1: Generate embeddings ===
        print(f"📝 Processing incident closure...")
        result = pipeline.process(request.incident_description)
        
        # === Step 2: Create incident record ===
        incident_id = str(uuid.uuid4())
        new_incident = {
            "incident_id": incident_id,
            "timestamp": datetime.now().isoformat(),
            "semantic_description": request.incident_description,
            "resolution_strategy": request.resolution_strategy,
            "operator_rating": request.operator_rating,
            "outcome_score": request.actual_outcome_score,
            "actual_delay_minutes": request.actual_delay_minutes,
            "operator_logs": request.notes,
            "is_golden_run": request.actual_outcome_score >= 0.9,  # Auto-mark excellent outcomes
            "created_at": datetime.now().isoformat(),
            "source": "live_operations",  # Mark as real (not synthetic)
            
            # Add embeddings for immediate Qdrant upload
            "embeddings": result['embeddings']
        }
        
        # === Step 3: Upload to Qdrant (IMMEDIATE - searchable now!) ===
        try:
            pipeline.storage.upload_incident_memory([new_incident])
            searchable = True
            print(f"   ✅ Uploaded to Qdrant - immediately searchable")
        except Exception as e:
            searchable = False
            print(f"   ⚠ Qdrant upload failed: {e}")
        
        # === Step 4: Append to incidents.json (for weekly training) ===
        try:
            incidents_path = Path(abs_data_dir) / "processed" / "incidents.json"
            
            # Load existing data
            if incidents_path.exists():
                with open(incidents_path, 'r') as f:
                    data = json.load(f)
                
                # Append to train set
                if "train" in data:
                    data["train"].append(new_incident)
                    data["metadata"]["train_count"] = len(data["train"])
                else:
                    # Fallback if different structure
                    if isinstance(data, list):
                        data.append(new_incident)
                    else:
                        data = [new_incident]
                
                # Save back
                with open(incidents_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                training_queue = True
                print(f"   ✅ Appended to incidents.json - ready for next training")
            else:
                # Create new file if doesn't exist
                with open(incidents_path, 'w') as f:
                    json.dump({
                        "metadata": {"train_count": 1, "generated_at": datetime.now().isoformat()},
                        "train": [new_incident]
                    }, f, indent=2)
                training_queue = True
                print(f"   ✅ Created incidents.json with first live incident")
                
        except Exception as e:
            training_queue = False
            print(f"   ⚠ Failed to append to incidents.json: {e}")
        
        return {
            "status": "success",
            "message": "Incident added to knowledge base",
            "incident_id": incident_id,
            "searchable": searchable,  # True = findable in search now
            "training_queue": training_queue,  # True = will be used in next Model 5 training
            "is_golden": new_incident["is_golden_run"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close incident: {str(e)}")


# =====================================================================
# === SERVER STARTUP ===
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 STARTING FRESH SERVER (VERSION: FINAL FIX)")
    print("✨ Using Port 8001 to bypass zombie processes")
    print("="*60 + "\n")
    
    print("\n" + "=" * 70)
    print("🚄 QRail Neural Rail Conductor API Server")
    print("=" * 70)
    print("\n📍 Main Endpoints:")
    print("   POST http://localhost:8001/api/analyze     ← Main analysis")
    print("   POST http://localhost:8001/api/search      ← Search similar")
    print("   GET  http://localhost:8001/api/stations    ← Network data")
    print("   GET  http://localhost:8001/api/segments    ← Track data")
    print("\n📚 Interactive Docs:")
    print("   http://localhost:8001/docs                 ← FastAPI Swagger UI")
    print("\n🔧 Debug:")
    print("   GET  http://localhost:8001/api/debug/models-status")
    print("   GET  http://localhost:8001/api/debug/qdrant-status")
    print("=" * 70)
    print("\n⏳ Starting server...")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8002,       # Standard development port
        reload=False     # Disabled for direct python execution
    )



r"""
================================================================================
📘 USAGE GUIDE
================================================================================

STEP BEFORE RUNNING THIS:
--------------------------
1. Make sure uploader.py has populated Qdrant (800 incidents)
2. Make sure integration.py works (test it standalone)
3. Install FastAPI: pip install fastapi uvicorn

HOW TO RUN:
-----------
cd C:\Users\ASUS\Desktop\projects2025\QRail
python src/api/main.py

OR with uvicorn directly:
uvicorn src.api.main:app --reload --port 8000

STEP AFTER RUNNING THIS:
-------------------------
1. Open http://localhost:8000/docs to see interactive API docs
2. Test the /api/analyze endpoint with sample text
3. Build frontend that calls these endpoints

TESTING THE API:
----------------
# Using curl:
curl -X POST "http://localhost:8000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Signal failure at Central Station"}'

# Using Python:
import requests
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={"text": "Signal failure at Central Station"}
)
print(response.json())

FRONTEND INTEGRATION:
---------------------
// React/Next.js example:
const analyzeIncident = async (text) => {
  const response = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  const data = await response.json();
  return data;
};

CHANGES WHEN MODELS 4 & 5 READY:
---------------------------------
MINIMAL! Just these lines in analyze_incident():

BEFORE (now):
    "conflicts": result.get('conflicts', {}),
    "recommendations": result.get('recommendations', [])

AFTER (when ready):
    "conflicts": result['conflicts'],  # Direct access
    "recommendations": result['recommendations']

That's it! The endpoint structure stays exactly the same.
================================================================================
"""
