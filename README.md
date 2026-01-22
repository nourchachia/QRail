# 🚄 Neural Rail Conductor (QRail)

AI-powered rail network incident management system using multi-vector similarity search and deep learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [AI Models](#ai-models)
- [Data Flow](#data-flow)
- [Team Workflow](#team-workflow)

## 🎯 Overview

Neural Rail Conductor is a decision support system for rail network operators. It uses AI to:
- **Detect** incidents in real-time
- **Search** historical similar cases using multi-vector similarity
- **Predict** conflicts and outcomes
- **Recommend** resolution strategies based on past successes

### Key Features

- **Triple-Vector Architecture**: Combines topology (GNN), temporal (LSTM), and semantic (Transformer) embeddings
- **Multi-Vector Search**: Qdrant vector database for finding similar historical incidents
- **5 AI Models**: GNN, LSTM, Semantic Encoder, Conflict Classifier, Outcome Predictor
- **Real-time Processing**: Handles live network status and incident analysis

## 🏗️ Architecture

```
Raw Data (JSON) → Feature Extraction → AI Models → Embeddings → Qdrant → Search → Recommendations
```

### Components

1. **Data Generation** (`data_gen/`): Creates synthetic network and incident data
2. **Storage Layer** (`src/backend/database.py`): Manages JSON files and Qdrant operations
3. **Feature Extraction** (`src/backend/feature_extractor.py`): Converts JSON → model-ready features
4. **AI Models** (`src/models/`): 5 models for encoding and prediction
5. **Backend API** (`src/backend/`): FastAPI endpoints for incident analysis

## 📁 Project Structure

```
QRail/
├── data/
│   ├── network/                    # Infrastructure data
│   │   ├── stations.json           # 50 stations
│   │   ├── segments.json           # 70 track segments
│   │   └── timetable.json          # Train schedules
│   └── processed/                  # Operational data
│       ├── incidents.json          # Historical incidents (1000+)
│       ├── live_status.json        # Real-time network state
│       └── golden_run_accidents.json # Perfect resolution examples (50)
│
├── src/
│   ├── backend/
│   │   ├── database.py             # StorageManager: JSON + Qdrant
│   │   └── feature_extractor.py    # DataFuelPipeline: Feature extraction
│   └── models/                     # AI Models
│       ├── gnn_encoder.py          # Model 1: GNN Topology Encoder
│       ├── lstm_encoder.py         # Model 2: LSTM Cascade Encoder
│       ├── semantic_encoder.py     # Model 3: Semantic Text Encoder
│       ├── conflict_classifier.py  # Model 4: Conflict Classifier (MLP)
│       └── outcome_predictor_xgb.py  # Model 5: Outcome Predictor (XGBoost)
│
└── data_gen/                       # Data generation scripts
    ├── generate_network.py
    ├── generate_incidents.py
    └── generate_golden_runs.py
```

## 🚀 Setup

### Prerequisites

- Python 3.8+
- Qdrant (Docker or Cloud)

### Installation

```bash
# Navigate to project root
cd QRail

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install torch-geometric
pip install sentence-transformers
pip install xgboost
pip install qdrant-client
pip install fastapi uvicorn
pip install pandas numpy
pip install python-dotenv
pip install pydantic
```

### Create Package Structure

```bash
# Create __init__.py files
touch src/__init__.py
touch src/backend/__init__.py
touch src/models/__init__.py
touch data_gen/__init__.py
```

### Start Qdrant

```bash
# Using Docker
use Qdrant Cloud (free tier available)
```

## 💻 Usage

### 1. Generate Data

```bash
# Generate network infrastructure
python data_gen/generate_network.py

# Generate incidents
python data_gen/generate_incidents.py

# Generate golden runs
python data_gen/generate_golden_runs.py
```

### 2. Test Storage System

```bash
cd QRail
python src/backend/database.py
```

**Expected Output:**
```
✓ Saved 1 stations
✓ Created Qdrant collection: operational_memory
✓ Storage system ready!
```

### 3. Test Feature Extraction

```bash
cd QRail
python src/backend/feature_extractor.py
```

**Expected Output:**
```
=== GNN Features ===
Nodes: 2                    # Affected stations
Edges: 11                   # Segments connecting them

=== LSTM Sequence ===
Sequence shape: [10, 4]     # 10 time steps, 4 features

=== Semantic Text ===
Description: Signal Failure at core zone...

=== Conflict Features ===
Feature vector: [0.85, 1.0, 0.33, ...]
```

### 4. Test AI Models

```bash
# Test GNN Encoder
python src/models/gnn_encoder.py

# Test LSTM Encoder
python src/models/lstm_encoder.py

# Test Semantic Encoder
python src/models/semantic_encoder.py

# Test Conflict Classifier
python src/models/conflict_classifier.py
```

## 🤖 AI Models

### Model 1: GNN Encoder (`gnn_encoder.py`)
- **Purpose**: Encodes network topology (station relationships)
- **Input**: Graph structure (nodes=stations, edges=segments)
- **Output**: 64-dim embedding
- **Architecture**: Graph Attention Network (GATv2)

### Model 2: LSTM Encoder (`lstm_encoder.py`)
- **Purpose**: Encodes temporal delay cascades
- **Input**: Time-series sequence [10, 4] (delay history)
- **Output**: 64-dim embedding
- **Architecture**: 2-layer LSTM

### Model 3: Semantic Encoder (`semantic_encoder.py`)
- **Purpose**: Encodes natural language descriptions
- **Input**: Text (operator logs, incident descriptions)
- **Output**: 384-dim embedding
- **Architecture**: Sentence-Transformer (all-MiniLM-L6-v2)

### Model 4: Conflict Classifier (`conflict_classifier.py`)
- **Purpose**: Predicts 8 types of operational conflicts
- **Input**: Combined embeddings (64 + 64 + 384 = 512-dim)
- **Output**: 8 probabilities (one per conflict type)
- **Architecture**: MLP with dropout

### Model 5: Outcome Predictor (`outcome_predictor_xgb.py`)
- **Purpose**: Predicts resolution success probability (0-1)
- **Input**: Context features + action features
- **Output**: Success probability
- **Architecture**: XGBoost Regressor

## 🔄 Data Flow

### Pipeline Overview

```
1. Data Generation
   └─→ JSON files saved to data/network/ and data/processed/

2. Feature Extraction
   └─→ DataFuelPipeline loads JSON → extracts features for each model

3. AI Model Processing
   └─→ Models generate embeddings:
       - GNN: 64-dim topology embedding
       - LSTM: 64-dim temporal embedding
       - Semantic: 384-dim text embedding

4. Qdrant Storage
   └─→ StorageManager uploads incidents with embeddings to Qdrant

5. Search & Analysis
   └─→ When new incident occurs:
       - Extract features → Generate embeddings
       - Search Qdrant for similar historical cases
       - Predict conflicts and outcomes
       - Recommend resolution strategies
```

### File Directory Rules

- **Infrastructure files** → `data/network/`:
  - `stations.json`
  - `segments.json`
  - `timetable.json`

- **Operational files** → `data/processed/`:
  - `incidents.json`
  - `live_status.json`
  - `golden_run_accidents.json`


## 📝 Key Modules

### `database.py` (StorageManager)
- **Purpose**: Storage and retrieval layer
- **Core Functions**:
  - `load_json()`: Loads files from correct directories
  - `upload_incident_memory()`: Uploads embeddings to Qdrant
  - `search_similar_incidents()`: Multi-vector search
- **Optional**: Save methods (if you want to use StorageManager in data generation)

### `feature_extractor.py` (DataFuelPipeline)
- **Purpose**: Converts JSON data → model-ready features
- **Key Methods**:
  - `extract_gnn_features()`: Graph structure for GNN
  - `extract_lstm_sequence()`: Time-series for LSTM
  - `extract_semantic_text()`: Text for Semantic Encoder
  - `extract_conflict_features()`: Context for Conflict Classifier
  - `extract_outcome_context()`: Features for Outcome Predictor

## 🧪 Testing

All modules include test scripts. Run from project root:

```bash
cd QRail

# Test storage
python src/backend/database.py

# Test feature extraction
python src/backend/feature_extractor.py

# Test models
python src/models/gnn_encoder.py
python src/models/lstm_encoder.py
python src/models/semantic_encoder.py
python src/models/conflict_classifier.py
```
