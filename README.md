# 🚄 QRail: Neural Rail Conductor

**AI-Powered Railway Incident Management & Decision Support System**

QRail is an intelligent operational dashboard that helps railway dispatchers manage incidents in real-time. It combines **Network Topology Analysis (GNN)**, **Historical Pattern Recognition (LSTM)**, and **Semantic Search (Vector DB)** to recommend optimal resolutions for critical railway failures.

![System Architecture](C:/Users/USER/.gemini/antigravity/brain/28950788-b1fb-4ae1-b5e1-271a34e2cc17/qrail_architecture_diagram_1769463054033.png)

---

## ✨ Key Features

- **Real-Time Network Visualization**: Live view of 50 stations and 70 segments with animated train movements based on actual timetables.
- **Instant Incident Analysis**: Type a description (e.g., *"Signal failure at Central Station"*), and the AI pipeline identifies the failure type, location, and severity.
- **Intelligent Decision Support**:
  - **Graph Neural Network (GNN)** analyzes network impact.
  - **Vector Search (Qdrant)** retrieves similar past incidents (800+ historical cases).
  - **Outcome Predictor** ranks solution strategies by success probability.
- **Conflict Prediction**: Anticipates downstream operational conflicts (e.g., platform oversubscription, headway violations).
- **Interactive Simulation**: "Time Travel" mode to simulate network conditions at any time of day (Morning Peak, Off-Peak).

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Node.js (optional, for advanced frontend dev)
- Modern Web Browser (Chrome/Edge/Firefox)

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/QRail.git
cd QRail

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install core requirements (FastAPI, PyTorch, Sentence Transformers)
pip install -r requirements.txt

# Install visualization & vector DB tools
pip install qdrant-client plotly matplotlib numpy pandas
```

---

## 🚀 Usage

### 1. Start the Backend API
The backend handles AI processing, data retrieval, and simulation logic.
```bash
# Run from project root
python src/api/main.py
```
> **Note:** The server will start on **http://localhost:8002**.

### 2. Start the Frontend Dashboard
Open a new terminal window to serve the web interface.
```bash
# Run from project root
python -m http.server 8080
```

### 3. Open in Browser
Navigate to:
👉 **[http://localhost:8080/src/frontend/index.html](http://localhost:8080/src/frontend/index.html)**

---

## 🎮 How to Demo

1. **View Live Network**: Watch trains moving in real-time. Zoom/pan the map.
2. ** Simulate an Incident**:
   - Click the "Create Incident" button.
   - Enter text: *"Power outage at Station 5 affecting traction."*
   - Click **Analyze**.
3. **Review AI Recommendations**:
   - See specific **Conflict Predictions** (e.g., "High risk of platform overcrowding").
   - Review **Recommended Solutions** from historical "Golden Runs".
   - Check **Similar Cases** retrieved from the vector database.
4. **Time Travel**:
   - Use the slider in the "Simulation Time" panel to jump to 8:00 AM (Rush Hour) and see train density increase.

---

## 📂 Project Structure

```
d:\QRail
├── data/
│   ├── network/            # Static topology (stations.json, segments.json)
│   ├── processed/          # Generated datasets (incidents.json, golden_runs.json)
│   └── raw/                # Raw inputs
├── src/
│   ├── api/                # FastAPI backend (main.py)
│   ├── backend/            # Business logic & AI pipelines
│   ├── frontend/           # Web dashboard (HTML/CSS/JS)
│   │   ├── css/            # Stylesheets
│   │   └── js/             # Application logic (D3.js, State Management)
│   └── models/             # PyTorch AI models (GNN, LSTM, etc.)
├── data_gen/               # Scripts to generate synthetic data
└── requirements.txt        # Python dependencies
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Address already in use"** | Kill old python processes: `taskkill /IM python.exe /F` |
| **Trains not appearing** | Ensure Backend API is running on port **8002**. Refresh page. |
| **404 Error in Browser** | Make sure you run `python -m http.server` from `d:\QRail` root, not inside `src/`. |
| **Simulation Clock Stuck** | Click the "Play" button in the Time Control panel. |

---

## 🏗️ Architecture

QRail uses a **microservices-inspired architecture**:
- **Frontend**: Lightweight HTML5/JS (D3.js for graphs) communicating via REST.
- **Backend**: FastAPI aggregator that orchestrates 5 distinct AI models.
- **Data Layer**: 
  - **Qdrant**: Vector Similarity Search for 800 detailed operational logs.
  - **JSON**: Fast static storage for network topology and timetables.

---

**Developed by:** QRail Team (Google DeepMind Agentic Coding)  
**License:** MIT

---

## 🧠 Model Training

QRail's AI models are pre-trained, but you can retrain them on new data using the provided scripts.

### 1. LSTM Cascade Encoder (Model 2)
Trains the temporal pattern recognition model on synthetic telemetry sequences.
```bash
# Train for 50 epochs
python src/models/train_lstm.py
```
> **Output:** Saves checkpoints to `checkpoints/lstm/` and logs to `runs/lstm_cascade_encoder/`.

### 2. Conflict Classifier (Model 4)
Trains the multi-label classifier to predict 8 types of operational conflicts.
```bash
# Train for 15 epochs
python src/models/train_conflict_classifier.py --epochs 15
```
> **Output:** Saves best model to `checkpoints/conflict_classifier/best_model.pt`.

### 3. GNN Encoder (Model 1)
Trains the Graph Neural Network to understand network topology (stations & segments).
```bash
python src/models/train_gnn.py
```

### 4. Outcome Predictor (Model 5)
Trains the ranking model on "Golden Run" resolution data.
```bash
python src/models/train_outcome_model.py
```
