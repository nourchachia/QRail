# Anomaly Detection - Quick Start Guide

## 🚀 Testing the Anomaly Detection System

### Method 1: Automated Test Script (Easiest)

Run the test suite with pre-defined normal and black swan incidents:

```bash
python scripts/test_anomaly.py
```

**What it tests:**
- ✅ 3 Normal incidents (should NOT trigger alerts)
- ✅ 3 Black Swan incidents (should trigger alerts)
- ✅ Logs black swans to Qdrant
- ✅ Shows pass/fail for each test

**Expected Output:**
```
Test 1/6: Normal: Signal Failure
[✓] Result: NORMAL (score: 0.1234)
✓ PASS

Test 4/6: Black Swan: Alien Invasion
[!] Result: ANOMALY (score: -0.3456)
✓ PASS

Passed: 6/6 (100.0%)
```

---

### Method 2: API Testing (For Frontend Integration)

#### **Step 1: Start Backend**
```bash
python src/api/main.py
```

#### **Step 2: Send Test Request**

**PowerShell:**
```powershell
$body = @{text="Dragon attack on railway bridge"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8002/api/v1/incidents/analyze" -Method POST -Body $body -ContentType "application/json"
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8002/api/v1/incidents/analyze",
    json={"text": "Space aliens demanding intergalactic routes"}
)

print(response.json()['anomaly'])
# {'is_anomaly': True, 'score': -0.34, 'description': 'Black Swan: ...'}
```

---

### Method 3: Using the Frontend

#### **Step 1: Start Backend**
```bash
python src/api/main.py
```

#### **Step 2: Open Frontend**
Open `src/frontend/index.html` in a browser

#### **Step 3: Enter Incident**
Type in the incident input field:
- **Normal**: "Signal failure at Central Station"
- **Black Swan**: "Zombie outbreak on platform 5"

#### **Step 4: Click "Analyze"**

**If Black Swan:**
- 🚨 Red warning banner appears
- Text: "BLACK SWAN EVENT DETECTED"
- Shows anomaly score
- Operator caution message

**If Normal:**
- Shows similar incidents
- Shows recommendations
- No warning banner

---

## 📊 Viewing Logged Black Swans

After running tests, check what was logged:

```bash
python src/models/query_black_swans.py
```

**Output:**
```
BLACK SWAN INCIDENTS DETECTED: 3

1. Incident ID: a3b4c5d6
   Text: Unidentified flying objects hovering over railway tracks...
   Anomaly Score: -0.3456
   Detected: 2026-01-27T17:30:00
   Type: unknown
   Severity: unknown
   Requires Review: True

STATISTICS:
Average Anomaly Score: -0.3221
Incident Types:
  - unknown: 3
```

---

## 🔄 Retraining with Feedback

After collecting black swan incidents:

```bash
python src/models/retrain_from_feedback.py
```

This will:
1. Load golden runs (normal patterns)
2. Load black swans from Qdrant
3. Retrain model with both
4. Save improved model

**Output:**
```
[INFO] Found 50 golden runs
[INFO] Found 3 black swan incidents
[INFO] Training with 53 total samples
[SUCCESS] Model retrained and saved
```

---

## 🎯 Example Test Incidents

### Normal (Should NOT flag):
- "Signal failure at Central Station affecting Train EXP_001"
- "Track maintenance between STN_005 and STN_006"
- "Power outage at West Hub, backup systems activated"
- "Late train due to heavy passenger load"
- "Weather delay: Heavy rain affecting visibility"

### Black Swan (SHOULD flag):
- "Alien spacecraft landed on tracks"
- "Time traveler from 2050 warns of railway disaster"
- "Dragon nesting in main tunnel"
- "Zombie outbreak at station"
- "Wormhole opened in platform 3"
- "Robot uprising in signal control room"

---

## 🐛 Troubleshooting

### "Anomaly detector not loaded"
```bash
python src/models/train_anomaly_simple.py
```

### "Qdrant connection failed"
Check `.env` file has:
```
QDRANT_URL=your-url
QDRANT_API_KEY=your-key
```

### All incidents flagged as anomalies
Model needs more training data. Add more golden runs to:
`data/processed/golden_runs_accidents_enhanced.json`

### None flagged as anomalies
- Model contamination too high
- Retrain with `contamination=0.05` in `train_anomaly_simple.py`

---

## 📝 Next Steps

1. ✅ Run `scripts/test_anomaly.py`
2. ✅ Check `src/models/query_black_swans.py`
3. ✅ Test via frontend
4. ✅ Retrain periodically with `retrain_from_feedback.py`
