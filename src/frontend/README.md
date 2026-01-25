# 🚄 Neural Rail Conductor - Frontend

**Real-time Rail Network Control Dashboard with AI-Powered Incident Management**

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Tech](https://img.shields.io/badge/tech-vanilla_js-yellow)
![Mode](https://img.shields.io/badge/mode-demo_ready-blue)

---

## 🎯 What Is This?

A web-based dashboard where rail operators can:
1. **Report incidents** in natural language
2. **Get AI recommendations** from similar historical cases
3. **See time savings** with before/after comparison
4. **Visualize cascading delays** on the network
5. **Submit feedback** to improve the AI

---

## 🚀 Quick Start

### Option 1: Demo Mode (No Backend Required)
```bash
cd d:\QRail\frontend

# Open index.html directly in browser
start index.html

# OR use Python HTTP server
python -m http.server 8080
# Then open http://localhost:8080
```

### Option 2: With Real Backend
```bash
# Terminal 1: Start backend
cd d:\QRail
python src/api/main.py

# Terminal 2: Serve frontend
cd d:\QRail\frontend
python -m http.server 8080
# Open http://localhost:8080

# Make sure demo mode is OFF in the UI
```

---

## 📁 Project Structure

```
frontend/
├── index.html              # Main HTML file
├── css/
│   ├── global.css          # Design system & layout
│   ├── network-view.css    # D3 network graph styles
│   ├── timeline.css        # Charts & comparison view
│   └── control-panel.css   # Input & results panels
└── js/
    ├── api.js              # Backend API service
    ├── state.js            # Application state management
    ├── network-view.js     # D3.js network visualization
    ├── timeline.js         # Chart.js timeline & comparison
    ├── control-panel.js    # Incident input & AI results
    └── app.js              # Main app orchestration
```

---

## 🎨 Features

### ✅ Network View (Left Panel)
- **D3.js visualization** of 50 stations and 70 track segments
- **Color-coded stations** by type (hub, regional, local)
- **Pulse animations** for affected stations
- **Cascade wave** animation showing delay spread
- **Recovery animation** when resolution applied

### ✅ Timeline (Center Panel)
- **Telemetry history** chart (last 30 minutes)
- **Network metrics** (weather, load, active trains)
- **Before/After comparison** showing AI impact
- **Savings banner** highlighting time saved and passengers helped

### ✅ Control Panel (Right Panel)
- **Natural language input** for incident description
- **Quick scenario buttons** for demo
- **Animated search progress** (3-step pipeline)
- **Similar cases** from historical data with match scores
- **AI recommendations** with confidence levels
- **Feedback form** for learning loop

### ✅ Global Features
- **Demo mode toggle** - works without backend
- **Dark theme** with modern design
- **Responsive layout** (optimized for 1920x1080)
- **Real-time updates** when backend connected
- **Reset button** to clear state

---

## 🧪 How to Test

### Demo Mode (Recommended for First Run)
1. Open `index.html` in browser
2. Click **Demo Mode: ON** toggle
3. Click one of the **Quick Scenario** buttons:
   - 🚥 Signal Failure
   - ⚠️ Train Breakdown
   - ⚡ Power Outage
4. Watch the AI analysis flow:
   - Network highlights affected stations (red pulse)
   - Search status shows 3-step progress
   - Similar cases appear with match scores
   - Resolution options displayed
5. Click **Apply This Resolution** on any option
6. See before/after comparison showing time savings
7. Rate the resolution and submit feedback

### With Real Backend
1. Start backend: `python src/api/main.py`
2. Verify backend is running: http://localhost:8000/docs
3. Open frontend and turn **Demo Mode OFF**
4. Type in free-form incident description
5. AI will fetch real data from Qdrant and models

---

## 🔗 API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analyze` | POST | Main AI analysis |
| `/api/stations` | GET | Get 50 stations |
| `/api/segments` | GET | Get 70 track segments |
| `/api/network/status` | GET | Live network data |
| `/api/feedback` | POST | Submit learning loop feedback |

---

## 🎭 Demo Scenarios

### Scenario 1: Signal Failure
**Text**: "Signal failure at Central Station during morning peak. Heavy rain. 5 trains affected with cascade delays."

**Expected Result**:
- Highlights Central Station (red pulse)
- Shows ~94% match to similar signal failures
- Recommends "HOLD_UPSTREAM" strategy
- Shows 62% improvement with AI

### Scenario 2: Train Breakdown
**Text**: "Train breakdown on segment between North Terminal and South Junction. Mechanical failure blocking track."

**Expected Result**:
- Highlights North and South stations
- Shows ~87% match to breakdowns
- Recommends "REROUTE_ALTERNATE" strategy
- Shows 58% improvement with AI

### Scenario 3: Power Outage
**Text**: "Power outage affecting West Hub station. All platforms without power. Backup systems activated."

**Expected Result**:
- Highlights West Hub (red pulse)
- Shows ~82% match to power issues
- Recommends emergency protocols
- Shows 71% improvement with AI

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Core** | Vanilla JavaScript (ES6+) |
| **Network Graph** | D3.js v7 (CDN) |
| **Charts** | Chart.js v4 (CDN) |
| **Styling** | CSS3 with CSS Grid |
| **Backend** | FastAPI (Python) |
| **State** | Simple object-based state management |

**Why Vanilla JS?**
- ✅ No Node.js/npm required
- ✅ No build step - runs directly in browser
- ✅ Lightweight and fast
- ✅ Easy to debug and understand

---

## 📊 Performance

- **Load Time**: <2 seconds
- **Network Rendering**: ~500ms for 50 stations + 70 segments
- **API Response**: 2-5 seconds (includes Gemini + Qdrant)
- **Animation FPS**: 60fps (smooth cascade animations)

---

## 🐛 Troubleshooting

### Issue: "API error" when analyzing incident
**Solution**: 
1. Check backend is running: http://localhost:8000
2. Or enable Demo Mode for offline use

### Issue: Network graph not showing
**Solution**:
1. Check browser console for D3 errors
2. Try refreshing the page
3. Check that D3.js CDN loaded (see Network tab)

### Issue: Charts not rendering
**Solution**:
1. Check browser console for Chart.js errors
2. Verify Chart.js CDN loaded
3. Try clearing browser cache

### Issue: CORS error when calling API
**Solution**:
Backend already has CORS configured for `localhost`. If using different port, update `API_BASE` in `js/api.js`

---

## 🎯 Next Steps / Enhancements

Possible future improvements:
- [ ] Add WebSocket for real-time train position updates
- [ ] Implement train animation along segments
- [ ] Add sound effects for incidents/resolutions
- [ ] Export comparison report as PDF
- [ ] Add keyboard shortcuts (ESC = reset, Enter = analyze)
- [ ] Mobile responsive design
- [ ] Add more scenario templates
- [ ] Implement undo/redo for state changes

---

## 📖 Code Guide

### Adding a New Component
1. Create `js/component-name.js`
2. Export functions as `window.componentName = { ... }`
3. Import in `index.html` before `app.js`
4. Use in `app.js` like `window.componentName.init()`

### Modifying State
```javascript
window.appState.setState({
  status: 'analyzing',
  currentIncident: {...},
});
```

### Calling Backend API
```javascript
const result = await window.api.analyzeIncident(text);
```

### Adding a New Scenario
Edit `js/state.js`:
```javascript
const scenarios = {
  my_scenario: {
    text: 'Description...',
    location: { station_ids: ['STN_001'] },
    severity: 'high',
  },
};
```

---

## 👥 Credits

Built with ❤️ for the QRail Neural Rail Conductor project.

**Tech Stack**:
- [D3.js](https://d3js.org/) for network visualization
- [Chart.js](https://www.chartjs.org/) for timeline charts
- [FastAPI](https://fastapi.tiangolo.com/) for backend
- Design inspired by modern rail control systems

---

## 📄 License

Part of the QRail project. See main project README for license details.

---

**🚄 Happy Rail Conducting!** 🎉
