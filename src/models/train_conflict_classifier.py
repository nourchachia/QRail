"""
Train Conflict Classifier (Model 4)

- Builds embeddings for incidents (GNN 64 + LSTM 64 + Semantic 384)
- Trains MLP classifier with BCEWithLogitsLoss on 8 multi-label conflicts
- Saves best checkpoint to checkpoints/conflict_classifier/best_model.pt

Usage:
    # Activate venv first
    # & ".venv/Scripts/Activate.ps1"

    # Quick run on subset
    python src/models/train_conflict_classifier.py --epochs 3 --max-samples 300

    # Full run
    python src/models/train_conflict_classifier.py --epochs 15
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Local imports
from conflict_classifier import ConflictClassifier
from conflict_pipeline import ConflictPredictionPipeline

CONFLICT_TYPES = [
    "Signal_Failure",
    "Track_Blockage",
    "Train_Collision_Risk",
    "Schedule_Conflict",
    "Power_Failure",
    "Weather_Impact",
    "Crew_Unavailability",
    "Emergency_Response_Needed",
]


def map_incident_to_labels(incident: Dict) -> np.ndarray:
    """Heuristic multi-label mapping from incident fields → 8-dim vector."""
    y = np.zeros(8, dtype=np.float32)
    itype = (incident.get("type") or "").lower()
    weather = (incident.get("weather_condition") or incident.get("weather") or "").lower()
    sev = int(incident.get("severity_level", incident.get("severity", 3)))
    is_peak = bool(incident.get("is_peak", False))
    load_pct = int(incident.get("network_load_pct", 0))
    desc = " ".join([
        str(incident.get("semantic_description", "")),
        str(incident.get("resolution_strategy", "")),
        str(incident.get("operator_logs", "")),
    ]).lower()

    # Direct type matches
    if "signal" in itype:
        y[0] = 1
    if "breakdown" in itype or "derail" in itype or "track" in itype:
        y[1] = 1
    if "collision" in itype:
        y[2] = 1
    if "schedule" in itype or "delay" in itype:
        y[3] = 1
    if "power" in itype:
        y[4] = 1
    if "crew" in itype or "staff" in itype:
        y[6] = 1
    if "emergency" in itype or "alarm" in itype or "medical" in desc:
        y[7] = 1

    # Weather-driven label
    if weather in {"rain", "storm", "snow"}:
        y[5] = 1

    # Heuristics for schedule conflict due to load/peak
    if load_pct >= 80 and is_peak:
        y[3] = 1

    # Severity heuristic for collision risk (rare, but a signal)
    if sev >= 5 and ("block" in itype or "collision" in itype or "derail" in itype):
        y[2] = 1

    # Ensure at least one label
    if y.sum() == 0:
        y[3] = 1  # schedule default
    return y


class ConflictEmbeddingDataset(Dataset):
    def __init__(self, split: str = "train", data_dir: str = "data"):
        assert split in {"train", "test"}
        self.split = split
        self.data_dir = data_dir
        self.incidents, self.meta = self._load_incidents()
        self.pipeline = ConflictPredictionPipeline(data_dir=data_dir, device="cpu")

    def _load_incidents(self) -> Tuple[List[Dict], Dict]:
        path = Path(self.data_dir) / "processed" / "incidents.json"
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        meta = blob.get("metadata", {})
        records = blob.get(self.split, [])
        return records, meta

    def __len__(self):
        return len(self.incidents)

    def __getitem__(self, idx: int):
        inc = self.incidents[idx]
        # Build embeddings (no classifier usage here)
        gnn_features, lstm_seq, semantic_text, _ = self.pipeline.extract_all_features(inc)
        gnn_emb = self.pipeline.generate_gnn_embedding(inc)
        lstm_emb = self.pipeline.generate_lstm_embedding(lstm_seq)
        sem_emb = self.pipeline.generate_semantic_embedding(semantic_text)
        x = np.concatenate([gnn_emb, lstm_emb, sem_emb]).astype(np.float32)
        y = map_incident_to_labels(inc)
        return torch.from_numpy(x), torch.from_numpy(y)


def train(args):
    device = torch.device("cpu")

    # Dataset and loaders
    full_train = ConflictEmbeddingDataset(split="train", data_dir=args.data_dir)
    n = len(full_train)
    if args.max_samples and args.max_samples < n:
        # Subsample deterministically for quick runs
        indices = list(range(args.max_samples))
        subset = torch.utils.data.Subset(full_train, indices)
        full_train = subset
        n = len(full_train)

    val_ratio = 0.1
    val_size = max(1, int(n * val_ratio))
    train_size = n - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = ConflictClassifier(input_dim=512, hidden_dim=256, output_dim=8, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    ckpt_dir = Path("checkpoints/conflict_classifier")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        avg_val = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | train {avg_train:.4f} | val {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved checkpoint: {best_path}")

    print("Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=0, help="Limit training samples for quick runs")
    p.add_argument("--data-dir", type=str, default="data")
    args = p.parse_args()
    train(args)
