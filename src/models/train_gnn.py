# Create: src/models/train_gnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.gnn_encoder import HeterogeneousGATEncoder
from src.models.gnn_dataset import get_dataloaders
from src.models.trainer import ModelTrainer


def train_gnn():
    """Train GNN encoder on incident data"""
    
    print("=" * 70)
    print("TRAINING GNN ENCODER - Neural Rail Conductor")
    print("=" * 70)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print("\n📂 Loading data...")
    train_loader, val_loader = get_dataloaders(batch_size=16)
    
    # Create model
    print("\n🧠 Initializing model...")
    model = HeterogeneousGATEncoder(
        node_feature_dim=14,  # After padding
        edge_feature_dim=8,   # After padding
        hidden_dim=64,
        output_dim=64,
        num_node_types=4  # major_hub, regional, local, minor_halt
    )
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Loss function (for classification)
    criterion = nn.CrossEntropyLoss()
    
    # For regression (outcome score):
    # criterion = nn.MSELoss()
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        log_dir="runs/gnn_training",
        checkpoint_dir="checkpoints/gnn",
        gradient_clip=1.0
    )
    
    # Train
    print("\n🚀 Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        is_gnn=True,  # CRITICAL: Tell trainer this is a GNN
        early_stopping_patience=10,
        save_best_only=True
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Training Summary:")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"   Best Val Loss: {trainer.best_val_loss:.4f}")
    print("=" * 70)
    
    trainer.close()
    
    print("\n✅ GNN training complete!")
    print("   View logs: tensorboard --logdir=runs/gnn_training")
    print("   Model saved: checkpoints/gnn/best_model.pt")


if __name__ == "__main__":
    train_gnn()