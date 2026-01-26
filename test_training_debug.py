import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import sys

# Set to DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.gnn_encoder import HeterogeneousGATEncoder
from src.models.gnn_dataset import get_dataloaders
from src.models.trainer import ModelTrainer

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load data
print("\nLoading data...")
train_loader, val_loader = get_dataloaders(batch_size=4)
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Create model
print("\nInitializing model...")
model = HeterogeneousGATEncoder(
    node_feature_dim=14,
    edge_feature_dim=8,
    hidden_dim=64,
    output_dim=64,
    num_node_types=4
)

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Initialize trainer
trainer = ModelTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    log_dir="runs/debug_test",
    checkpoint_dir="checkpoints/debug_test",
    gradient_clip=1.0
)

# Train for just 2 epochs
print("\nStarting DEBUG training (2 epochs only)...")
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=2,  # Just 2 epochs
    is_gnn=True,
    early_stopping_patience=10,
    save_best_only=False  # Save all checkpoints for debugging
)

print("\n" + "=" * 70)
print("Debug Training Summary:")
print(f"Train losses: {history['train_loss']}")
print(f"Val losses: {history['val_loss']}")
print("=" * 70)

trainer.close()
