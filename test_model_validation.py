"""
Test if the model can actually process validation batches
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root
project_root = Path(__file__).parent  
sys.path.insert(0, str(project_root))

from src.models.gnn_encoder import HeterogeneousGATEncoder
from src.models.gnn_dataset import get_dataloaders

print("=" * 70)
print("MODEL VALIDATION TEST")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load data
print("\nLoading data...")
train_loader, val_loader = get_dataloaders(batch_size=4)

# Create model  
print("\nCreating model...")
model = HeterogeneousGATEncoder(
    node_feature_dim=14,
    edge_feature_dim=8,
    hidden_dim=64,
    output_dim=64,
    num_node_types=4
).to(device)

criterion = nn.CrossEntropyLoss()

# Test validation batch processing
print("\n" + "=" * 70)
print("Testing validation batch processing...")
print("=" * 70)

model.eval()
total_loss = 0.0
batch_count_success = 0
batch_count_error = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        print(f"\nBatch {batch_idx + 1}/{len(val_loader)}:")
        try:
            # Move to device
            batch = batch.to(device)
            print(f"  - Graphs: {batch.num_graphs}")
            print(f"  - Nodes: {batch.num_nodes}")
            print(f"  - Has labels: {hasattr(batch, 'y')}")
            
            # Forward pass
            print(f"  - Running forward pass...")
            outputs = model(batch)
            print(f"  - Output shape: {outputs.shape}")
            
            # Check labels
            if hasattr(batch, 'y'):
                targets = batch.y.to(device)
                print(f"  - Target shape: {targets.shape}")
                print(f"  - Target values: {targets}")
                
                # Calculate loss
                print(f"  - Calculating loss...")
                loss = criterion(outputs, targets)
                print(f"  - Loss: {loss.item():.4f}")
                
                total_loss += loss.item()
                batch_count_success += 1
                print(f"  [OK] Batch processed successfully")
            else:
                print(f"  [ERROR] No labels in batch!")
                batch_count_error += 1
                
        except Exception as e:
            print(f"  [ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            batch_count_error += 1
        
        # Only test first 3 batches for speed
        if batch_idx >= 2:
            print(f"\n(Testing first 3 batches only...)")
            break

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Successful batches: {batch_count_success}")
print(f"Failed batches: {batch_count_error}")

if batch_count_success > 0:
    avg_loss = total_loss / batch_count_success
    print(f"Average validation loss: {avg_loss:.4f}")
    print("\n[OK] Model CAN process validation data correctly!")
    print("The issue must be in the trainer validation loop.")
else:
    print(f"\n[ERROR] Model CANNOT process validation data!")
    print("This is why validation loss is 0.")
