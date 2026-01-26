"""
Diagnostic script to check validation data
"""
import sys
from pathlib import Path
import os

# Force UTF-8 encoding for Windows console
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Temporarily patch print to avoid emoji issues
import builtins
original_print = builtins.print

def safe_print(*args, **kwargs):
    try:
        original_print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: remove emojis
        safe_args = [str(arg).encode('ascii', 'ignore').decode('ascii') for arg in args]
        original_print(*safe_args, **kwargs)

builtins.print = safe_print

from src.models.gnn_dataset import get_dataloaders, IncidentGraphDataset

print("=" * 70)
print("VALIDATION DATA DIAGNOSTIC")
print("=" * 70)

# Check datasets directly
print("\n1. Checking raw datasets...")
train_dataset = IncidentGraphDataset(split="train")
val_dataset = IncidentGraphDataset(split="val")

print(f"   Train dataset size: {len(train_dataset)}")
print(f"   Val dataset size: {len(val_dataset)}")

if len(val_dataset) == 0:
    print("   [X] CRITICAL: Validation dataset is EMPTY!")
    print("   This is why you get 0 validation loss.")
else:
    print(f"   [OK] Validation dataset has {len(val_dataset)} samples")

# Check dataloaders
print("\n2. Checking dataloaders with batch_size=4...")
train_loader, val_loader = get_dataloaders(batch_size=4)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

if len(val_loader) == 0:
    print("   [X] Validation dataloader has 0 batches")
else:
    print(f"   [OK] Validation dataloader has {len(val_loader)} batches")
    
    # Test loading a batch
    try:
        print("\n3. Testing validation batch...")
        batch = next(iter(val_loader))
        print(f"   [OK] Successfully loaded batch")
        print(f"   - Graphs in batch: {batch.num_graphs}")
        print(f"   - Has labels (y): {hasattr(batch, 'y')}")
        if hasattr(batch, 'y'):
            print(f"   - Labels shape: {batch.y.shape}")
            print(f"   - Labels: {batch.y}")
        else:
            print("   [X] Batch has NO labels (y attribute)")
    except Exception as e:
        print(f"   [X] Error loading batch: {e}")
        import traceback
        traceback.print_exc()

# Check incidents.json
print("\n4. Checking incidents.json file...")
import json
incidents_file = "data/processed/incidents.json"
try:
    with open(incidents_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        total_incidents = len(data)
        print(f"   Total incidents (list format): {total_incidents}")
    elif isinstance(data, dict):
        train_count = len(data.get('train', []))
        test_count = len(data.get('test', []))
        total_incidents = train_count + test_count
        print(f"   Total incidents (dict format): {total_incidents}")
        print(f"   - Train: {train_count}")
        print(f"   - Test: {test_count}")
    
    print(f"   With train_ratio=0.8:")
    print(f"   - Train split would have: {int(total_incidents * 0.8)} incidents")
    print(f"   - Val split would have: {total_incidents - int(total_incidents * 0.8)} incidents")
    
except FileNotFoundError:
    print(f"   [X] incidents.json not found at {incidents_file}")
except Exception as e:
    print(f"   [X] Error reading incidents.json: {e}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
