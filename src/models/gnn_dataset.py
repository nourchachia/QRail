# Create: src/models/gnn_dataset.py

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.feature_extractor import DataFuelPipeline
from src.models.gnn_adapter import GraphConverter


class IncidentGraphDataset(Dataset):
    """
    PyTorch Dataset for loading incidents and converting to graph format.
    Integrates DataFuelPipeline → GraphConverter → PyG Data
    """
    
    def __init__(self, 
                 incidents_file: str = "data/processed/incidents.json",
                 data_dir: str = "data",
                 split: str = "train",  # "train" or "val"
                 train_ratio: float = 0.8):
        """
        Args:
            incidents_file: Path to incidents.json
            data_dir: Path to data directory (for network files)
            split: "train" or "val"
            train_ratio: Proportion of data for training
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load incidents
        with open(incidents_file, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            all_incidents = data
        else:
            all_incidents = data.get('train', []) + data.get('test', [])
        
        # Split into train/val
        split_idx = int(len(all_incidents) * train_ratio)
        if split == "train":
            self.incidents = all_incidents[:split_idx]
        else:
            self.incidents = all_incidents[split_idx:]
        
        print(f"✅ Loaded {len(self.incidents)} {split} incidents")
        
        # Initialize pipeline and converter
        self.pipeline = DataFuelPipeline(data_dir=data_dir)
        self.converter = GraphConverter(
            node_feature_dim=10,  # DataFuelPipeline output
            edge_feature_dim=5     # DataFuelPipeline output
        )
    
    def __len__(self):
        return len(self.incidents)
    
    def __getitem__(self, idx):
        """
        Returns:
            PyG Data object with graph structure
        """
        incident = self.incidents[idx]
        
        # Extract graph features (dictionary format)
        graph_dict = self.pipeline.extract_gnn_features(incident)
        
        # Convert to PyG Data (with automatic padding to 14-dim nodes, 8-dim edges)
        data = self.converter.convert(graph_dict, incident, pad_to_gnn_dims=True)
        
        # Add target label (for supervised training)
        # Option 1: Severity classification (0-3)
        severity_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        severity = incident.get('severity', 'medium')
        if isinstance(severity, str):
            data.y = torch.tensor([severity_map.get(severity, 1)], dtype=torch.long)
        else:
            data.y = torch.tensor([severity], dtype=torch.long)
        
        # Option 2: Outcome score (for regression)
        # data.y = torch.tensor([incident.get('outcome_score', 0.5)], dtype=torch.float32)
        
        return data


# For PyTorch Geometric DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

def get_dataloaders(batch_size=16, num_workers=0):
    """
    Create train and validation dataloaders.
    
    Args:
        batch_size: Number of graphs per batch
        num_workers: Number of parallel workers (0 for single-threaded)
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = IncidentGraphDataset(split="train")
    val_dataset = IncidentGraphDataset(split="val")
    
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    """Test the dataset"""
    print("=== Testing IncidentGraphDataset ===\n")
    
    # Create dataset
    dataset = IncidentGraphDataset(split="train")
    
    # Test single sample
    data = dataset[0]
    print(f"✅ Sample graph loaded:")
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Node features: {data.x.shape}")
    print(f"   Edge features: {data.edge_attr.shape}")
    print(f"   Target: {data.y}")
    
    # Test dataloader
    train_loader, val_loader = get_dataloaders(batch_size=4)
    
    print(f"\n📊 DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\n🎲 Sample batch:")
    print(f"   Batch size: {batch.num_graphs}")
    print(f"   Total nodes: {batch.num_nodes}")
    print(f"   Total edges: {batch.num_edges}")
    print(f"   Targets shape: {batch.y.shape}")