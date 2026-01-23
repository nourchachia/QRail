"""
Graph Neural Network Adapter - src/models/gnn_adapter.py

Purpose:
    Bridges the dictionary output from DataFuelPipeline.extract_gnn_features() 
    to torch_geometric.data.Data format required by the GNN encoder.

Architecture:
    DataFuelPipeline.extract_gnn_features() → Dictionary format
    ↓
    GraphConverter.convert() → torch_geometric.data.Data
    ↓
    HeterogeneousGATEncoder.forward() → Graph embeddings

Input Format (from extract_gnn_features):
    {
        'nodes': [
            {'id': 'STN_001', 'features': [1.0, 10, 0.85, ...]},
            ...
        ],
        'edges': [
            {'from': 'STN_001', 'to': 'STN_002', 'features': [0.8, 0.5, ...]},
            ...
        ],
        'global_features': [0.85, 1.0, 0.33, ...]
    }

Output Format (PyG Data):
    Data(
        x=[num_nodes, node_feature_dim],
        node_type=[num_nodes],
        edge_index=[2, num_edges],
        edge_attr=[num_edges, edge_feature_dim],
        global_features=[global_feature_dim] (optional)
    )
"""

import torch
from torch_geometric.data import Data
from typing import Dict, List, Any, Optional
import numpy as np


class GraphConverter:
    """
    Converts dictionary-based graph representation to PyTorch Geometric Data format.
    
    This adapter bridges the gap between:
    - DataFuelPipeline.extract_gnn_features() (dictionary output)
    - HeterogeneousGATEncoder (PyG Data input)
    
    Key Features:
    - Handles node ID mapping (string IDs → integer indices)
    - Constructs edge_index from edge list
    - Validates feature dimensions
    - Supports optional global features
    - Handles missing node types gracefully
    """
    
    def __init__(self, 
                 node_feature_dim: Optional[int] = None,
                 edge_feature_dim: Optional[int] = None,
                 infer_node_types: bool = True):
        """
        Initialize the converter.
        
        Args:
            node_feature_dim: Expected node feature dimension. If None, inferred from data.
            edge_feature_dim: Expected edge feature dimension. If None, inferred from data.
            infer_node_types: If True, infer node types from station IDs (STN_XXX format).
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.infer_node_types = infer_node_types
    
    def convert(self, graph_dict: Dict[str, Any], 
                incident: Optional[Dict[str, Any]] = None) -> Data:
        """
        Convert dictionary graph representation to PyG Data object.
        
        Args:
            graph_dict: Dictionary from DataFuelPipeline.extract_gnn_features() with keys:
                - 'nodes': List of {'id': str, 'features': List[float]}
                - 'edges': List of {'from': str, 'to': str, 'features': List[float]}
                - 'global_features': Optional List[float]
            incident: Optional incident dictionary for additional context.
        
        Returns:
            PyG Data object ready for GNN encoder.
        """
        nodes = graph_dict.get('nodes', [])
        edges = graph_dict.get('edges', [])
        global_features = graph_dict.get('global_features', [])
        
        if len(nodes) == 0:
            raise ValueError("Graph must contain at least one node")
        
        # Build node ID to index mapping
        node_id_to_idx = {}
        node_features = []
        node_types = []
        
        for idx, node in enumerate(nodes):
            node_id = node['id']
            node_id_to_idx[node_id] = idx
            
            # Extract features
            features = node['features']
            if not isinstance(features, (list, np.ndarray, torch.Tensor)):
                raise TypeError(f"Node features must be list/array, got {type(features)}")
            
            # Convert to list if needed
            if isinstance(features, (np.ndarray, torch.Tensor)):
                features = features.tolist()
            
            # Validate feature dimension
            if self.node_feature_dim is not None:
                if len(features) != self.node_feature_dim:
                    raise ValueError(
                        f"Node feature dimension mismatch: expected {self.node_feature_dim}, "
                        f"got {len(features)}"
                    )
            
            node_features.append(features)
            
            # Infer node type from station ID if enabled
            if self.infer_node_types:
                node_type = self._infer_node_type(node_id)
            else:
                # Default to 'local' (type 2) if not inferring
                node_type = 2
            
            node_types.append(node_type)
        
        # Build edge_index and edge_attr
        edge_list = []
        edge_features = []
        
        for edge in edges:
            from_id = edge['from']
            to_id = edge['to']
            
            # Map string IDs to integer indices
            if from_id not in node_id_to_idx:
                # Skip edges connecting to nodes not in the graph
                continue
            if to_id not in node_id_to_idx:
                continue
            
            src_idx = node_id_to_idx[from_id]
            dst_idx = node_id_to_idx[to_id]
            
            edge_list.append([src_idx, dst_idx])
            
            # Extract edge features
            features = edge['features']
            if not isinstance(features, (list, np.ndarray, torch.Tensor)):
                raise TypeError(f"Edge features must be list/array, got {type(features)}")
            
            # Convert to list if needed
            if isinstance(features, (np.ndarray, torch.Tensor)):
                features = features.tolist()
            
            # Validate feature dimension
            if self.edge_feature_dim is not None:
                if len(features) != self.edge_feature_dim:
                    raise ValueError(
                        f"Edge feature dimension mismatch: expected {self.edge_feature_dim}, "
                        f"got {len(features)}"
                    )
            
            edge_features.append(features)
        
        if len(edge_list) == 0:
            # Graph with no edges - create empty edge_index
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(edge_features[0]) if edge_features else 8), dtype=torch.float32)
        else:
            # Convert to tensors
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        # Convert node features and types to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        node_type = torch.tensor(node_types, dtype=torch.long)
        
        # Build PyG Data object
        data = Data(
            x=x,
            node_type=node_type,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # Add global features if provided
        if global_features:
            if not isinstance(global_features, (list, np.ndarray, torch.Tensor)):
                raise TypeError(f"Global features must be list/array, got {type(global_features)}")
            
            if isinstance(global_features, (np.ndarray, torch.Tensor)):
                global_features = global_features.tolist()
            
            data.global_features = torch.tensor(global_features, dtype=torch.float32)
        
        # Store metadata for debugging
        data.num_nodes = len(nodes)
        data.num_edges = len(edge_list)
        
        return data
    
    def _infer_node_type(self, station_id: str) -> int:
        """
        Infer node type from station ID format.
        
        Station types mapping:
        - major_hub: 0 (STN_001 to STN_005 typically)
        - regional: 1 (STN_006 to STN_020 typically)
        - local: 2 (STN_021 to STN_040 typically)
        - minor_halt: 3 (STN_041 to STN_050 typically)
        
        Args:
            station_id: Station ID string (e.g., 'STN_001')
        
        Returns:
            Node type index (0-3)
        """
        # Extract number from station ID
        try:
            # Format: STN_XXX
            parts = station_id.split('_')
            if len(parts) >= 2:
                num = int(parts[1])
                
                # Heuristic mapping based on typical numbering
                if 1 <= num <= 5:
                    return 0  # major_hub
                elif 6 <= num <= 20:
                    return 1  # regional
                elif 21 <= num <= 40:
                    return 2  # local
                else:
                    return 3  # minor_halt
        except (ValueError, IndexError):
            pass
        
        # Default to 'local' if parsing fails
        return 2
    
    def convert_batch(self, graph_dicts: List[Dict[str, Any]], 
                      incidents: Optional[List[Dict[str, Any]]] = None) -> List[Data]:
        """
        Convert multiple graphs to PyG Data objects.
        
        Args:
            graph_dicts: List of graph dictionaries
            incidents: Optional list of incident dictionaries
        
        Returns:
            List of PyG Data objects
        """
        if incidents is None:
            incidents = [None] * len(graph_dicts)
        
        if len(graph_dicts) != len(incidents):
            raise ValueError("graph_dicts and incidents must have same length")
        
        data_list = []
        for graph_dict, incident in zip(graph_dicts, incidents):
            data = self.convert(graph_dict, incident)
            data_list.append(data)
        
        return data_list


# ============================================================================
# Convenience Functions
# ============================================================================

def convert_graph_dict_to_pyg(graph_dict: Dict[str, Any], 
                               incident: Optional[Dict[str, Any]] = None,
                               node_feature_dim: Optional[int] = None,
                               edge_feature_dim: Optional[int] = None) -> Data:
    """
    Convenience function to convert a single graph dictionary to PyG Data.
    
    Args:
        graph_dict: Dictionary from DataFuelPipeline.extract_gnn_features()
        incident: Optional incident dictionary
        node_feature_dim: Expected node feature dimension
        edge_feature_dim: Expected edge feature dimension
    
    Returns:
        PyG Data object
    """
    converter = GraphConverter(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim
    )
    return converter.convert(graph_dict, incident)


# ============================================================================
# Example Usage & Integration
# ============================================================================

def example_integration():
    """
    Example showing full integration:
    DataFuelPipeline → GraphConverter → GNN Encoder
    """
    try:
        from src.backend.feature_extractor import DataFuelPipeline
        from src.models.gnn_encoder import HeterogeneousGATEncoder
        
        # Step 1: Extract features using DataFuelPipeline
        pipeline = DataFuelPipeline(data_dir="data")
        
        # Mock incident (in production, load from incidents.json)
        incident = {
            'type': 'signal_failure',
            'location': {
                'station_ids': ['STN_001', 'STN_002'],
                'is_junction': True,
                'zone': 'core',
                'segment_id': 'SEG_001'
            },
            'severity_level': 4,
            'hour_of_day': 8,
            'is_peak': True,
            'network_load_pct': 85,
            'trains_affected_count': 6
        }
        
        # Extract graph features (dictionary format)
        graph_dict = pipeline.extract_gnn_features(incident)
        
        # Step 2: Convert to PyG Data using GraphConverter
        converter = GraphConverter(
            node_feature_dim=10,  # Match DataFuelPipeline output
            edge_feature_dim=5    # Match DataFuelPipeline output
        )
        data = converter.convert(graph_dict, incident)
        
        # Step 3: Encode using GNN
        model = HeterogeneousGATEncoder(
            node_feature_dim=10,  # Match converter output
            edge_feature_dim=5,
            hidden_dim=64,
            output_dim=64
        )
        
        model.eval()
        with torch.no_grad():
            embedding = model(data)
        
        print(f"✅ Full pipeline successful!")
        print(f"   Graph: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"   Embedding shape: {embedding.shape}")
        
        return embedding
        
    except ImportError as e:
        print(f"⚠️  Import error (expected in test): {e}")
        return None


if __name__ == "__main__":
    """
    Test script demonstrating GraphConverter usage.
    """
    print("=== Graph Converter Test ===\n")
    
    # Mock graph dictionary (simulating DataFuelPipeline.extract_gnn_features() output)
    # Note: DataFuelPipeline returns 10-dim node features and 5-dim edge features
    mock_graph_dict = {
        'nodes': [
            {'id': 'STN_001', 'features': [1.0, 10, 0.85, 1, 1, 0.7, 0.5, 3, 1, 0]},
            {'id': 'STN_002', 'features': [1.0, 8, 0.72, 1, 1, 0.56, 0.69, 3, 1, 0]},
            {'id': 'STN_003', 'features': [0.0, 6, 0.45, 0, 0, 0.33, 0.61, 2, 0, 0]},
        ],
        'edges': [
            {'from': 'STN_001', 'to': 'STN_002', 'features': [0.8, 1.0, 1, 1, 1.18]},
            {'from': 'STN_002', 'to': 'STN_003', 'features': [0.8, 1.0, 1, 1, 1.18]},
        ],
        'global_features': [0.85, 1.0, 0.33, 0.06]
    }
    
    # Convert to PyG Data
    converter = GraphConverter(
        node_feature_dim=10,  # Match DataFuelPipeline format
        edge_feature_dim=5     # Match DataFuelPipeline format
    )
    
    data = converter.convert(mock_graph_dict)
    
    print(f"✅ Conversion successful!")
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Node features shape: {data.x.shape}")
    print(f"   Edge features shape: {data.edge_attr.shape}")
    print(f"   Edge index shape: {data.edge_index.shape}")
    print(f"   Node types: {data.node_type.tolist()}")
    
    if hasattr(data, 'global_features'):
        print(f"   Global features shape: {data.global_features.shape}")
    
    print("\n--- Testing Full Integration ---")
    example_integration()
    
    print("\n✓ GraphConverter ready for production use!")
