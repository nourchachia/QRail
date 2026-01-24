"""
Conflict Prediction Pipeline - Integration of all embedding models
File: src/models/conflict_pipeline.py

Orchestrates the complete flow:
1. Extract features (GNN, LSTM, Semantic, Context)
2. Generate embeddings (GNN → 64-dim, LSTM → 64-dim, Semantic → 384-dim)
3. Concatenate embeddings (512-dim total)
4. Predict conflicts using MLP classifier

Architecture:
    Incident JSON → Feature Extraction → Embeddings → Concatenation → ConflictClassifier
    
    GNN Encoder (64-dim) ─┐
    LSTM Encoder (64-dim) ┼─→ Concatenate (512-dim) → MLP Conflict Classifier → [0-1]^8
    Semantic Encoder (384-dim) ┘
    Context Features (0-dim, embedded in training)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import json
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backend.feature_extractor import DataFuelPipeline
from src.models.conflict_classifier import ConflictClassifier
from src.models.gnn_encoder import HeterogeneousGATEncoder, DynamicGraphBuilder
from src.models.cascade.lstm_encoder import LSTMEncoder
from src.models.semantic_encoder import SemanticEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConflictPredictionPipeline:
    """
    Complete end-to-end pipeline for conflict prediction.
    
    Takes raw incident JSON and outputs conflict probabilities.
    """
    
    def __init__(
        self,
        gnn_checkpoint: Optional[str] = None,
        lstm_checkpoint: Optional[str] = None,
        classifier_checkpoint: Optional[str] = None,
        data_dir: str = "data",
        device: str = "cpu"
    ):
        """
        Initialize pipeline with all models.
        
        Args:
            gnn_checkpoint: Path to trained GNN weights (optional)
            lstm_checkpoint: Path to trained LSTM weights (optional)
            classifier_checkpoint: Path to trained Conflict Classifier weights (optional)
            data_dir: Directory containing data files
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.data_dir = data_dir
        
        # Initialize feature extractor
        self.pipeline = DataFuelPipeline(data_dir=data_dir)
        
        # Initialize embedding models
        logger.info("Loading embedding models...")
        self.gnn_encoder = HeterogeneousGATEncoder(
            node_feature_dim=14,
            edge_feature_dim=8,
            hidden_dim=64,
            output_dim=64,
            num_layers=3,
            num_heads=4
        ).to(self.device).eval()

        # Graph builder to convert incidents → PyG Data using network files
        network_dir = Path(self.data_dir) / "network"
        self.graph_builder = DynamicGraphBuilder(
            stations_path=str(network_dir / "stations.json"),
            segments_path=str(network_dir / "segments.json")
        )
        
        self.lstm_encoder = LSTMEncoder(
            input_size=4,
            hidden_size=64,
            num_layers=2,
            output_size=64,
            bidirectional=False,
            use_attention=True
        ).to(self.device).eval()
        
        self.semantic_encoder = SemanticEncoder()  # Singleton, already initialized
        
        # Initialize conflict classifier
        self.classifier = ConflictClassifier(
            input_dim=512,  # 64 + 64 + 384
            hidden_dim=256,
            output_dim=8,
            dropout=0.3
        ).to(self.device).eval()
        
        # Load checkpoints if provided
        if gnn_checkpoint and Path(gnn_checkpoint).exists():
            logger.info(f"Loading GNN from {gnn_checkpoint}")
            self.gnn_encoder.load_state_dict(torch.load(gnn_checkpoint, map_location=self.device))
        
        # Auto-detect LSTM checkpoint if not provided
        if not lstm_checkpoint:
            default_lstm = Path("checkpoints/lstm/best_model.pt")
            if default_lstm.exists():
                lstm_checkpoint = str(default_lstm)
        if lstm_checkpoint and Path(lstm_checkpoint).exists():
            logger.info(f"Loading LSTM from {lstm_checkpoint}")
            try:
                obj = torch.load(lstm_checkpoint, map_location=self.device)
                state = obj.get('model_state_dict', obj) if isinstance(obj, dict) else obj
                self.lstm_encoder.load_state_dict(state, strict=False)
            except Exception as e:
                logger.warning(f"Failed to load LSTM checkpoint: {e}. Continuing with random weights.")
        
        # Auto-detect classifier checkpoint if not provided
        if not classifier_checkpoint:
            default_cls = Path("checkpoints/conflict_classifier/best_model.pt")
            if default_cls.exists():
                classifier_checkpoint = str(default_cls)
        if classifier_checkpoint and Path(classifier_checkpoint).exists():
            logger.info(f"Loading Classifier from {classifier_checkpoint}")
            try:
                obj = torch.load(classifier_checkpoint, map_location=self.device)
                state = obj.get('model_state_dict', obj) if isinstance(obj, dict) else obj
                self.classifier.load_state_dict(state, strict=False)
            except Exception as e:
                logger.warning(f"Failed to load Classifier checkpoint: {e}. Continuing with random weights.")
        
        logger.info("✓ All models initialized and ready")
    
    def extract_all_features(self, incident: Dict[str, Any]) -> Tuple[Dict, List, str, List]:
        """
        Extract features for all models.
        
        Returns:
            gnn_features: Dict with nodes, edges, global_features
            lstm_sequence: List of 10 timesteps × 4 features
            semantic_text: String description
            conflict_context: List of 8 context features
        """
        gnn_features = self.pipeline.extract_gnn_features(incident)
        lstm_sequence = self.pipeline.extract_lstm_sequence(
            train_id=incident.get('train_id', 'TRAIN_001'),
            history_window=10
        )
        semantic_text = self.pipeline.extract_semantic_text(incident)
        conflict_context = self.pipeline.extract_conflict_features(incident)
        
        return gnn_features, lstm_sequence, semantic_text, conflict_context
    
    def generate_gnn_embedding(self, incident: Dict[str, Any]) -> np.ndarray:
        """
        Generate GNN embedding from graph features.
        
        Args:
            incident: Raw incident dict used to build the graph
        
        Returns:
            embedding: [64] numpy array
        """
        try:
            # Use production graph builder to construct PyG Data with correct features
            data = self.graph_builder.build_graph(incident)
            data = data.to(self.device)
            with torch.no_grad():
                embedding = self.gnn_encoder(data, return_embedding=True)
            arr = embedding.cpu().numpy()
            return np.squeeze(arr, axis=0) if arr.ndim == 2 else arr
        except Exception as e:
            logger.error(f"GNN embedding failed: {e}, returning zeros")
            return np.zeros(64)
    
    def generate_lstm_embedding(self, lstm_sequence: List) -> np.ndarray:
        """
        Generate LSTM embedding from sequence.
        
        Args:
            lstm_sequence: List of [timesteps, features]
        
        Returns:
            embedding: [64] numpy array
        """
        try:
            # Convert to tensor [1, 10, 4] (batch_size=1)
            sequence_tensor = torch.tensor(
                [lstm_sequence],
                dtype=torch.float32
            ).to(self.device)
            
            with torch.no_grad():
                embedding = self.lstm_encoder(sequence_tensor)
            
            return embedding.cpu().numpy()[0]  # [64]
        except Exception as e:
            logger.error(f"LSTM embedding failed: {e}, returning zeros")
            return np.zeros(64)
    
    def generate_semantic_embedding(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding from text.
        
        Args:
            text: Natural language description
        
        Returns:
            embedding: [384] numpy array
        """
        try:
            embedding = self.semantic_encoder.encode(text, normalize=True)
            return embedding
        except Exception as e:
            logger.error(f"Semantic embedding failed: {e}, returning zeros")
            return np.zeros(384)
    
    def concatenate_embeddings(
        self,
        gnn_emb: np.ndarray,
        lstm_emb: np.ndarray,
        semantic_emb: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate three embeddings into 512-dim vector.
        
        Args:
            gnn_emb: [64]
            lstm_emb: [64]
            semantic_emb: [384]
        
        Returns:
            combined: [512]
        """
        combined = np.concatenate([gnn_emb, lstm_emb, semantic_emb])
        return combined  # avoid shrinking logits; normalization can be learned by BN
    
    def predict_conflicts(
        self,
        incident: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        End-to-end prediction: incident → embeddings → conflict predictions.
        
        Args:
            incident: Raw incident dictionary
        
        Returns:
            {
                'conflict_types': ['Type1', 'Type2', ...],  # Predicted conflicts
                'probabilities': [0.85, 0.72, ...],  # Confidence for each
                'high_risk': Bool,  # True if any prob > 0.7
                'embeddings': {
                    'gnn': [64],
                    'lstm': [64],
                    'semantic': [384],
                    'combined': [512]
                }
            }
        """
        # Step 1: Extract features
        gnn_features, lstm_seq, semantic_text, conflict_ctx = self.extract_all_features(incident)
        
        # Step 2: Generate embeddings
        gnn_emb = self.generate_gnn_embedding(incident)
        lstm_emb = self.generate_lstm_embedding(lstm_seq)
        semantic_emb = self.generate_semantic_embedding(semantic_text)
        
        # Step 3: Concatenate
        combined_emb = self.concatenate_embeddings(gnn_emb, lstm_emb, semantic_emb)
        
        # Step 4: Predict conflicts
        combined_tensor = torch.tensor(combined_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.classifier(combined_tensor)
            conflict_probs_t = torch.sigmoid(logits)
        
        conflict_probs = conflict_probs_t.cpu().numpy()[0]  # [8]
        
        # Step 5: Format results
        conflict_types = [
            "Signal_Failure",
            "Track_Blockage",
            "Train_Collision_Risk",
            "Schedule_Conflict",
            "Power_Failure",
            "Weather_Impact",
            "Crew_Unavailability",
            "Emergency_Response_Needed"
        ]
        
        high_prob_conflicts = [
            (ctype, float(prob))
            for ctype, prob in zip(conflict_types, conflict_probs)
            if prob > 0.5
        ]
        
        return {
            'all_conflict_probs': {
                conflict_types[i]: float(conflict_probs[i])
                for i in range(8)
            },
            'detected_conflicts': [item[0] for item in high_prob_conflicts],
            'confidences': [item[1] for item in high_prob_conflicts],
            'high_risk': any(prob > 0.7 for prob in conflict_probs),
            'max_risk_type': conflict_types[np.argmax(conflict_probs)],
            'max_risk_prob': float(np.max(conflict_probs)),
            'embeddings': {
                'gnn': gnn_emb.tolist(),
                'lstm': lstm_emb.tolist(),
                'semantic': semantic_emb.tolist(),
                'combined': combined_emb.tolist()
            },
            'incident_id': incident.get('id', 'UNKNOWN')
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("CONFLICT PREDICTION PIPELINE - INTEGRATION TEST")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = ConflictPredictionPipeline(data_dir="data", device="cpu")
    
    # Create mock incident (using flat location fields from incidents.json)
    test_incident = {
        'id': 'INC_TEST_001',
        'type': 'signal_failure',
        'severity_level': 4,
        'location_id': 'STN_001',
        'location_name': 'Central Hub',
        'location_type': 'major_hub',
        'zone': 'core',
        'station_ids': ['STN_001', 'STN_002'],
        'segment_id': 'SEG_001',
        'trains_affected_count': 5,
        'network_load_pct': 75,
        'is_peak': True,
        'weather_condition': 'clear',
        'cascade_depth': 2,
        'hour_of_day': 9,
        'semantic_description': 'Signal failure at core zone during peak hours affecting 5 trains',
        'is_junction': True,
        'has_switches': True
    }
    
    print("\n📊 Input Incident:")
    print(f"  ID: {test_incident['id']}")
    print(f"  Type: {test_incident['type']}")
    print(f"  Severity: {test_incident['severity']}")
    print(f"  Trains Affected: {test_incident['trains_affected_count']}")
    
    # Run prediction
    print("\n⚙️  Generating embeddings...")
    results = pipeline.predict_conflicts(test_incident)
    
    print("\n✅ CONFLICT PREDICTIONS:")
    print(f"  High Risk: {results['high_risk']}")
    print(f"  Top Risk Type: {results['max_risk_type']}")
    print(f"  Top Risk Probability: {results['max_risk_prob']:.3f}")
    
    print("\n🔍 All Conflict Probabilities:")
    for conflict_type, prob in results['all_conflict_probs'].items():
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {conflict_type:30} {bar} {prob:.3f}")
    
    if results['detected_conflicts']:
        print("\n⚠️  Detected High-Confidence Conflicts (>50%):")
        for conflict, confidence in zip(results['detected_conflicts'], results['confidences']):
            print(f"  • {conflict}: {confidence:.3f}")
    
    print("\n📐 Embedding Dimensions:")
    print(f"  GNN: {len(results['embeddings']['gnn'])}")
    print(f"  LSTM: {len(results['embeddings']['lstm'])}")
    print(f"  Semantic: {len(results['embeddings']['semantic'])}")
    print(f"  Combined: {len(results['embeddings']['combined'])}")
    
    print("\n" + "=" * 70)
    print("✅ Pipeline test complete!")
    print("=" * 70)
