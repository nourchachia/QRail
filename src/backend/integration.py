"""
Neural Rail Conductor - Integration Pipeline (Person 4)
src/backend/integration.py

ROLE: 🔗 THE GLUE CODE
    - Purpose: Connects all the separate AI pieces together. (run the app)
    - When to run: EVERY TIME you want to process a new incident report.
    - Flow: Parser -> Features -> Embeddings -> Search -> Classifier -> Predictor
    - Input: Raw text description
    - Output: Complete analysis and recommendations

Purpose:
    Glue code that connects all components of the AI pipeline:
    1. IncidentParser (Gemini) → Parses operator text
    2. DataFuelPipeline → Extracts features for AI models
    3. Encoders (GNN/LSTM/Semantic) → Generate embeddings
    4. NeuralSearcher → Find similar incidents in Qdrant
    5. ConflictClassifier → Predict conflicts
    6. OutcomePredictor → Predict resolution success

Usage:
    from src.backend.integration import IncidentPipeline
    
    pipeline = IncidentPipeline()
    
    # Process a new incident
    result = pipeline.process("Signal failure at Central Station. Peak hour. 5 trains affected.")
    
    # Get similar incidents
    similar = result['similar_incidents']
    
    # Get predicted conflicts
    conflicts = result['conflicts']
    
    # Get resolution recommendations
    recommendations = result['recommendations']
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import numpy as np

# Robust .env loading: Search up to 3 levels up
# 1. Start at this file's folder (src/backend/)
current_path = Path(__file__).resolve()

# 2. Go up the folder tree 3 times (backend -> src -> QRail)
for _ in range(3):
    env_path = current_path / ".env"
    
    # 3. If we find .env, force load it!
    if env_path.exists():
        print(f"✅ Found .env at: {env_path}")
        load_dotenv(env_path)
        break
        
    # 4. Move one level up for next try
    current_path = current_path.parent
else:
    # Fallback: Just try looking in current working directory
    load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class IncidentPipeline:
    """
    Complete incident processing pipeline.
    
    Flow:
        Operator Text
            ↓
        [IncidentParser] → Structured JSON (via Gemini)
            ↓
        [DataFuelPipeline] → Feature Vectors
            ↓
        [Encoders] → Embeddings (GNN=64, LSTM=64, Semantic=384)
            ↓
        [NeuralSearcher] → Similar Incidents from Qdrant
            ↓
        [ConflictClassifier] → Conflict Predictions
            ↓
        [OutcomePredictor] → Resolution Recommendations
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize all pipeline components.
        
        Args:
            data_dir: Path to data directory
            qdrant_url: Qdrant Cloud URL (or set QDRANT_URL env var)
            qdrant_api_key: Qdrant API key (or set QDRANT_API_KEY env var)
        """
        print("🚄 Initializing Neural Rail Conductor Pipeline...")
        
        # 1. Storage Manager (for loading data)
        from src.backend.database import StorageManager
        self.storage = StorageManager(data_dir=data_dir)
        print("   ✓ StorageManager ready")
        
        # 2. Incident Parser (Gemini)
        try:
            from src.backend.incident_parser import IncidentParser
            self.parser = IncidentParser(data_dir=data_dir)
            print("   ✓ IncidentParser ready (Gemini)")
        except Exception as e:
            print(f"   ⚠ IncidentParser failed: {e}")
            self.parser = None
        
        # 3. Feature Extractor
        try:
            from src.backend.feature_extractor import DataFuelPipeline
            self.feature_pipeline = DataFuelPipeline()
            print("   ✓ DataFuelPipeline ready")
        except Exception as e:
            print(f"   ⚠ DataFuelPipeline failed: {e}")
            self.feature_pipeline = None
        
        # 4. Neural Searcher (Qdrant Cloud)
        try:
            from src.backend.search_engine import NeuralSearcher
            self.searcher = NeuralSearcher(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key
            )
            if self.searcher.client:
                print("   ✓ NeuralSearcher ready (Qdrant Cloud)")
            else:
                print("   ⚠ NeuralSearcher: Qdrant Cloud not connected")
        except Exception as e:
            print(f"   ⚠ NeuralSearcher failed: {e}")
            self.searcher = None
        
        # 5. Semantic Encoder
        try:
            from src.models.semantic_encoder import SemanticEncoder
            self.semantic_encoder = SemanticEncoder()
            print("   ✓ SemanticEncoder ready")
        except Exception as e:
            print(f"   ⚠ SemanticEncoder failed: {e}")
            self.semantic_encoder = None
        
        # 6. Conflict Classifier (Model 4)
        try:
            from src.models.conflict_classifier import ConflictClassifier
            self.conflict_classifier = ConflictClassifier()
            print("   ✓ ConflictClassifier ready (Model 4)")
        except Exception as e:
            print(f"   ⚠ ConflictClassifier failed: {e}")
            self.conflict_classifier = None
        
        # 7. Outcome Predictor (Model 5)
        try:
            from src.models.outcome_predictor_xgb import OutcomePredictor
            self.outcome_predictor = OutcomePredictor()
            print("   ✓ OutcomePredictor ready (Model 5)")
        except Exception as e:
            print(f"   ⚠ OutcomePredictor failed: {e}")
            self.outcome_predictor = None
        
        print("✅ Pipeline initialization complete!")
    
    def process(self, incident_text: str) -> Dict[str, Any]:
        """
        Process an incident from raw text to recommendations.
        
        Args:
            incident_text: Natural language incident description
        
        Returns:
            {
                'parsed': {...},  # Structured incident data
                'features': {...},  # Feature vectors
                'embeddings': {...},  # AI embeddings
                'similar_incidents': [...],  # From Qdrant
                'conflicts': {...},  # Predicted conflicts
                'recommendations': [...]  # Ranked resolutions
            }
        """
        result = {
            'raw_text': incident_text,
            'parsed': {},
            'features': {},
            'embeddings': {},
            'similar_incidents': [],
            'conflicts': {},
            'recommendations': []
        }
        
        # Step 1: Parse with Gemini
        print("\n🔍 Step 1: Parsing incident...")
        if self.parser:
            try:
                result['parsed'] = self.parser.parse(incident_text)
                print(f"   Parsed: {result['parsed'].get('primary_failure_code', 'unknown')}")
            except Exception as e:
                print(f"   ⚠ Parse failed: {e}")
                result['parsed'] = self._fallback_parse(incident_text)
        else:
            result['parsed'] = self._fallback_parse(incident_text)
        
        # Step 2: Extract features
        print("📊 Step 2: Extracting features...")
        if self.feature_pipeline:
            try:
                result['features'] = self.feature_pipeline.extract_all_features(result['parsed'])
                print(f"   GNN features: {len(result['features'].get('gnn', {}).get('node_features', []))} nodes")
            except Exception as e:
                print(f"   ⚠ Feature extraction failed: {e}")
        
        # Step 3: Generate semantic embedding
        print("🧠 Step 3: Generating embeddings...")
        semantic_vec = [0.0] * 384  # Default
        structural_vec = [0.0] * 64
        temporal_vec = [0.0] * 64
        
        if self.semantic_encoder:
            try:
                text = result['features'].get('semantic_text', incident_text)
                semantic_vec = self.semantic_encoder.encode(text).tolist()
                print(f"   Semantic: {len(semantic_vec)}-dim")
            except Exception as e:
                print(f"   ⚠ Semantic encoding failed: {e}")
        
        result['embeddings'] = {
            'semantic': semantic_vec,
            'structural': structural_vec,
            'temporal': temporal_vec
        }
        
        # Step 4: Search similar incidents
        print("🔎 Step 4: Searching similar incidents...")
        if self.searcher and self.searcher.client:
            try:
                similar = self.searcher.search(
                    semantic_vec=semantic_vec,
                    structural_vec=structural_vec,
                    temporal_vec=temporal_vec,
                    limit=5
                )
                result['similar_incidents'] = [
                    {
                        'incident_id': s.incident_id,
                        'score': s.similarity_score,
                        'is_golden': s.is_golden_run,
                        'explanation': self.searcher.explain_match(s)
                    }
                    for s in similar
                ]
                print(f"   Found {len(similar)} similar incidents")
            except Exception as e:
                print(f"   ⚠ Search failed: {e}")
        
        # Step 5: Predict conflicts
        print("⚠️ Step 5: Predicting conflicts...")
        if self.conflict_classifier:
            try:
                gnn_vec = np.array(structural_vec, dtype=np.float32)
                lstm_vec = np.array(temporal_vec, dtype=np.float32)
                sem_vec = np.array(semantic_vec, dtype=np.float32)
                
                result['conflicts'] = self.conflict_classifier.predict(
                    gnn_vec, lstm_vec, sem_vec
                )
                
                # Find high-probability conflicts
                high_conflicts = [k for k, v in result['conflicts'].items() if v > 0.5]
                print(f"   Detected: {high_conflicts if high_conflicts else 'None'}")
            except Exception as e:
                print(f"   ⚠ Conflict prediction failed: {e}")
        
        # Step 6: Generate recommendations
        print("💡 Step 6: Generating recommendations...")
        result['recommendations'] = self._generate_recommendations(result)
        print(f"   Generated {len(result['recommendations'])} recommendations")
        
        print("\n✅ Processing complete!")
        return result
    
    def _fallback_parse(self, text: str) -> Dict:
        """Simple fallback when Gemini isn't available."""
        return {
            'primary_failure_code': 'UNKNOWN',
            'estimated_delay_minutes': 30,
            'confidence': 0.5,
            'reasoning': 'Fallback parsing'
        }
    
    def _generate_recommendations(self, result: Dict) -> List[Dict]:
        """
        Generate resolution recommendations based on similar incidents.
        """
        recommendations = []
        
        # Use similar incidents to suggest resolutions
        for incident in result.get('similar_incidents', [])[:3]:
            if incident.get('is_golden'):
                recommendations.append({
                    'strategy': 'Based on Golden Run',
                    'incident_id': incident['incident_id'],
                    'confidence': 0.9,
                    'score': incident['score']
                })
        
        # Default recommendations if no similar found
        if not recommendations:
            recommendations = [
                {'strategy': 'HOLD_TRAIN', 'confidence': 0.7},
                {'strategy': 'REROUTE', 'confidence': 0.6},
                {'strategy': 'EXTEND_DWELL', 'confidence': 0.5}
            ]
        
        return recommendations


# ========== Example Usage ==========

if __name__ == "__main__":
    print("=" * 60)
    print("🚄 Neural Rail Conductor - Integration Pipeline Test")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = IncidentPipeline()
    
    # Test with sample incident
    test_text = """
    Signal failure at Central Station during morning peak. 
    Heavy rain conditions. 5 trains affected with cascade delays.
    Platform 3 and 4 blocked.
    """
    
    print("\n" + "=" * 60)
    print("📋 Processing Test Incident")
    print("=" * 60)
    
    result = pipeline.process(test_text)
    
    print("\n" + "=" * 60)
    print("📊 Results Summary")
    print("=" * 60)
    print(f"Failure Code: {result['parsed'].get('primary_failure_code', 'N/A')}")
    print(f"Similar Incidents: {len(result['similar_incidents'])}")
    print(f"High-Risk Conflicts: {[k for k, v in result['conflicts'].items() if v > 0.5]}")
    print(f"Top Recommendation: {result['recommendations'][0] if result['recommendations'] else 'None'}")
