"""
================================================================================
Comprehensive Test Suite for Integration Pipeline
================================================================================
Tests the IncidentPipeline class from src.backend.integration
Covers:
    - End-to-End pipeline execution
    - Component initialization (graceful degradation)
    - Error handling and fallbacks
    - Data integrity and output structure
================================================================================
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import numpy as np

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend.integration import IncidentPipeline


class TestIncidentPipeline(unittest.TestCase):
    """Comprehensive integration pipeline tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.test_incident_text = """
        Signal failure at Central Station during morning peak.
        Heavy rain conditions. 5 trains affected with cascade delays.
        Platform 3 and 4 blocked. Estimated 25 minute delay.
        """
        
        cls.expected_result_keys = [
            'raw_text',
            'parsed',
            'features',
            'embeddings',
            'similar_incidents',
            'conflicts',
            'recommendations'
        ]
    
    def test_01_pipeline_initialization(self):
        """Test that pipeline initializes without crashing."""
        print("\n=== TEST 1: Pipeline Initialization ===")
        
        try:
            pipeline = IncidentPipeline()
            self.assertIsNotNone(pipeline)
            self.assertIsNotNone(pipeline.storage)
            print("✓ Pipeline initialized successfully")
        except Exception as e:
            self.fail(f"Pipeline initialization failed: {e}")
    
    def test_02_end_to_end_processing(self):
        """Test complete pipeline execution with real components."""
        print("\n=== TEST 2: End-to-End Processing ===")
        
        pipeline = IncidentPipeline()
        result = pipeline.process(self.test_incident_text)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        for key in self.expected_result_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # Verify raw text is preserved
        self.assertEqual(result['raw_text'], self.test_incident_text)
        
        # Verify parsed data structure
        self.assertIsInstance(result['parsed'], dict)
        
        # Verify embeddings structure
        self.assertIsInstance(result['embeddings'], dict)
        self.assertIn('semantic', result['embeddings'])
        self.assertIn('structural', result['embeddings'])
        self.assertIn('temporal', result['embeddings'])
        
        # Verify embeddings are lists of numbers
        for emb_type in ['semantic', 'structural', 'temporal']:
            emb = result['embeddings'][emb_type]
            self.assertIsInstance(emb, list)
            if len(emb) > 0:
                self.assertTrue(all(isinstance(x, (int, float)) for x in emb))
        
        # Verify similar incidents structure
        self.assertIsInstance(result['similar_incidents'], list)
        
        # Verify conflicts structure
        self.assertIsInstance(result['conflicts'], dict)
        
        # Verify recommendations structure
        self.assertIsInstance(result['recommendations'], list)
        
        print(f"✓ Pipeline processed incident successfully")
        print(f"  - Parsed data: {result['parsed'].get('primary_failure_code', 'N/A')}")
        print(f"  - Similar incidents found: {len(result['similar_incidents'])}")
        print(f"  - Recommendations generated: {len(result['recommendations'])}")
    
    def test_03_embedding_dimensions(self):
        """Test that embeddings have correct dimensions."""
        print("\n=== TEST 3: Embedding Dimensions ===")
        
        pipeline = IncidentPipeline()
        result = pipeline.process(self.test_incident_text)
        
        embeddings = result['embeddings']
        
        # Check semantic embedding (384-dim from FastEmbed)
        # Note: May be zeros if optimization is enabled
        self.assertEqual(len(embeddings['semantic']), 384)
        
        # Check structural embedding (64-dim from GNN)
        self.assertEqual(len(embeddings['structural']), 64)
        
        # Check temporal embedding (64-dim from LSTM)
        self.assertEqual(len(embeddings['temporal']), 64)
        
        print("✓ All embeddings have correct dimensions")
    
    def test_04_parser_fallback(self):
        """Test fallback parsing when Gemini is unavailable."""
        print("\n=== TEST 4: Parser Fallback ===")
        
        with patch('src.backend.incident_parser.IncidentParser') as MockParser:
            # Simulate parser initialization failure
            MockParser.side_effect = Exception("Gemini API unavailable")
            
            pipeline = IncidentPipeline()
            result = pipeline.process(self.test_incident_text)
            
            # Should still return a result
            self.assertIsInstance(result, dict)
            self.assertIn('parsed', result)
            
            # Fallback should have been used
            self.assertIn('primary_failure_code', result['parsed'])
            
            print("✓ Fallback parser works correctly")
    
    def test_05_missing_components_graceful_degradation(self):
        """Test that pipeline handles missing components gracefully."""
        print("\n=== TEST 5: Graceful Degradation ===")
        
        # Test with various component failures
        with patch('src.backend.search_engine.NeuralSearcher') as MockSearcher:
            MockSearcher.side_effect = Exception("Qdrant unavailable")
            
            pipeline = IncidentPipeline()
            result = pipeline.process(self.test_incident_text)
            
            # Should still complete processing
            self.assertIsInstance(result, dict)
            self.assertIn('similar_incidents', result)
            
            print("✓ Pipeline degrades gracefully with missing searcher")
    
    def test_06_conflict_classifier_output(self):
        """Test conflict classifier output structure."""
        print("\n=== TEST 6: Conflict Classifier Output ===")
        
        pipeline = IncidentPipeline()
        result = pipeline.process(self.test_incident_text)
        
        conflicts = result['conflicts']
        
        # Check if conflicts is a dict
        self.assertIsInstance(conflicts, dict)
        
        # Expected conflict types
        expected_conflicts = [
            'headway_violation',
            'platform_oversubscription',
            'crew_timeout',
            'signal_blockage',
            'track_capacity',
            'power_demand',
            'safety_margin',
            'passenger_overflow'
        ]
        
        # If conflicts are detected, verify structure
        if conflicts:
            for conflict_name in conflicts.keys():
                # Should be a probability value
                prob = conflicts[conflict_name]
                self.assertIsInstance(prob, (int, float))
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)
        
        print(f"✓ Conflict classifier returned valid output")
        if conflicts:
            high_risk = {k: v for k, v in conflicts.items() if v > 0.5}
            if high_risk:
                print(f"  - High-risk conflicts: {list(high_risk.keys())}")
    
    def test_07_recommendations_structure(self):
        """Test recommendations output structure."""
        print("\n=== TEST 7: Recommendations Structure ===")
        
        pipeline = IncidentPipeline()
        result = pipeline.process(self.test_incident_text)
        
        recommendations = result['recommendations']
        
        # Should be a list
        self.assertIsInstance(recommendations, list)
        
        # Check structure of recommendations if any exist
        if recommendations:
            for rec in recommendations:
                self.assertIsInstance(rec, dict)
                
                # Should have key fields
                self.assertIn('strategy', rec)
                self.assertIn('confidence', rec)
                
                # Confidence should be between 0 and 1
                self.assertGreaterEqual(rec['confidence'], 0.0)
                self.assertLessEqual(rec['confidence'], 1.0)
        
        print(f"✓ Recommendations have valid structure")
        print(f"  - Number of recommendations: {len(recommendations)}")
    
    def test_08_similar_incidents_structure(self):
        """Test similar incidents output structure."""
        print("\n=== TEST 8: Similar Incidents Structure ===")
        
        pipeline = IncidentPipeline()
        result = pipeline.process(self.test_incident_text)
        
        similar = result['similar_incidents']
        
        # Should be a list
        self.assertIsInstance(similar, list)
        
        # Check structure if any matches exist
        if similar:
            for incident in similar:
                self.assertIsInstance(incident, dict)
                
                # Should have key fields
                self.assertIn('incident_id', incident)
                self.assertIn('score', incident)
                self.assertIn('is_golden', incident)
                
                # Score should be between 0 and 1
                self.assertGreaterEqual(incident['score'], 0.0)
                self.assertLessEqual(incident['score'], 1.0)
                
                # is_golden should be boolean
                self.assertIsInstance(incident['is_golden'], bool)
        
        print(f"✓ Similar incidents have valid structure")
        print(f"  - Number of matches: {len(similar)}")
        if similar:
            golden_count = sum(1 for s in similar if s['is_golden'])
            print(f"  - Golden runs: {golden_count}")
    
    def test_09_feature_extraction(self):
        """Test feature extraction produces expected structure."""
        print("\n=== TEST 9: Feature Extraction ===")
        
        pipeline = IncidentPipeline()
        result = pipeline.process(self.test_incident_text)
        
        features = result['features']
        
        # Should have the main feature types
        expected_features = ['gnn', 'lstm', 'semantic_text', 'conflict_context']
        
        for feat_type in expected_features:
            self.assertIn(feat_type, features, f"Missing feature type: {feat_type}")
        
        # GNN features should have nodes
        if 'gnn' in features and features['gnn']:
            gnn = features['gnn']
            self.assertIsInstance(gnn, dict)
        
        print("✓ Feature extraction produces valid structure")
    
    def test_10_empty_input_handling(self):
        """Test pipeline handles empty/minimal input."""
        print("\n=== TEST 10: Empty Input Handling ===")
        
        pipeline = IncidentPipeline()
        
        # Test with minimal input
        minimal_text = "Issue reported"
        result = pipeline.process(minimal_text)
        
        # Should still return valid structure
        self.assertIsInstance(result, dict)
        for key in self.expected_result_keys:
            self.assertIn(key, result)
        
        print("✓ Pipeline handles minimal input gracefully")
    
    def test_11_multiple_incidents_processing(self):
        """Test processing multiple incidents sequentially."""
        print("\n=== TEST 11: Multiple Incidents Processing ===")
        
        pipeline = IncidentPipeline()
        
        incidents = [
            "Signal failure at Central Station",
            "Track blocked by debris at North Terminal",
            "Power outage affecting multiple lines"
        ]
        
        results = []
        for incident_text in incidents:
            result = pipeline.process(incident_text)
            results.append(result)
            
            # Verify each result is valid
            self.assertIsInstance(result, dict)
            for key in self.expected_result_keys:
                self.assertIn(key, result)
        
        self.assertEqual(len(results), 3)
        
        print(f"✓ Successfully processed {len(results)} incidents")
    
    def test_12_truth_attribution(self):
        """Test that truth attribution is present in results."""
        print("\n=== TEST 12: Truth Attribution ===")
        
        pipeline = IncidentPipeline()
        result = pipeline.process(self.test_incident_text)
        
        # Should have truth attribution
        self.assertIn('truth_attribution', result)
        
        attribution = result['truth_attribution']
        self.assertIsInstance(attribution, dict)
        
        # Should have key attribution fields
        expected_fields = [
            'parsing_logic',
            'station_data',
            'weather_data',
            'train_identity',
            'mathematical_vectors',
            'similarity_search'
        ]
        
        for field in expected_fields:
            self.assertIn(field, attribution, f"Missing attribution field: {field}")
        
        print("✓ Truth attribution is present and complete")


class TestPipelineComponentIntegration(unittest.TestCase):
    """Test integration between pipeline components."""
    
    def test_parser_to_features_flow(self):
        """Test data flows correctly from parser to feature extraction."""
        print("\n=== TEST: Parser to Features Flow ===")
        
        pipeline = IncidentPipeline()
        
        # Process and check flow
        test_text = "Signal failure at Central Station"
        result = pipeline.process(test_text)
        
        # Parsed data should influence features
        self.assertIn('parsed', result)
        self.assertIn('features', result)
        
        # Features should be based on parsed data
        parsed = result['parsed']
        features = result['features']
        
        self.assertIsNotNone(features)
        
        print("✓ Data flows correctly from parser to features")
    
    def test_features_to_embeddings_flow(self):
        """Test features correctly generate embeddings."""
        print("\n=== TEST: Features to Embeddings Flow ===")
        
        pipeline = IncidentPipeline()
        result = pipeline.process("Track blocked at station")
        
        # Features should generate embeddings
        self.assertIn('features', result)
        self.assertIn('embeddings', result)
        
        embeddings = result['embeddings']
        
        # All embedding types should be present
        self.assertIn('semantic', embeddings)
        self.assertIn('structural', embeddings)
        self.assertIn('temporal', embeddings)
        
        print("✓ Features correctly generate embeddings")


def run_comprehensive_tests():
    """Run all tests with detailed output."""
    print("\n" + "=" * 70)
    print("🧪 Running Comprehensive Integration Pipeline Tests")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIncidentPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineComponentIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Review output above.")
    
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    run_comprehensive_tests()
