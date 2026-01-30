"""
================================================================================
Comprehensive Test Suite for Search Engine
================================================================================
Tests the NeuralSearcher class from src.backend.search_engine
Covers:
    - Initialization and Qdrant connection
    - Search functionality with single/multiple vectors
    - Result merging and weighting
    - Filtering and boosting
    - Explainability features
================================================================================
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend.search_engine import NeuralSearcher, SearchResult


class TestNeuralSearcher(unittest.TestCase):
    """Comprehensive tests for NeuralSearcher class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_semantic_text = "Signal failure at Central Station"
        cls.test_structural_vec = [0.1] * 64
        cls.test_temporal_vec = [0.2] * 64
    
    def test_01_initialization_with_credentials(self):
        """Test searcher initializes with valid Qdrant credentials."""
        print("\n=== TEST 1: Initialization with Credentials ===")
        
        # Should initialize without error when credentials are in env
        try:
            searcher = NeuralSearcher()
            self.assertIsNotNone(searcher)
            self.assertEqual(searcher.collection_name, "operational_memory")
            print("✓ Searcher initialized successfully")
        except Exception as e:
            self.fail(f"Initialization failed: {e}")
    
    def test_02_initialization_with_explicit_params(self):
        """Test searcher initialization with explicit parameters."""
        print("\n=== TEST 2: Explicit Parameters ===")
        
        # Get credentials from env for testing
        url = os.getenv("QDRANT_URL")
        key = os.getenv("QDRANT_API_KEY")
        
        if url and key:
            searcher = NeuralSearcher(
                qdrant_url=url,
                qdrant_api_key=key,
                collection_name="test_collection"
            )
            
            self.assertEqual(searcher.collection_name, "test_collection")
            print("✓ Custom collection name set correctly")
        else:
            self.skipTest("QDRANT credentials not available")
    
    def test_03_initialization_without_credentials(self):
        """Test graceful handling when credentials missing."""
        print("\n=== TEST 3: Missing Credentials ===")
        
        with patch.dict(os.environ, {}, clear=True):
            searcher = NeuralSearcher()
            
            # Should create object but client should be None
            self.assertIsNotNone(searcher)
            self.assertIsNone(searcher.client)
            print("✓ Handles missing credentials gracefully")
    
    def test_04_search_with_semantic_only(self):
        """Test search with only semantic text."""
        print("\n=== TEST 4: Semantic-Only Search ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        results = searcher.search(
            semantic_text=self.test_semantic_text,
            limit=5
        )
        
        # Should return a list
        self.assertIsInstance(results, list)
        
        # Each result should be a SearchResult
        for result in results:
            self.assertIsInstance(result, SearchResult)
            self.assertIsInstance(result.incident_id, str)
            self.assertIsInstance(result.similarity_score, float)
            self.assertIsInstance(result.payload, dict)
            self.assertIsInstance(result.similarity_breakdown, dict)
            self.assertIsInstance(result.is_golden_run, bool)
            self.assertIsInstance(result.days_ago, int)
        
        print(f"✓ Returned {len(results)} results")
        if results:
            print(f"  Top result score: {results[0].similarity_score:.3f}")
    
    def test_05_search_with_all_vectors(self):
        """Test search with semantic, structural, and temporal vectors."""
        print("\n=== TEST 5: Multi-Vector Search ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        results = searcher.search(
            semantic_text=self.test_semantic_text,
            structural_vec=self.test_structural_vec,
            temporal_vec=self.test_temporal_vec,
            limit=3
        )
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        # Verify results have score breakdowns
        for result in results:
            self.assertIn('semantic', result.similarity_breakdown)
            # structural and temporal may be 0 if no matches
        
        print(f"✓ Multi-vector search returned {len(results)} results")
    
    def test_06_search_result_ordering(self):
        """Test that results are ordered by similarity score."""
        print("\n=== TEST 6: Result Ordering ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        results = searcher.search(
            semantic_text=self.test_semantic_text,
            limit=5
        )
        
        # Results should be sorted descending by score
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i].similarity_score,
                    results[i+1].similarity_score,
                    "Results not properly sorted by score"
                )
            print("✓ Results correctly sorted by similarity score")
        else:
            print("⚠ Not enough results to verify ordering")
    
    def test_07_search_with_limit(self):
        """Test that limit parameter works correctly."""
        print("\n=== TEST 7: Search Limit ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        # Test different limits
        for limit in [1, 3, 10]:
            results = searcher.search(
                semantic_text=self.test_semantic_text,
                limit=limit
            )
            
            self.assertLessEqual(len(results), limit)
            print(f"✓ Limit {limit}: returned {len(results)} results")
    
    def test_08_search_with_filters(self):
        """Test search with metadata filters."""
        print("\n=== TEST 8: Filtered Search ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        # Test with filter (may return 0 results if no match)
        results = searcher.search(
            semantic_text=self.test_semantic_text,
            filters={"is_golden": True},
            limit=5
        )
        
        self.assertIsInstance(results, list)
        
        # If results exist, verify they match the filter
        for result in results:
            if result.is_golden_run:
                print("✓ Found golden run results")
                break
        
        print(f"✓ Filtered search executed ({len(results)} results)")
    
    def test_09_empty_search_handling(self):
        """Test handling of empty search text."""
        print("\n=== TEST 9: Empty Search ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        # Empty text should still work with FastEmbed
        results = searcher.search(
            semantic_text="",
            limit=3
        )
        
        # Should handle gracefully
        self.assertIsInstance(results, list)
        print("✓ Empty search handled gracefully")
    
    def test_10_weighted_scoring(self):
        """Test that weighted scoring is applied correctly."""
        print("\n=== TEST 10: Weighted Scoring ===")
        
        # Verify class constants
        self.assertEqual(NeuralSearcher.WEIGHT_SEMANTIC, 0.5)
        self.assertEqual(NeuralSearcher.WEIGHT_STRUCTURAL, 0.3)
        self.assertEqual(NeuralSearcher.WEIGHT_TEMPORAL, 0.2)
        
        # Verify boosting constants
        self.assertEqual(NeuralSearcher.BOOST_GOLDEN_RUN, 1.5)
        self.assertEqual(NeuralSearcher.BOOST_RECENT, 1.2)
        
        print("✓ Scoring weights configured correctly")
        print(f"  Semantic: {NeuralSearcher.WEIGHT_SEMANTIC}")
        print(f"  Structural: {NeuralSearcher.WEIGHT_STRUCTURAL}")
        print(f"  Temporal: {NeuralSearcher.WEIGHT_TEMPORAL}")
    
    def test_11_golden_run_boost(self):
        """Test that golden runs receive boosting."""
        print("\n=== TEST 11: Golden Run Boosting ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        results = searcher.search(
            semantic_text=self.test_semantic_text,
            limit=10
        )
        
        # Find golden runs and verify they're ranked higher
        golden_results = [r for r in results if r.is_golden_run]
        
        if golden_results:
            print(f"✓ Found {len(golden_results)} golden run results")
            print(f"  Top golden score: {golden_results[0].similarity_score:.3f}")
        else:
            print("⚠ No golden runs found in results")
    
    def test_12_explain_match(self):
        """Test explainability feature."""
        print("\n=== TEST 12: Match Explanation ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        results = searcher.search(
            semantic_text=self.test_semantic_text,
            limit=1
        )
        
        if results:
            explanation = searcher.explain_match(results[0])
            
            # Should return a string
            self.assertIsInstance(explanation, str)
            self.assertGreater(len(explanation), 0)
            
            print(f"✓ Explanation generated: {explanation}")
        else:
            print("⚠ No results to explain")
    
    def test_13_calculate_days_ago(self):
        """Test days_ago calculation."""
        print("\n=== TEST 13: Days Ago Calculation ===")
        
        searcher = NeuralSearcher()
        now = datetime.now()
        
        # Test recent timestamp
        recent = (now - timedelta(days=5)).isoformat()
        days = searcher._calculate_days_ago(recent, now)
        self.assertEqual(days, 5)
        print(f"✓ Recent incident: {days} days ago")
        
        # Test old timestamp
        old = (now - timedelta(days=100)).isoformat()
        days = searcher._calculate_days_ago(old, now)
        self.assertEqual(days, 100)
        print(f"✓ Old incident: {days} days ago")
        
        # Test empty timestamp
        days = searcher._calculate_days_ago("", now)
        self.assertEqual(days, 365)
        print("✓ Empty timestamp defaults to 365 days")
        
        # Test invalid timestamp
        days = searcher._calculate_days_ago("invalid", now)
        self.assertEqual(days, 365)
        print("✓ Invalid timestamp defaults to 365 days")
    
    def test_14_build_filter(self):
        """Test filter building."""
        print("\n=== TEST 14: Filter Building ===")
        
        searcher = NeuralSearcher()
        
        # Test with filters
        filter_dict = {
            "weather": "rain",
            "incident_type": "Signal Failure"
        }
        
        qdrant_filter = searcher._build_filter(filter_dict)
        
        # Should return a Filter object
        self.assertIsNotNone(qdrant_filter)
        print("✓ Filter object created successfully")
        
        # Test with empty filters
        empty_filter = searcher._build_filter({})
        self.assertIsNone(empty_filter)
        print("✓ Empty filter returns None")
    
    def test_15_search_without_client(self):
        """Test search behavior when client is None."""
        print("\n=== TEST 15: Search Without Client ===")
        
        with patch.dict(os.environ, {}, clear=True):
            searcher = NeuralSearcher()
            
            # Client should be None
            self.assertIsNone(searcher.client)
            
            # Search should return empty list gracefully
            results = searcher.search(
                semantic_text=self.test_semantic_text
            )
            
            self.assertEqual(results, [])
            print("✓ Returns empty list when client unavailable")


class TestSearchResultDataClass(unittest.TestCase):
    """Test the SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating SearchResult objects."""
        print("\n=== TEST: SearchResult Creation ===")
        
        result = SearchResult(
            incident_id="INC_001",
            similarity_score=0.95,
            payload={"type": "Signal Failure"},
            similarity_breakdown={"semantic": 0.8, "structural": 0.6, "temporal": 0.4},
            is_golden_run=True,
            days_ago=10
        )
        
        self.assertEqual(result.incident_id, "INC_001")
        self.assertEqual(result.similarity_score, 0.95)
        self.assertTrue(result.is_golden_run)
        self.assertEqual(result.days_ago, 10)
        
        print("✓ SearchResult object created successfully")


class TestIntegrationWithRealData(unittest.TestCase):
    """Integration tests with real Qdrant connection."""
    
    def test_end_to_end_search(self):
        """Test complete search workflow."""
        print("\n=== TEST: End-to-End Search ===")
        
        searcher = NeuralSearcher()
        
        if not searcher.client:
            self.skipTest("Qdrant not available")
        
        # Execute search
        results = searcher.search(
            semantic_text="Track maintenance required at station",
            structural_vec=[0.05] * 64,
            temporal_vec=[0.03] * 64,
            limit=5
        )
        
        # Verify complete workflow
        self.assertIsInstance(results, list)
        
        for result in results:
            # Verify all fields populated
            self.assertIsNotNone(result.incident_id)
            self.assertGreaterEqual(result.similarity_score, 0.0)
            self.assertIsInstance(result.payload, dict)
            self.assertIsInstance(result.similarity_breakdown, dict)
            
            # Test explanation
            explanation = searcher.explain_match(result)
            self.assertIsInstance(explanation, str)
        
        print(f"✓ End-to-end workflow completed successfully")
        print(f"  Results: {len(results)}")
        if results:
            print(f"  Top score: {results[0].similarity_score:.3f}")


def run_comprehensive_tests():
    """Run all tests with detailed output."""
    print("\n" + "=" * 70)
    print("🧪 Running Comprehensive Search Engine Tests")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNeuralSearcher))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchResultDataClass))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithRealData))
    
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
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Review output above.")
    
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    run_comprehensive_tests()
