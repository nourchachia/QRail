"""
Neural Rail Conductor - Search Engine 
src/backend/search_engine.py

ROLE: 🔎 THE LIBRARIAN
    - Purpose: Finds similar past events in the database (Qdrant).
    - Input: Mathematical vectors (embeddings)
    - Output: "This looks like the incident from 2 years ago"

Purpose:
    Performs weighted multi-vector search in Qdrant Cloud to find similar historical incidents.
    Uses the formula: 0.5*Semantic + 0.3*Structural + 0.2*Temporal

Integration:
    - Connects to Qdrant Cloud using QDRANT_URL and QDRANT_API_KEY from .env
    - Called by the main API when new incidents occur
    - Returns ranked similar incidents with explainability

Usage:
    from src.backend.search_engine import NeuralSearcher
    
    searcher = NeuralSearcher()
    results = searcher.search(
        semantic_vec=[...],   # 384-dim from Sentence-Transformer
        structural_vec=[...], # 64-dim from GNN
        temporal_vec=[...]    # 64-dim from LSTM
    )
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue
)


@dataclass
class SearchResult:
    """Structured search result with explainability."""
    incident_id: str
    similarity_score: float
    payload: Dict[str, Any]
    similarity_breakdown: Dict[str, float]
    is_golden_run: bool
    days_ago: int


class NeuralSearcher:
    """
    Weighted multi-vector search engine using Qdrant Cloud.
    
    WEIGHTS (from blueprint Part 4.3):
        - Semantic: 0.5 (meaning/context)
        - Structural: 0.3 (topology/location)
        - Temporal: 0.2 (cascade pattern)
    
    BOOSTING:
        - Golden Runs: 1.5x multiplier
        - Recent incidents (<30 days): 1.2x multiplier
    """
    
    # Configurable weights
    WEIGHT_SEMANTIC = 0.5
    WEIGHT_STRUCTURAL = 0.3
    WEIGHT_TEMPORAL = 0.2
    
    # Boosting factors
    BOOST_GOLDEN_RUN = 1.5
    BOOST_RECENT = 1.2
    
    def __init__(
        self, 
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "operational_memory"
    ):
        """
        Initialize the Neural Searcher with Qdrant Cloud.
        
        Args:
            qdrant_url: Qdrant Cloud URL (or set QDRANT_URL env var)
            qdrant_api_key: Qdrant API key (or set QDRANT_API_KEY env var)
            collection_name: Name of the Qdrant collection
        """
        self.collection_name = collection_name
        
        # Get credentials from env if not provided
        url = qdrant_url or os.getenv("QDRANT_URL")
        api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        if url and api_key:
            try:
                self.client = QdrantClient(url=url, api_key=api_key)
                # Test connection
                self.client.get_collections()
                print(f"✅ Connected to Qdrant Cloud: {url[:40]}...")
            except Exception as e:
                print(f"❌ Qdrant Cloud connection failed: {e}")
                self.client = None
        else:
            print("⚠️ QDRANT_URL or QDRANT_API_KEY not set in .env")
            self.client = None
    
    def search(
        self,
        semantic_vec: List[float],
        structural_vec: Optional[List[float]] = None,
        temporal_vec: Optional[List[float]] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Execute weighted triple-vector search.
        
        Args:
            semantic_vec: 384-dim semantic embedding (REQUIRED)
            structural_vec: 64-dim GNN embedding (optional)
            temporal_vec: 64-dim LSTM embedding (optional)
            limit: Max results to return
            filters: Optional filters like {"weather": "rain", "incident_type": "Signal Failure"}
        
        Returns:
            List of SearchResult objects sorted by weighted score
        """
        if not self.client:
            print("⚠️ Qdrant client not available")
            return []
        
        # Build Qdrant filter if provided
        qdrant_filter = self._build_filter(filters) if filters else None
        
        try:
            # DEBUG: Check collection exists and has data
            try:
                collection_info = self.client.get_collection(self.collection_name)
                print(f"🔍 DEBUG: Collection '{self.collection_name}' exists")
                print(f"   Points in collection: {collection_info.points_count}")
            except Exception as e:
                print(f"❌ DEBUG: Failed to get collection info: {e}")
                return []
            
            # Primary search using semantic vector (most important)
            try:
                print(f"🔍 DEBUG: Executing search with semantic_vec length={len(semantic_vec)}")
                # UPDATED: Use query_points instead of search
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=semantic_vec,
                    using="semantic",
                    query_filter=qdrant_filter,
                    limit=limit * 3
                )
                semantic_results = response.points
                print(f"   ✓ Semantic search returned {len(semantic_results)} results")
            except AttributeError as e:
                print(f"❌ CRITICAL QDRANT ERROR: {e}")
                print(f"   Available methods on client: {[m for m in dir(self.client) if not m.startswith('_')]}")
                return []
            except Exception as e:
                print(f"❌ SEARCH ERROR: {e}")
                return []
            
            # If we have structural/temporal vectors, do additional searches for fusion
            structural_results = []
            temporal_results = []
            
            if structural_vec:
                try:
                    resp = self.client.query_points(
                        collection_name=self.collection_name,
                        query=structural_vec,
                        using="structural",
                        query_filter=qdrant_filter,
                        limit=limit * 2
                    )
                    structural_results = resp.points
                except Exception:
                    pass # Ignore secondary errors
            
            if temporal_vec:
                try:
                    resp = self.client.query_points(
                        collection_name=self.collection_name,
                        query=temporal_vec,
                        using="temporal",
                        query_filter=qdrant_filter,
                        limit=limit * 2
                    )
                    temporal_results = resp.points
                except Exception:
                    pass
            
            # Merge and re-rank results
            merged = self._merge_results(
                semantic_hits=semantic_results,
                structural_hits=structural_results,
                temporal_hits=temporal_results
            )
            
            return merged[:limit]
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Convert simple filter dict to Qdrant Filter object."""
        conditions = []
        
        for key, value in filters.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        
        return Filter(must=conditions) if conditions else None
    
    def _merge_results(
        self,
        semantic_hits: List,
        structural_hits: List,
        temporal_hits: List
    ) -> List[SearchResult]:
        """
        Merge results from multiple vector searches with weighted scoring.
        
        Algorithm:
        1. Collect all unique incident IDs
        2. Calculate weighted score for each
        3. Apply boosting factors
        4. Sort by final score
        """
        from datetime import datetime
        
        merged: Dict[int, Dict[str, Any]] = {}
        
        # Helper to add hits to merged dict
        def add_hits(hits, vector_name, weight):
            for hit in hits:
                inc_id = hit.id
                
                if inc_id not in merged:
                    merged[inc_id] = {
                        "payload": hit.payload or {},
                        "scores": {"semantic": 0, "structural": 0, "temporal": 0},
                        "weighted_total": 0.0
                    }
                
                merged[inc_id]["scores"][vector_name] = hit.score
                merged[inc_id]["weighted_total"] += hit.score * weight
        
        # Process each vector's results
        add_hits(semantic_hits, "semantic", self.WEIGHT_SEMANTIC)
        add_hits(structural_hits, "structural", self.WEIGHT_STRUCTURAL)
        add_hits(temporal_hits, "temporal", self.WEIGHT_TEMPORAL)
        
        # Convert to SearchResult objects with boosting
        results = []
        now = datetime.now()
        
        for inc_id, data in merged.items():
            payload = data["payload"]
            base_score = data["weighted_total"]
            
            # Apply boosts
            final_score = base_score
            
            # Golden Run boost
            is_golden = payload.get("is_golden", False)
            if is_golden:
                final_score *= self.BOOST_GOLDEN_RUN
            
            # Recency boost
            days_ago = self._calculate_days_ago(payload.get("timestamp", ""), now)
            if days_ago < 30:
                final_score *= self.BOOST_RECENT
            
            results.append(SearchResult(
                incident_id=str(payload.get("incident_id", inc_id)),
                similarity_score=final_score,
                payload=payload,
                similarity_breakdown=data["scores"],
                is_golden_run=is_golden,
                days_ago=days_ago
            ))
        
        # Sort by final weighted score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    def _calculate_days_ago(self, timestamp_str: str, now) -> int:
        """Calculate how many days ago an incident occurred."""
        from datetime import datetime
        
        if not timestamp_str:
            return 365
        
        try:
            incident_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            delta = now - incident_time.replace(tzinfo=None)
            return max(0, delta.days)
        except Exception:
            return 365
    
    def explain_match(self, result: SearchResult) -> str:
        """
        Generate human-readable explanation of why this incident matched.
        
        Used for the "Explainability" feature in the dashboard.
        """
        breakdown = result.similarity_breakdown
        payload = result.payload
        parts = []
        
        # Dominant factor
        max_factor = max(breakdown.items(), key=lambda x: x[1])
        if max_factor[0] == "semantic":
            parts.append(f"📝 Similar description ({max_factor[1]:.0%})")
        elif max_factor[0] == "structural":
            parts.append(f"🗺️ Similar location ({max_factor[1]:.0%})")
        else:
            parts.append(f"📈 Similar cascade ({max_factor[1]:.0%})")
        
        # Context
        if payload.get("weather"):
            parts.append(f"🌤️ {payload['weather']}")
        
        if result.is_golden_run:
            parts.append("⭐ Golden Run")
        
        if result.days_ago < 30:
            parts.append(f"🕐 {result.days_ago}d ago")
        
        return " | ".join(parts)


# Backward compatibility alias
HybridSearcher = NeuralSearcher


# ========== Example Usage ==========

if __name__ == "__main__":
    print("=" * 50)
    print("🕸️ Neural Rail Conductor - Search Engine Test")
    print("=" * 50)
    
    searcher = NeuralSearcher()
    
    if searcher.client:
        print("✅ Connected to Qdrant")
        
        # Test search with dummy vectors
        test_semantic = [0.1] * 384
        test_structural = [0.1] * 64
        test_temporal = [0.1] * 64
        
        results = searcher.search(
            semantic_vec=test_semantic,
            structural_vec=test_structural,
            temporal_vec=test_temporal,
            limit=3
        )
        
        print(f"\n🔍 Found {len(results)} results")
        
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"   ID: {r.incident_id}")
            print(f"   Score: {r.similarity_score:.3f}")
            print(f"   Golden: {r.is_golden_run}")
            print(f"   Explain: {searcher.explain_match(r)}")
    else:
        print("⚠️ Qdrant not running locally")
        print("\nTo start Qdrant locally:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        print("\nSearch weights configured:")
        print(f"   Semantic: {NeuralSearcher.WEIGHT_SEMANTIC}")
        print(f"   Structural: {NeuralSearcher.WEIGHT_STRUCTURAL}")
        print(f"   Temporal: {NeuralSearcher.WEIGHT_TEMPORAL}")
