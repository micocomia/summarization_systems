# retrieval.py
import numpy as np
import random
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

class RetrievalSystem:
    """
    Retrieval system for finding relevant document chunks for summarization and QA.
    """
    def __init__(self, vector_store: VectorStore, 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retrieval system.
        
        Args:
            vector_store: VectorStore instance for document retrieval
            embedding_model: Name of the sentence-transformers model to use
        """
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(text, show_progress_bar=False)
    
    def retrieve(self, query: str, top_k: int = 5, 
                 filter_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks based on a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_topics: Optional list of topics to filter results
            
        Returns:
            List of relevant document chunks with metadata and scores
        """
        # Get the embedding for the query
        query_embedding = self.get_embedding(query)
        
        # Search the vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=top_k,
            filter_topics=filter_topics
        )
        
        # Compute additional relevance metrics
        return self.compute_relevance_scores(results)
    
    def compute_relevance_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute additional relevance metrics for retrieved results.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Documents with additional relevance metrics
        """
        if not results:
            return []
            
        # ChromaDB already provides a similarity score, but we can add more metrics
        
        # For example, we could add recency boost if documents have timestamps
        for doc in results:
            # Check if there's a created_at field in metadata
            if "metadata" in doc and "created_at" in doc["metadata"]:
                try:
                    # This is a simple example - you would use actual date parsing in production
                    # Add a small boost for newer documents (0.0 to 0.1)
                    recency_boost = 0.05  # Small constant boost for simplicity
                    doc["adjusted_score"] = doc["similarity"] + recency_boost
                except Exception:
                    doc["adjusted_score"] = doc["similarity"]
            else:
                doc["adjusted_score"] = doc["similarity"]
                
        # Re-sort results by adjusted score
        results.sort(key=lambda x: x["adjusted_score"], reverse=True)
            
        return results
    
    def retrieve_for_summarization(self, top_k: int = 20, 
                                  filter_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve a diverse set of contexts suitable for document summarization.
        
        Args:
            top_k: Number of contexts to retrieve
            filter_topics: Optional list of topics to filter results
            
        Returns:
            List of relevant document chunks for summarization
        """
        # Create diverse queries to get good document coverage
        diverse_queries = [
            "important information",
            "main points",
            "key concepts",
            "conclusions",
            "findings",
            "summary",
            "introduction",
            "methodology",
            "results",
            "discussion"
        ]
        
        # If specific topics are requested, add topic-specific queries
        if filter_topics:
            for topic in filter_topics:
                diverse_queries.append(f"information about {topic}")
                diverse_queries.append(f"key points about {topic}")
        
        all_results = []
        
        # Retrieve results for each query and combine
        for query in diverse_queries:
            results = self.retrieve(
                query=query,
                top_k=3  # Get a few results per query
            )
            all_results.extend(results)
        
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                unique_results.append(result)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Return the top_k results
        return unique_results[:top_k]
    
    def get_available_topics(self) -> List[str]:
        """
        Get all available topics in the vector store.
        
        Returns:
            List of unique topics
        """
        return self.vector_store.get_topics()