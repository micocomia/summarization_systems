import os
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple, Optional, Any, Union

class VectorStore:
    """
    Vector database for storing and retrieving document embeddings using ChromaDB.
    """
    def __init__(self, collection_name: str = "document_collection", 
                 persist_directory: str = "chroma_data",
                 embedding_dim: int = 768):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to save the vector store data
            embedding_dim: Dimension of the embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dim = embedding_dim
        
        # Create persistence directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} documents")
        except Exception as e:
            # Collection doesn't exist or other error, create it
            print(f"Collection error: {str(e)}. Creating new collection.")
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection '{collection_name}'")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with at least:
                - 'embedding': numpy array of the document embedding
                - 'content': text content of the document
                - 'metadata': dictionary with source, chunk_id, topics, etc.
        """
        if not documents:
            return
            
        # Prepare data for ChromaDB
        ids = [f"doc_{i}_{doc['metadata'].get('chunk_id', hash(doc['content']) % 10000)}" 
               for i, doc in enumerate(documents)]
        
        embeddings = [doc['embedding'].tolist() if isinstance(doc['embedding'], np.ndarray) 
                      else doc['embedding'] for doc in documents]
        
        documents_text = [doc['content'] for doc in documents]
        
        # Process metadata to make it compatible with ChromaDB
        metadatas = []
        for doc in documents:
            # Create a copy of the metadata to avoid modifying the original
            processed_metadata = dict(doc['metadata'])
            
            # Convert list values to strings (JSON format for better parsing later)
            for key, value in processed_metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = ','.join(str(item) for item in value)
                    
            metadatas.append(processed_metadata)
        
        # Add to ChromaDB collection in batches
        # ChromaDB works better with smaller batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents_text[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
        print(f"Added {len(documents)} documents to ChromaDB collection")
    
    def search(self, query_embedding: Union[np.ndarray, List[float]], k: int = 5, 
               filter_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: Embedding of the query (numpy array or list)
            k: Number of results to return
            filter_topics: Optional list of topics to filter results
            
        Returns:
            List of document dictionaries with content, metadata, and similarity score
        """
        # Prepare query embedding
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        # For topic filtering, we'll first get results without filtering in ChromaDB
        # and then apply post-filtering ourselves
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 10,  # Get more results to ensure we have enough after filtering
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results into consistent format with manual topic filtering
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                
                # Apply topic filtering if needed
                if filter_topics and len(filter_topics) > 0 and "topics" in metadata:
                    topics_str = metadata["topics"]
                    topics_list = [t.strip() for t in topics_str.split(',')]
                    
                    # Only include if any of the filter topics is in the topics list
                    if not any(topic in topics_list for topic in filter_topics):
                        continue
                
                # Convert distance to score (lower distance = higher score)
                distance = results["distances"][0][i]
                similarity_score = 1.0 / (1.0 + distance)  # Convert to a 0-1 scale (higher is better)
                
                formatted_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": metadata,
                    "score": distance,
                    "similarity": similarity_score
                })
        
        # Trim to requested k after filtering
        return formatted_results[:k]
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search directly with a text query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        # Import sentence transformers for embedding
        try:
            from sentence_transformers import SentenceTransformer
            
            # Create embedding
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode(query, show_progress_bar=False)
            
            # Search using the embedding
            return self.search(query_embedding=query_embedding, k=k)
        except ImportError:
            raise ImportError("SentenceTransformer is required for semantic_search. Please install it.")
            
    def get_collection_size(self) -> int:
        """Get the number of documents in the store."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear the vector store."""
        # Delete the collection and create a new one
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        print(f"Cleared collection '{self.collection_name}'")
        
    def get_topics(self) -> List[str]:
        """Get all unique topics in the collection."""
        topics = set()
        
        # Get all documents to extract topics
        # Note: For large collections, you'd want to implement pagination here
        if self.collection.count() > 0:
            all_results = self.collection.get(include=["metadatas"])
            
            for metadata in all_results["metadatas"]:
                if "topics" in metadata:
                    if isinstance(metadata["topics"], str) and ',' in metadata["topics"]:
                        # Split comma-separated topics
                        topic_list = [topic.strip() for topic in metadata["topics"].split(',')]
                        topics.update(topic_list)
                    elif isinstance(metadata["topics"], str):
                        topics.add(metadata["topics"])
                    
        return sorted(list(topics))
    
    def get_document_stats(self, source_name: str) -> Dict[str, Any]:
        """
        Get statistics and information about a specific document.
        
        Args:
            source_name: Name of the source document
            
        Returns:
            Dictionary with document statistics
        """
        # Check if the document exists in the collection
        results = self.collection.get(
            where={"source": source_name},
            include=["metadatas"]
        )
        
        if not results or not results["ids"]:
            return {
                "exists": False,
                "chunk_count": 0,
                "topics": [],
                "pages": []
            }
        
        # Get statistics about the document
        chunk_count = len(results["ids"])
        topics = set()
        pages = set()
        
        for metadata in results["metadatas"]:
            # Extract topics
            if "topics" in metadata:
                topic_data = metadata["topics"]
                if isinstance(topic_data, list):
                    topics.update(topic_data)
                elif isinstance(topic_data, str):
                    topics.update([t.strip() for t in topic_data.split(',')])
            
            # Extract page information
            if "page_number" in metadata:
                pages.add(metadata["page_number"])
        
        return {
            "exists": True,
            "chunk_count": chunk_count,
            "topics": sorted(list(topics)),
            "pages": sorted(list(pages)),
            "page_count": len(pages)
        }