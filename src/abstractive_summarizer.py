# abstractive_summarizer.py
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import requests
import json
import os

logger = logging.getLogger(__name__)

class AbstractiveSummarizer:
    """
    Generates abstractive summaries of documents using LLM.
    """
    def __init__(self, retrieval_system, use_ollama: bool = True):
        """
        Initialize the abstractive summarizer.
        
        Args:
            retrieval_system: RetrievalSystem for finding relevant document chunks
            use_ollama: Whether to use local Ollama LLM (True) or API (False)
        """
        self.retrieval_system = retrieval_system
        self.use_ollama = use_ollama
        
        # Initialize Ollama if requested
        self.ollama_available = False
        if self.use_ollama:
            self._setup_ollama_llm()
    
    def _setup_ollama_llm(self):
        """Set up the Ollama LLM integration."""
        try:
            self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
            self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            
            # Test connection to Ollama
            response = requests.get(self.ollama_endpoint.replace("/generate", "/tags"))
            if response.status_code == 200:
                self.ollama_available = True
                available_models = response.json().get("models", [])
                model_names = [model.get("name") for model in available_models]
                
                # If our preferred model isn't available, choose one that is
                if self.ollama_model not in model_names and model_names:
                    self.ollama_model = model_names[0]
                    
                logger.info(f"Ollama integration available with model: {self.ollama_model}")
            else:
                self.ollama_available = False
                logger.warning("Ollama server responded but with an error")
        except Exception as e:
            self.ollama_available = False
            logger.warning(f"Ollama integration not available: {e}")

    def generate_summary(self, length: str = "medium", focus_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate an abstractive summary of the documents in the vector store.
        
        Args:
            length: Summary length - 'short', 'medium', or 'long'
            focus_topics: Optional list of topics to focus the summary on
            
        Returns:
            Dictionary with summary text and metadata
        """
        # Map length to token/word targets
        length_targets = {
            "short": {"words": 100, "description": "concise"},
            "medium": {"words": 250, "description": "comprehensive"},
            "long": {"words": 500, "description": "detailed"}
        }
        
        target = length_targets.get(length, length_targets["medium"])
        
        # Retrieve relevant contexts for summarization
        # For summarization, we want broad coverage of the document(s)
        num_contexts = 1000  # Retrieve more contexts for full coverage
        
        # Build diverse retrieval queries to get good coverage
        queries = [
            "important concepts and key points",
            "main findings and conclusions",
            "critical information and highlights",
            "summary of main arguments",
            "essential details and examples"
        ]
        
        # If focus topics are provided, add topic-specific queries
        if focus_topics:
            for topic in focus_topics:
                queries.append(f"key information about {topic}")
        
        # Retrieve contexts for each query
        all_contexts = []
        for query in queries:
            contexts = self.retrieval_system.retrieve(
                query=query,
                top_k=5  # Get top 5 results per query
            )
            all_contexts.extend(contexts)
        
        # Remove duplicates by keeping highest scoring instances
        unique_contexts = {}
        for context in all_contexts:
            context_id = context["id"]
            if context_id not in unique_contexts or context["similarity"] > unique_contexts[context_id]["similarity"]:
                unique_contexts[context_id] = context
        
        # Convert back to list and sort by score
        contexts = sorted(unique_contexts.values(), key=lambda x: x["similarity"], reverse=True)
        
        # Limit to the top contexts to avoid exceeding context limits
        contexts = contexts[:num_contexts]
        
        if not contexts:
            return {
                "summary": "Unable to generate summary. No document content found.",
                "metadata": {
                    "length": length,
                    "focus_topics": focus_topics,
                    "context_count": 0
                }
            }
        
        # Extract metadata about the sources
        source_documents = set()
        topics = set()
        for ctx in contexts:
            if "metadata" in ctx:
                if "source" in ctx["metadata"]:
                    source_documents.add(ctx["metadata"]["source"])
                if "topics" in ctx["metadata"]:
                    if isinstance(ctx["metadata"]["topics"], list):
                        topics.update(ctx["metadata"]["topics"])
                    elif isinstance(ctx["metadata"]["topics"], str):
                        topics.update([t.strip() for t in ctx["metadata"]["topics"].split(',')])
        
        # Combine contexts for the prompt
        context_text = "\n\n".join([ctx["content"] for ctx in contexts])
        
        # Generate the summary
        summary = ""
        
        # Try Ollama first if available
        if self.ollama_available and self.use_ollama:
            try:
                summary = self._generate_with_ollama(context_text, target)
            except Exception as e:
                logger.warning(f"Ollama summarization failed: {e}")
                summary = ""
        
        # Fallback for simple summarization if needed
        if not summary:
            summary = self._generate_simple_summary(context_text, target)
            
        # Return with metadata
        return {
            "summary": summary,
            "metadata": {
                "length": length,
                "word_count": len(summary.split()),
                "focus_topics": list(topics) if focus_topics is None else focus_topics,
                "source_documents": list(source_documents),
                "context_count": len(contexts)
            }
        }
    
    def _generate_with_ollama(self, context: str, target: Dict[str, Any]) -> str:
        """Generate a summary using Ollama."""
        import requests
        
        # Create the prompt for abstractive summarization
        prompt = self._create_summary_prompt(context, target)
        
        # Prepare the request to Ollama
        data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.000001,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 2048
            }
        }
        
        # Call the Ollama API
        response = requests.post(self.ollama_endpoint, json=data)
        
        if response.status_code == 200:
            result = response.json()
            summary_text = result.get("response", "")
            
            # Clean up the summary
            # Remove any "Summary:" prefix if present
            summary_text = summary_text.replace("Summary:", "").strip()
            
            return summary_text
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return ""
    
    def _create_summary_prompt(self, context: str, target: Dict[str, Any]) -> str:
        """Create a prompt for summary generation."""
        return f"""
        You are a professional summarizer. Create a {target["description"]} and cohesive abstractive summary of the following content.

        CONTENT TO SUMMARIZE:
        {context}

        IMPORTANT REQUIREMENTS FOR THE SUMMARY:
        1. DO NOT GENERATE information that cannot be found in the content.
        2. When defining acronyms, define using the provided content. Else, keep the acronym as is.
        3. Capture the main ideas, conclusions, and important details
        4. Provide a brief overview of the content using 1 to 2 short sentences.
        5. Provide 3 key points regarding the content.
        6. The MAIN SUMMARY should be written in a cohesive, flowing style that is composed of paragraph(s).
        7. ONLY INCLUDE information present in the original text.
        8. DO NOT ADD any information that is irrelevant to the content.
        9. Avoid repetition and focus on the most important information
        10. Use clear and concise language
        11. Be able to stand alone as a document
        12. DO NOT add introductory or closing text. STRICTLY follow the format in Summary Output Requirements.
        
        SUMMARY OUTPUT REQUIREMENTS:
        Brief Overview:
        "BRIEF OVERVIEW HERE"

        Key Points:
        1. "Key Point 1 Here"
        2. "Key Point 2 here"
        3. "Key Point 3 Here"

        Summary:
        "MAIN SUMMARY HERE"
        """
    
    def _generate_simple_summary(self, context: str, target: Dict[str, Any]) -> str:
        """Generate a simple extractive summary when LLM is not available."""
        # Split into sentences
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 20]
        
        if not sentences:
            return "No content available to summarize."
        
        # Extract important sentences
        sentence_scores = {}
        
        # Score based on position (earlier sentences often more important)
        for i, sentence in enumerate(sentences):
            # Decay score as we progress through the document
            position_score = 1.0 / (1 + 0.1 * i)
            sentence_scores[sentence] = position_score
        
        # Score based on presence of important words
        important_words = ["important", "significant", "key", "main", "critical", 
                           "essential", "primary", "fundamental", "crucial", "major",
                           "conclusion", "result", "finding", "determine", "show", 
                           "demonstrate", "reveal", "highlight", "indicate", "prove"]
        
        for sentence in sentences:
            for word in important_words:
                if word in sentence.lower():
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + 0.2
        
        # Sort sentences by score
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top sentences based on target length
        target_sentences = min(max(3, target["words"] // 25), len(sorted_sentences))
        top_sentences = [sentence for sentence, score in sorted_sentences[:target_sentences]]
        
        # Re-order sentences to maintain original flow
        original_order_sentences = [s for s in sentences if s in top_sentences]
        
        # Combine into summary
        summary = " ".join(original_order_sentences)
        
        return summary