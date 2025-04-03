# abstractive_summarizer.py
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import requests
import json
import os
import re
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

    def _preprocess_context_citations(self, context_text: str) -> str:
        """
        Preprocess the context to handle existing citation patterns.
        This prevents the model from copying existing citations.
        
        Args:
            context_text: The raw context text with potential existing citations
            
        Returns:
            Processed context with existing citations transformed
        """
        # Pattern to detect existing citations like [21], [22], etc.
        existing_citation_pattern = r'\[(\d+)\]'
        
        # Function to replace citation with a descriptive text
        def replace_citation(match):
            citation_num = match.group(1)
            return f"(document reference {citation_num})"
        
        # Replace existing citations with descriptive text
        processed_context = re.sub(existing_citation_pattern, replace_citation, context_text)
        
        return processed_context

    def _validate_citations(self, text: str, sources: List[Dict[str, Any]]) -> Tuple[str, Set[int]]:
        """
        Validate citations in the text and remove any that don't correspond to actual sources.
        
        Args:
            text: The text containing potential citations
            sources: List of source dictionaries with 'index' keys
            
        Returns:
            Tuple of (cleaned_text, set of valid citation indices used)
        """
        if not sources:
            # If no sources, remove all citation-like patterns
            cleaned_text = re.sub(r'\[\d+\]', '', text)
            return cleaned_text, set()
        
        # Get the valid source indices
        valid_indices = {source['index'] for source in sources}
        
        # Find all citation-like patterns
        citation_pattern = r'\[(\d+)\]'
        
        # Function to replace invalid citations in the re.sub function
        def replace_invalid_citation(match):
            citation_num = int(match.group(1))
            if citation_num in valid_indices:
                # Keep valid citations
                return match.group(0)
            else:
                # Remove invalid citations
                return ''
        
        # Replace invalid citations
        cleaned_text = re.sub(citation_pattern, replace_invalid_citation, text)
        
        # Find all remaining (valid) citations
        valid_citations = set()
        for match in re.finditer(citation_pattern, cleaned_text):
            try:
                valid_citations.add(int(match.group(1)))
            except ValueError:
                continue
                
        return cleaned_text, valid_citations

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
            "short": {"words": 50, "description": "concise"},
            "medium": {"words": 200, "description": "comprehensive"},
            "long": {"words": 500, "description": "detailed"}
        }
        
        target = length_targets.get(length, length_targets["medium"])
        
        # Retrieve relevant contexts for summarization
        # For summarization, we want broad coverage of the document(s)
        num_contexts = 50  # Retrieve more contexts for full coverage
        
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
                top_k=10
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

        # Limit to 1000 contexts
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
        
        # Create a mapping of contexts to their source information for citation
        context_sources = {}
        sources = []

        for i, ctx in enumerate(contexts):
            if "metadata" in ctx and "source" in ctx["metadata"]:
                source = ctx["metadata"]["source"]
                page = ctx["metadata"].get("page_number", "")  # Get page_number from metadata
                source_info = {"title": source, "page": page, "index": len(sources) + 1}
                
                # Find unique sources
                source_exists = False
                for existing_source in sources:
                    if existing_source["title"] == source and existing_source["page"] == page:
                        source_exists = True
                        source_info = existing_source  # Use existing source reference
                        break
                
                if not source_exists:
                    sources.append(source_info)
                
                # Map context to its source for citation
                context_sources[i] = source_info
        

        # Extract metadata about the sources and topics
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
        
        context_text = ""
        for i, ctx in enumerate(contexts):
            # Get citation marker
            citation = ""
            if i in context_sources:
                citation = f" [{context_sources[i]['index']}]"
            
            # Preprocess the content to handle existing citations
            processed_content = self._preprocess_context_citations(ctx['content'])
            
            # Add to processed context text
            context_text += f"{processed_content}{citation}\n\n"
        
        # Generate the summary
        summary = ""
        
        # Try Ollama first if available
        if self.ollama_available and self.use_ollama:
            try:
                summary = self._generate_with_ollama(context_text, target, sources)
            except Exception as e:
                logger.warning(f"Ollama summarization failed: {e}")
                summary = ""
        
        # Fallback for simple summarization if needed
        if not summary:
            summary = self._generate_simple_summary_with_citations(context_text, target, sources)
        
        # After generating the summary, validate and clean citations
        cleaned_summary, used_citation_indices = self._validate_citations(summary, sources)
        
        # Add a sources section at the end, but only for citations that were actually used
        sources_text = ""
        if used_citation_indices:  # Only add sources section if citations were used
            sources_text = "\n\nSources:"
            for source in sources:
                # Only include sources that were actually cited in the answer/summary
                if source['index'] in used_citation_indices:
                    sources_text += f"\n\n[{source['index']}] {source['title']}"
                    if source['page']:  # Simply check if page has a truthy value
                        sources_text += f", page {source['page']}"
                    sources_text += "\n"
        
        # Filter sources to only include those that were used
        used_sources = [s for s in sources if s['index'] in used_citation_indices]
        
        # Return with metadata
        return {
            "summary": cleaned_summary,
            "summary_with_sources": cleaned_summary + sources_text if sources_text else cleaned_summary,
            "metadata": {
                "length": length,
                "word_count": len(cleaned_summary.split()),
                "focus_topics": list(topics) if focus_topics is None else focus_topics,
                "source_documents": list(source_documents),
                "context_count": len(contexts),
                "sources": used_sources,
            }
        }
        
    def _generate_with_ollama(self, context: str, target: Dict[str, Any], sources: List[Dict[str, Any]]) -> str:
        """Generate a summary using Ollama with embedded citations."""
        import requests
        
        # Create the prompt for abstractive summarization with citations
        prompt = self._create_summary_prompt_with_citations(context, target, sources)
        
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
    
    def _create_summary_prompt_with_citations(self, context: str, target: Dict[str, Any], sources: List[Dict[str, Any]]) -> str:
        """Create a prompt for summary generation with enhanced citation instructions."""
        # Create citation instructions if sources exist
        citation_instructions = ""
        if sources:
            # List out all valid citation numbers to make them explicit
            valid_citations = ", ".join([f"[{source['index']}]" for source in sources])
            
            citation_instructions = f"""
            CITATION REQUIREMENTS (VERY IMPORTANT):
            1. ONLY use these specific citation numbers: {valid_citations}
            2. When citing, use EXACTLY the citation numbers from the Context sections.
            3. Place citations [X] immediately after the specific information they support.
            4. Do not group citations at the end of sentences or paragraphs.
            5. Each piece of information should be cited individually.
            6. Do not reference context numbers in your summary (e.g., do not write "as mentioned in Context 3").
            7. Include at least one citation in each key point and multiple citations in the main summary.
            8. The citations in your summary MUST MATCH the source numbers provided in the context.
            """
        
        return f"""
        You are a professional summarizer. Create a {target["description"]} and cohesive abstractive summary of the following content.

        CONTENT TO SUMMARIZE:
        {context}

        IMPORTANT REQUIREMENTS FOR THE SUMMARY:
        1. DO NOT GENERATE information that cannot be found in the content.
        2. When defining acronyms, define using the provided content. Else, keep the acronym as is.
        3. Capture the main ideas, conclusions, and important details
        4. Provide 3 key points about the content to summarize. Each point should not exceed 20 words.
        5. The MAIN SUMMARY should be written in a cohesive, flowing style that is composed of paragraph(s).
        6. STRICTLY follow the format in Summary Output Requirements. Key Points are followed by the Main Summary.
        7. ONLY INCLUDE information present in the original text.
        9. Avoid repetition and focus on the most important information
        10. Use clear and concise language
        12. CRITICAL: DO NOT ADD ANY NOTES, DISCLAIMERS, OR COMMENTS ABOUT FOLLOWING THE REQUIREMENTS. Your response should ONLY contain the Key Points and Summary sections, nothing else.
        13. If there is not enough context to fulfill the length requirements, do not pad the summary with unnecessary content.
        14. DO NOT ADD a references section after the summary as this would be handled separately.
        {citation_instructions}
        
        SUMMARY OUTPUT REQUIREMENTS:
        Key Points:
        1. Key Point 1 Here [Include citations]
        2. Key Point 2 Here [Include citations]
        3. Key Point 3 Here [Include citations]

        Summary:
        MAIN SUMMARY HERE [With inline citations]

        FINAL CRITICAL INSTRUCTION: Do NOT add any statements about having followed the requirements. The output should contain ONLY the Key Points and Summary as specified above.
        """
    
    def _generate_simple_summary_with_citations(self, context: str, target: Dict[str, Any], sources: List[Dict[str, Any]]) -> str:
        """Generate a simple extractive summary with citations when LLM is not available."""
        import re
        
        # Extract contexts with their citation markers
        context_chunks = []
        citation_pattern = r"Context (\d+)(?:\s+\[(\d+)\])?:\n(.*?)(?=\n\nContext|\Z)"
        matches = re.finditer(citation_pattern, context, re.DOTALL)
        
        for match in matches:
            context_id = int(match.group(1))
            citation = match.group(2)
            content = match.group(3).strip()
            
            context_chunks.append({
                "id": context_id,
                "citation": citation,
                "content": content
            })
        
        # Extract sentences from each context with citation info
        all_sentences = []
        for chunk in context_chunks:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk["content"]) if len(s.strip()) > 20]
            for sentence in sentences:
                all_sentences.append({
                    "text": sentence,
                    "citation": chunk["citation"]
                })
        
        if not all_sentences:
            return "No content available to summarize."
        
        # Score sentences
        sentence_scores = {}
        
        # Important words to look for when scoring sentences
        important_words = ["important", "significant", "key", "main", "critical", 
                           "essential", "primary", "fundamental", "crucial", "major",
                           "conclusion", "result", "finding", "determine", "show", 
                           "demonstrate", "reveal", "highlight", "indicate", "prove"]
        
        for i, sentence_data in enumerate(all_sentences):
            sentence = sentence_data["text"]
            # Position score - earlier sentences often more important
            position_score = 1.0 / (1 + 0.1 * i)
            
            # Importance score - based on presence of key terms
            importance_score = 0
            for word in important_words:
                if word in sentence.lower():
                    importance_score += 0.2
                    
            # Combine scores
            sentence_scores[i] = position_score + importance_score
        
        # Get top sentences based on scores
        sorted_indices = sorted(sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True)
        
        # Calculate number of sentences to include based on target length
        target_sentences = min(max(3, target["words"] // 25), len(sorted_indices))
        
        # Get the top-scoring sentences
        top_indices = sorted_indices[:target_sentences]
        
        # Sort indices by their original order to maintain document flow
        top_indices.sort()
        
        # Build the summary with citations
        summary_parts = []
        for idx in top_indices:
            sentence = all_sentences[idx]["text"]
            citation = all_sentences[idx]["citation"]
            
            if citation:
                summary_parts.append(f"{sentence} [{citation}]")
            else:
                summary_parts.append(sentence)
                
        # Combine into a summary
        summary = " ".join(summary_parts)
        
        # Add key points section
        if len(top_indices) >= 3:
            key_points = []
            # Use the top 3 sentences as key points
            for i in range(min(3, len(top_indices))):
                idx = sorted_indices[i]
                sentence = all_sentences[idx]["text"]
                citation = all_sentences[idx]["citation"]
                
                if citation:
                    key_points.append(f"{sentence} [{citation}]")
                else:
                    key_points.append(sentence)
                    
            key_points_text = "Key Points:\n"
            for i, point in enumerate(key_points):
                key_points_text += f"{i+1}. {point}\n"
                
            summary = f"{key_points_text}\nSummary:\n{summary}"
            
        return summary