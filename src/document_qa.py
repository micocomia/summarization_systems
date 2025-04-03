# document_qa.py
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import os
import requests
import json
import re

logger = logging.getLogger(__name__)

class DocumentQA:
    """
    Question answering system for documents using RAG with conversation context.
    """
    def __init__(self, retrieval_system, use_ollama: bool = True):
        """
        Initialize the question answering system.
        
        Args:
            retrieval_system: RetrievalSystem for finding relevant document chunks
            use_ollama: Whether to use local Ollama LLM
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

    def answer_question_with_context(self, question: str, conversation_history: List[Dict[str, str]], 
                            max_context_items: int = 100) -> Dict[str, Any]:
        """
        Answer a question based on document content and previous conversation context.
        
        Args:
            question: The current question to answer
            conversation_history: List of previous Q&A pairs [{"question": q, "answer": a}, ...]
            max_context_items: Maximum number of context items to use
                
        Returns:
            Dictionary with answer and metadata
        """
        # Check if this is a topic question
        if self._is_topic_question(question):
            return self._answer_topic_question(question)

        # Check if this might be a follow-up question
        is_followup = self._detect_followup_question(question)
        
        # If this seems like a follow-up and we have conversation history
        if is_followup and conversation_history:
            # Include the most recent conversation in our retrieval query
            # This helps find more relevant context for the follow-up
            last_exchange = conversation_history[-1]
            augmented_query = f"{last_exchange['question']} {last_exchange['answer']} {question}"
            
            # Retrieve relevant contexts for the augmented query
            contexts = self.retrieval_system.retrieve(
                query=augmented_query,
                top_k=max_context_items
            )
        else:
            # Process as a standalone question
            contexts = self.retrieval_system.retrieve(
                query=question,
                top_k=max_context_items
            )
        
        if not contexts:
            return {
                "answer": "I couldn't find any relevant information to answer that question. Could you try rephrasing or asking something else?",
                "confidence": 0.0,
                "sources": []
            }
        
        # Create a mapping of contexts to their source information for citation
        context_sources = {}
        sources = []
        
        for i, ctx in enumerate(contexts):
            if "metadata" in ctx and "source" in ctx["metadata"]:
                source = ctx["metadata"]["source"]
                page = ctx["metadata"].get("page_number", "")
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
        
        # Prepare contexts with citation markers for the prompt
        context_text = ""
        for i, ctx in enumerate(contexts):
            # Get citation marker
            citation = ""
            if i in context_sources:
                citation = f" [{context_sources[i]['index']}]"
            
            # Preprocess the content to handle existing citations
            processed_content = self._preprocess_context_citations(ctx['content'])
            
            # Add to processed context text
            context_text += f"Context {i+1}{citation}:\n{processed_content}\n\n"
        
        # Generate the answer
        answer = ""
        confidence = 0.0
        
        # Try Ollama first if available
        if self.ollama_available and self.use_ollama:
            try:
                # Include conversation history and citation instructions in the prompt
                answer_data = self._generate_with_ollama_and_citations(
                    question, 
                    context_text, 
                    conversation_history if is_followup else [],
                    sources
                )
                answer = answer_data["answer"]
                confidence = answer_data["confidence"]
            except Exception as e:
                logger.warning(f"Ollama QA failed: {e}")
                answer = ""
        
        # Fallback if needed
        if not answer:
            answer_data = self._generate_simple_answer_with_citations(question, context_text, sources)
            answer = answer_data["answer"]
            confidence = answer_data["confidence"]
        
        # After generating the answer, validate and clean citations
        cleaned_answer, used_citation_indices = self._validate_citations(answer, sources)
        
        # Add a sources section at the end, but only for citations that were actually used
        sources_text = ""
        if used_citation_indices:  # Only add sources section if citations were used
            sources_text = "\n\nSources:"
            for source in sources:
                # Only include sources that were actually cited in the answer
                if source['index'] in used_citation_indices:
                    sources_text += f"\n\n[{source['index']}] {source['title']}"
                    if source['page']:
                        sources_text += f", page {source['page']}"
                    sources_text += "\n"
        
        # Filter sources to only include those that were used
        used_sources = [s for s in sources if s['index'] in used_citation_indices]
        
        # Return with metadata
        result = {
            "answer": cleaned_answer,
            "confidence": confidence,
            "sources": used_sources,
            "answer_with_citations": cleaned_answer + sources_text if sources_text else cleaned_answer
        }
        
        return result
        
    def _detect_followup_question(self, question: str) -> bool:
        """
        Detect if a question is likely a follow-up to a previous question.
        
        Args:
            question: The question to analyze
            
        Returns:
            Boolean indicating if this is likely a follow-up
        """
        question_lower = question.lower().strip()
        
        # Common follow-up patterns
        followup_patterns = [
            r"^(what|who|where|when|why|how) (is|are|was|were) (it|that|they|those|this|these)",
            r"^(can|could) you (explain|elaborate|clarify|tell me more)",
            r"^(please|) (explain|elaborate|clarify)",
            r"^(tell|show) me more",
            r"^(what|who|where|when|why|how) (about|else)",
            r"^(and|but|so) (what|who|where|when|why|how)",
            r"^(what|who|where|when|why|how) (if|then)",
            r"^(is|are|was|were|do|does|did|can|could|would|should) (it|that|they|this|these)",
            r"^why$",
            r"^how$",
            r"^(really|actually|seriously)(\?|)",
            r"^(go on|proceed|continue)"
        ]
        
        # Check for pronoun references without specific context
        # These typically indicate follow-up questions
        pronoun_patterns = [
            r"\b(it|this|that|they|them|those|these)\b",
        ]
        
        # Check for follow-up patterns
        for pattern in followup_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # If the question is very short and contains pronouns, likely a follow-up
        if len(question_lower.split()) < 5:
            for pattern in pronoun_patterns:
                if re.search(pattern, question_lower):
                    return True
        
        return False
    
    def _generate_with_ollama_and_citations(self, question: str, context: str, 
                                         conversation_history: List[Dict[str, str]],
                                         sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer using Ollama with conversation history and embedded citations."""
        import requests
        
        # Create prompt for QA with conversation history and citation instructions
        prompt = self._create_qa_prompt_with_citations(question, context, conversation_history, sources)
        
        # Prepare the request to Ollama
        data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.000001,  # Lower temperature for factual responses
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 1024
            }
        }
        
        # Call the Ollama API
        response = requests.post(self.ollama_endpoint, json=data)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            
            # Parse the response for structured output
            answer = response_text
            confidence = 0.8  # Default confidence
            
            # Try to extract confidence if present
            confidence_match = re.search(r'Confidence: (0\.\d+)', response_text)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    # Remove the confidence line from the answer
                    answer = re.sub(r'Confidence: 0\.\d+', '', answer).strip()
                except ValueError:
                    pass
            
            return {
                "answer": answer,
                "confidence": confidence
            }
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return {
                "answer": "I encountered an error while trying to answer your question.",
                "confidence": 0.0
            }
    
    def _create_qa_prompt_with_citations(self, question: str, context: str, 
                                      conversation_history: List[Dict[str, str]],
                                      sources: List[Dict[str, Any]]) -> str:
        """Create a prompt for QA with conversation history and enhanced citation instructions."""
        
        # Format conversation history if available
        conversation_context = ""
        if conversation_history:
            conversation_context = "Previous conversation:\n"
            for i, exchange in enumerate(conversation_history[-3:]):  # Use last 3 exchanges at most
                conversation_context += f"User: {exchange['question']}\n"
                conversation_context += f"Assistant: {exchange['answer']}\n\n"
        
        # Create enhanced citation instructions if sources exist
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
            6. Do not reference context numbers in your answer (e.g., do not write "as mentioned in Context 3").
            7. Include at least one citation in your answer, preferably multiple citations for different pieces of information.
            8. The citations in your answer MUST MATCH the source numbers provided in the context.
            """
        
        return f"""
        Answer the following question based on the provided context and conversation history.
        
        {conversation_context}
        
        Context:
        {context}
        
        Current Question: {question}

        IMPORTANT REQUIREMENTS:
        1. Use ONLY information from the provided context to answer the question
        2. For acronyms, pull information from the provided context to define the acronym. Otherwise, just leave the acronym as is.
        3. If the context doesn't contain the answer, say "I don't have enough information to answer that question"
        4. Take into account the conversation history for context when answering follow-up questions
        5. Keep answers concise and to the point
        6. Don't make up information or use prior knowledge
        7. Format your answer as readable text, not as a continuation of the prompt
        8. If inferring from the provided conversation history, do not mention you are inferring from the conversation history
        9. DO NOT ADD a references section after the answer as this would be handled separately
        10. CRITICAL: DO NOT ADD ANY NOTES, DISCLAIMERS, OR COMMENTS ABOUT FOLLOWING THE REQUIREMENTS. Your response should ONLY contain the direct answer to the question, nothing else.
        {citation_instructions}
        
        Answer:

        FINAL CRITICAL INSTRUCTION: Do NOT add any statements about having followed the requirements or about what you've done. The output should contain ONLY your direct answer to the question.
        """
    
    def _generate_simple_answer_with_citations(self, question: str, context: str, 
                                          sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a simple answer with embedded citations when LLM is not available."""
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
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk["content"]) if len(s.strip()) > 10]
            for sentence in sentences:
                all_sentences.append({
                    "text": sentence,
                    "citation": chunk["citation"]
                })
        
        if not all_sentences:
            return {
                "answer": "I couldn't find relevant information to answer that question.",
                "confidence": 0.0
            }
        
        # Basic keyword matching
        question_words = set(question.lower().split())
        sentence_scores = {}
        
        # Score each sentence based on word overlap with the question
        for i, sentence_data in enumerate(all_sentences):
            sentence = sentence_data["text"]
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            sentence_scores[i] = overlap
        
        # Get highest scoring sentences
        sorted_indices = sorted(sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True)
        
        # Take top 2-3 sentences
        top_count = min(3, len(sorted_indices))
        answer_sentences = []
        
        for i in range(top_count):
            if i < len(sorted_indices) and sentence_scores[sorted_indices[i]] > 0:
                sent_idx = sorted_indices[i]
                sentence = all_sentences[sent_idx]["text"]
                citation = all_sentences[sent_idx]["citation"]
                
                # Add citation if available
                if citation:
                    answer_sentences.append(f"{sentence} [{citation}]")
                else:
                    answer_sentences.append(sentence)
        
        if not answer_sentences:
            return {
                "answer": "I don't have enough information to answer that question based on the provided context.",
                "confidence": 0.2
            }
        
        # Combine sentences into an answer
        answer = " ".join(answer_sentences)
        
        # Calculate confidence based on keyword matching
        max_possible_overlap = len(question_words)
        highest_overlap = sentence_scores[sorted_indices[0]] if sorted_indices else 0
        confidence = min(0.9, highest_overlap / max_possible_overlap) if max_possible_overlap > 0 else 0.3
        
        return {
            "answer": answer,
            "confidence": confidence
        }
    
    def _is_topic_question(self, question: str) -> bool:
        """Detect if a question is asking about document topics."""
        question_lower = question.lower()
        topic_patterns = [
            r"what (are|were|is) the topics",
            r"what topics",
            r"(list|show|tell me) (the|) topics",
            r"what (is|are) this document about",
            r"main (topics|subjects|themes)"
        ]
        
        return any(re.search(pattern, question_lower) for pattern in topic_patterns)

    def _answer_topic_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about document topics."""
        try:
            # Get topics from retrieval system
            topics = self.retrieval_system.get_available_topics()
            
            if topics:
                answer = f"I found these main topics in your documents: {', '.join(topics)}"
            else:
                answer = "I couldn't identify any clear topics in your documents."
                
            return {
                "answer": answer,
                "confidence": 0.9,
                "sources": []
            }
        except Exception as e:
            logger.error(f"Error answering topic question: {e}")
            return {
                "answer": "I encountered an error while trying to identify topics in your documents.",
                "confidence": 0.0,
                "sources": []
            }