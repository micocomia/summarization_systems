# document_qa.py
import logging
from typing import List, Dict, Any, Optional
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
        
        # Extract source information for citation
        sources = []
        for ctx in contexts:
            if "metadata" in ctx and "source" in ctx["metadata"]:
                source = ctx["metadata"]["source"]
                page = ctx["metadata"].get("page_number", "")
                source_info = {"title": source, "page": page}
                if source_info not in sources:
                    sources.append(source_info)
        
        # Combine contexts for the prompt
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx['content']}" for i, ctx in enumerate(contexts)])
        
        # Generate the answer
        answer = ""
        confidence = 0.0
        
        # Try Ollama first if available
        if self.ollama_available and self.use_ollama:
            try:
                # Include conversation history in the prompt
                answer_data = self._generate_with_ollama_and_history(
                    question, 
                    context_text, 
                    conversation_history if is_followup else []
                )
                answer = answer_data["answer"]
                confidence = answer_data["confidence"]
            except Exception as e:
                logger.warning(f"Ollama QA failed: {e}")
                answer = ""
        
        # Fallback if needed
        if not answer:
            answer_data = self._generate_simple_answer(question, context_text)
            answer = answer_data["answer"]
            confidence = answer_data["confidence"]
            
        # Return with metadata
        result = {
            "answer": answer,
            "confidence": confidence,
            "sources": sources
        }
        
        # Add citations to the answer if sources are available
        if sources:
            sources_text = "\n\n\nSources:\n\n"
            for i, source in enumerate(sources):
                sources_text += f"\n\n[{i+1}] {source['title']}"
                if source['page']:
                    sources_text += f", page {source['page']}"
                    
            result["answer_with_citations"] = answer + sources_text
        else:
            result["answer_with_citations"] = answer
            
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
    
    def _generate_with_ollama_and_history(self, question: str, context: str, 
                                         conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate an answer using Ollama with conversation history."""
        import requests
        
        # Create prompt for QA with conversation history
        prompt = self._create_qa_prompt_with_history(question, context, conversation_history)
        
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
    
    def _create_qa_prompt_with_history(self, question: str, context: str, 
                                      conversation_history: List[Dict[str, str]]) -> str:
        """Create a prompt for QA with conversation history."""
        
        # Format conversation history if available
        conversation_context = ""
        if conversation_history:
            conversation_context = "Previous conversation:\n"
            for i, exchange in enumerate(conversation_history[-3:]):  # Use last 3 exchanges at most
                conversation_context += f"User: {exchange['question']}\n"
                conversation_context += f"Assistant: {exchange['answer']}\n\n"
        
        return f"""
        Answer the following question based on the provided context and conversation history.
        
        {conversation_context}
        
        Context:
        {context}
        
        Current Question: {question}

        IMPORTANT REQUIREMENTS:
        1. Use only information from the provided context to answer the question
        2. For acronyms, pull information from the provided context to define the acronym. Otherwise, just leave the acronym as is.
        3. If the context doesn't contain the answer, say "I don't have enough information to answer that question"
        4. Take into account the conversation history for context when answering follow-up questions
        5. Keep answers concise and to the point
        6. Don't make up information or use prior knowledge
        7. Format your answer as readable text, not as a continuation of the prompt
        8. Do not add any introductory or closing text that is not relevant to the question
        9. If inferring from the provided conversation history, do not mention you are inferring from the conversation history
        
        Answer:
        """
    
    def _generate_simple_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Generate a simple answer when LLM is not available."""
        import re
        import random
        
        # Extract sentences from the context
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 10]
        
        if not sentences:
            return {
                "answer": "I couldn't find relevant information to answer that question.",
                "confidence": 0.0
            }
        
        # Very basic keyword matching
        question_words = set(question.lower().split())
        sentence_scores = {}
        
        # Score each sentence based on word overlap with the question
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            sentence_scores[sentence] = overlap
        
        # Get highest scoring sentences
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 2-3 sentences
        top_count = min(3, len(sorted_sentences))
        answer_sentences = [s for s, _ in sorted_sentences[:top_count] if sentence_scores[s] > 0]
        
        if not answer_sentences:
            return {
                "answer": "I don't have enough information to answer that question based on the provided context.",
                "confidence": 0.2
            }
        
        # Combine sentences into an answer
        answer = " ".join(answer_sentences)
        
        # Calculate confidence based on keyword matching
        max_possible_overlap = len(question_words)
        highest_overlap = sentence_scores[sorted_sentences[0][0]] if sorted_sentences else 0
        confidence = min(0.9, highest_overlap / max_possible_overlap) if max_possible_overlap > 0 else 0.3
        
        return {
            "answer": answer,
            "confidence": confidence
        }