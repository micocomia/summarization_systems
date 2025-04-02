# intent_handler.py
import logging
import re
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SessionState:
    """Session state for conversation management."""
    def __init__(self):
        self.documents_loaded = False
        self.settings = {
            "summarization_type": "abstractive",  # abstractive or extractive
            "summary_length": "medium",  # short, medium, long
            "focus_topics": [],  # list of topics to focus on
        }
        self.last_summary = None
        self.last_qa_answer = None
        self.active_documents = []  # list of document names that are currently loaded
        
        # Add conversation history tracking
        self.qa_conversation_history = []  # List of {"question": q, "answer": a} dicts

class IntentHandlerManager:
    """
    Manages handling of different user intents in the conversation.
    """
    def __init__(self, 
                 document_processor=None, 
                 retrieval_system=None,
                 abstractive_summarizer=None,
                 news_extractive_summarizer_class=None,
                 document_qa=None):
        """
        Initialize the intent handler.
        
        Args:
            document_processor: DocumentProcessor instance
            retrieval_system: RetrievalSystem instance
            abstractive_summarizer: AbstractiveSummarizer instance
            document_qa: DocumentQA instance
        """
        self.document_processor = document_processor
        self.retrieval_system = retrieval_system
        self.abstractive_summarizer = abstractive_summarizer
        self.extractive_news_summarizer = news_extractive_summarizer_class
        self.document_qa = document_qa
        
        # Initialize session state
        self.session = SessionState()
        
    def handle_intent(self, intent_type: str, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a user intent based on its type.
        
        Args:
            intent_type: The type of intent
            intent_data: Intent data from Dialogflow
            
        Returns:
            Response data for the user
        """
        logger.info(f"Handling intent: {intent_type}")
        
        # Default response in case no handler is found
        response = {
            "text": "I'm not sure how to handle that request."
        }
        
        # Check if this is a QA_CQ intent
        if intent_type.startswith("QA_CQ"):
            return self._handle_dialogflow_qa(intent_data)
            
        # Map intent types to handler methods
        intent_handlers = {
            "SummaryTask_Abstractive": self._handle_abstractive_summarization,
            "Extractive_summarizer_news_articles": self._handle_extractive_summarization_news,
            "document_qa": self._handle_document_qa,
            "summarization_info": self._handle_summarization_info,
            "show_settings": self._handle_show_settings,
            "upload_document": self._handle_upload_document,
            "list_documents": self._handle_list_documents,
            "fallback": self._handle_fallback
        }
        
        # Call the appropriate handler
        if intent_type in intent_handlers:
            response = intent_handlers[intent_type](intent_data)
        else:
            # Default to document QA for unknown intents if documents are loaded
            if self.session.documents_loaded:
                response = self._handle_document_qa(intent_data)
            else:
                response = self._handle_fallback(intent_data)
                
        return response
    
    def _handle_abstractive_summarization(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle abstractive summarization intent."""
        if not self.session.documents_loaded:
            return {
                "text": "Please upload some documents first before asking for a summary."
            }
        
        # Extract parameters from the intent
        params = intent_data.get("parameters", {})
        
        # Extract the length parameter from Dialogflow
        # Check for 'length' parameter first (from Dialogflow context)
        summary_length = params.get("length", "")
        
        # Validate length or default to "medium"
        valid_lengths = ["short", "medium", "long"]
        if not summary_length or summary_length.lower() not in valid_lengths:
            summary_length = "medium"  # Default to medium when not specified
        else:
            summary_length = summary_length.lower()
        
        # Extract focus topics if present in the intent data
        focus_topics = params.get("focus_topics", [])
        if focus_topics and not isinstance(focus_topics, list):
            focus_topics = [focus_topics]
        
        # Generate abstractive summary
        try:
            summary_result = self.abstractive_summarizer.generate_summary(
                length=summary_length,
                focus_topics=focus_topics if focus_topics else None
            )
            
            # Build response with summary and metadata
            summary_text = summary_result["summary"]
            metadata = summary_result["metadata"]
            
            source_info = ""
            if "source_documents" in metadata and metadata["source_documents"]:
                source_docs = metadata["source_documents"]
                source_info = f"\n\n\nSource(s): {', '.join(source_docs)}"
            
            # Create response with length information
            response_text = f"{summary_text}{source_info}"
            
            return {
                "text": response_text,
                "summary": summary_result
            }
            
        except Exception as e:
            logger.error(f"Error generating abstractive summary: {str(e)}")
            return {
                "text": "I encountered an error while trying to generate a summary. Please try again or try with different settings."
            }
    
    def _handle_extractive_summarization_news(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the intent for extractive summarization of news articles or any text input.
        Works around Dialogflow's 256-character limit by handling long text inputs properly.
        """
        try:
            # Get the original raw text from the request before Dialogflow truncation
            # This should be passed from your main application logic
            raw_text = intent_data.get("raw_text", "")
        
            # Get whatever Dialogflow managed to capture (likely truncated)
            dialogflow_text = intent_data.get("text", "").strip()
            resolved_query = intent_data.get("resolved_query", "").strip()
        
            # Use the raw text if available, otherwise use whatever we got from Dialogflow
            content = raw_text if raw_text else (dialogflow_text if len(dialogflow_text) > len(resolved_query) else resolved_query)
        
            # Check if the content is empty
            if not content:
                return {"text": "Please provide the text you'd like me to summarize."}
        
            # If we hit the fallback intent due to text length, we should directly summarize
            # This handles the case where Dialogflow fails due to text length
            if intent_data.get("is_fallback", False) and len(content) > 250:
                logger.info(f"Detected long text input ({len(content)} chars), proceeding with summarization")
                text_to_summarize = content
            else:
            
                # Check for specific summarization patterns (for shorter texts that didn't trigger fallback)
                summarize_prefixes = ["summarize this:", "can you summarize this:", 
                                    "could you summarize this:", "please summarize this:", 
                                    "summarize the following:", "summary of this:"]
        
                summarize_suffixes = ["summarize this", "please summarize", "can you summarize this?",
                                    "summarize the above", "summarize this text", "create a summary"]
        
                # Extract text to summarize
                text_to_summarize = content
        
                # Check for prefix patterns like "summarize this: [text]"
                for prefix in summarize_prefixes:
                    if content.lower().startswith(prefix):
                        text_to_summarize = content[len(prefix):].strip()
                        break
                
                # Check for suffix patterns like "[text] summarize this"
                for suffix in summarize_suffixes:
                    if content.lower().endswith(suffix):
                        text_to_summarize = content[:-len(suffix)].strip()
                        break
        
            # Auto-detect if this is likely a summarization request but needs confirmation
            if not intent_data.get("is_fallback", False):
                is_likely_summarization_candidate = (
                    len(text_to_summarize) > 300 and 
                    "?" not in text_to_summarize and
                    len(text_to_summarize.split()) > 100
                )
            
                # If the text doesn't match specific patterns but seems like a summarization candidate,
                # confirm with the user
                if (not any(content.lower().startswith(p) for p in summarize_prefixes) and
                    not any(content.lower().endswith(s) for s in summarize_suffixes) and
                    is_likely_summarization_candidate):
                
                    return {
                        "text": "I noticed you've shared a long text. Would you like me to create a summary of it? If so, please confirm by saying 'yes' or 'summarize'.",
                        "requires_confirmation": True,
                        "pending_text": text_to_summarize
                    }
            # Check if text is too short for meaningful summarization
            if len(text_to_summarize.split()) < 50:
                return {"text": "The provided text is too short for effective summarization. Please provide a longer text (at least 50 words)."}
            
            logger.info(f"Summarizing text of length: {len(text_to_summarize)}")

            try:
                summary = self.extractive_news_summarizer.summarize(text_to_summarize)
        
                # Add a prefix to clearly indicate this is a summary
                response = "📝 SUMMARY:\n\n" + summary
        
                return {"text": response}
            
            except Exception as e:
                logger.error(f"Error in extractive summarization: {e}", exc_info=True)
                return {"text": "An error occurred during summarization. Please try again with different text."}
        
        except Exception as e:
            logging.error(f"Error in _handle_extractive_summarization_news: {e}", exc_info=True)
            return {"text": "An error occurred during summarization. Please try again with different text."}
        
    def _handle_document_qa(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document QA intent with conversation context."""
        if not self.session.documents_loaded:
            return {
                "text": "Please upload some documents first before asking questions about them."
            }
        
        # Get the user's question
        question = intent_data.get("resolved_query", "")
        
        # Answer the question with conversation history
        try:
            # Call answer_question_with_context with conversation history
            answer_result = self.document_qa.answer_question_with_context(
                question=question,
                conversation_history=self.session.qa_conversation_history
            )
            
            # Store the answer in session
            self.session.last_qa_answer = answer_result
            
            # Add this exchange to conversation history
            self.session.qa_conversation_history.append({
                "question": question,
                "answer": answer_result.get("answer", "")
            })
            
            # Keep conversation history to a reasonable size
            if len(self.session.qa_conversation_history) > 10:
                # Keep only the 10 most recent exchanges
                self.session.qa_conversation_history = self.session.qa_conversation_history[-10:]
            
            # Use the answer with citations for display
            response_text = answer_result.get("answer_with_citations", answer_result.get("answer", ""))
            
            return {
                "text": response_text,
                "qa_result": answer_result
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "text": "I encountered an error while trying to answer your question. Please try again or try rephrasing your question."
            }
    
    def _handle_summarization_info(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle intent for information about summarization systems."""
        response = intent_data.get("fulfillment_text", "")
        
        if not response:
            response = """
            I can perform two types of summarization:
            
            1. Abstractive Summarization: Creates a concise summary in my own words, capturing the main points while potentially using different phrasing than the original text.
            
            2. Extractive Summarization: Creates a summary by selecting and combining the most important sentences directly from the original document.
            
            You can specify parameters like length (short, medium, long) and focus topics.
            """
        
        return {
            "text": response
        }
     
    def _handle_show_settings(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Show current summary settings."""
        settings = self.session.settings
        
        # Get available topics if documents are loaded
        available_topics = []
        if self.session.documents_loaded and self.retrieval_system:
            try:
                available_topics = self.retrieval_system.get_available_topics()
            except Exception as e:
                logger.warning(f"Error getting available topics: {str(e)}")
        
        # Format the response
        response_text = "Current summarization settings:\n"
        response_text += f"- Type: {settings['summarization_type'].capitalize()}\n"
        response_text += f"- Length: {settings['summary_length'].capitalize()}\n"
        
        if settings["focus_topics"]:
            response_text += f"- Focus topics: {', '.join(settings['focus_topics'])}\n"
        else:
            response_text += "- Focus topics: None (will summarize everything)\n"
        
        if available_topics:
            response_text += f"\nAvailable topics in your documents: {', '.join(available_topics[:10])}"
            if len(available_topics) > 10:
                response_text += f" and {len(available_topics) - 10} more"
        
        return {
            "text": response_text
        }
        
    def _handle_dialogflow_qa(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle QA_CQ* intents by using the fulfillment text from Dialogflow 
        instead of falling back to document_qa.
        """
        # Get the response directly from Dialogflow
        response = intent_data.get("fulfillment_text", "")
        
        # If there's no fulfillment text for some reason, provide a default response
        if not response:
            response = "I understand your question, but I don't have a specific answer prepared for it."
        
        return {
            "text": response
        }
    
    def _handle_upload_document(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document upload intent."""
        # This is mostly handled in the main app flow, so this is just informational
        return {
            "text": "To upload a document, you can use the file upload button below. I support PDF and PPTX files."
        }
    
    def _handle_list_documents(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """List currently loaded documents."""
        if not self.session.active_documents:
            return {
                "text": "You haven't uploaded any documents yet. Use the file upload button below to add some."
            }
        
        # Format the document list
        doc_list = "\n".join([f"- {doc}" for doc in self.session.active_documents])
        return {
            "text": f"Currently loaded documents:\n{doc_list}"
        }
    
    def _handle_fallback(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fallback intent."""
        # Get the raw text if available
        raw_text = intent_data.get("raw_text", "")
    
        # If this is a fallback due to length and we have raw text, treat as summarization
        if intent_data.get("is_fallback", False) and len(raw_text) > 250:
            logger.info("Fallback intent triggered with long text - checking if summarization")

            # Check for summarization patterns
            summarize_prefixes = ["summarize this:", "can you summarize this:", 
                                "could you summarize this:", "please summarize this:", 
                                "summarize the following:", "summary of this:"]
        
            summarize_suffixes = ["summarize this", "please summarize", "can you summarize this?",
                                "summarize the above", "summarize this text", "create a summary"]
        
            is_summarization_request = (
                any(raw_text.lower().startswith(prefix) for prefix in summarize_prefixes) or
                any(raw_text.lower().endswith(suffix) for suffix in summarize_suffixes)
            )
        
            # Or check if it's a candidate for summarization
            is_summarization_candidate = (
                len(raw_text) > 300 and 
                "?" not in raw_text and
                len(raw_text.split()) > 100
            )
        
            if is_summarization_request:
                # Treat as explicit summarization request
                logger.info("Fallback intent with long text detected as summarization request")
                return self._handle_extractive_summarization_news({
                    "raw_text": raw_text,
                    "is_fallback": True
                })
            elif is_summarization_candidate:
                # Ask for confirmation
                return {
                    "text": "I noticed you've shared a long text. Would you like me to create a summary of it? If so, please confirm by saying 'yes' or 'summarize'.",
                    "requires_confirmation": True,
                    "pending_text": raw_text
                }
    
        # Original fallback logic
        response = intent_data.get("fulfillment_text", "")
    
        if not response:
            if not self.session.documents_loaded:
                response = "I didn't understand that. You can upload documents using the file button below, or ask me about summarization systems."
            else:
                response = "I didn't understand that. Since you have documents loaded, I'll try to answer as a question about your documents.\n\n"
                # Try to handle as a document QA intent
                qa_response = self._handle_document_qa(intent_data)
                response += qa_response["text"]
    
        return {
            "text": response
        }