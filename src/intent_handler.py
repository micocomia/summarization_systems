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
                 news_extractive_summarizer=None,
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
        self.news_extractive_summarizer = news_extractive_summarizer
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
            summary_text = summary_result["summary_with_sources"]
                        
            return {
                "text": summary_text,
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
        Implements a two-step process to work around Dialogflow's 256-character limit.
        """
        try:
            # Get the original raw text from the request
            raw_text = intent_data.get("raw_text", "")
            
            # Define phrases that indicate a request for summarization capability
            initial_summarization_phrases = [
                "generate an extractive summary", 
                "create an extractive summary",
                "i want an extractive summary", 
                "provide an extractive summary",
                "i need an extractive summary",
                "extractive summary of a news article",
                "summarize a news article",
                "extractive summary of an article",
                "extractive summarization"
            ]
            
            # Check if this is just asking for summarization capability
            is_initial_request = any(phrase in raw_text.lower() for phrase in initial_summarization_phrases)
            has_article_text = len(raw_text.split()) > 50  # Assuming articles have more than 50 words
            
            # If this is a request for summarization without article text, prompt for it
            if is_initial_request and not has_article_text:
                return {
                    "text": "I'd be happy to create an extractive summary for you. Please enter the news article or text you want me to summarize in the chat.",
                    "awaiting_article": True
                }
                
            # If we have text to summarize, proceed with summarization
            if has_article_text:
                text_to_summarize = raw_text
            else:
                # Try to extract text to summarize from patterns
                summarize_prefixes = [
                    "summarize this:", 
                    "can you summarize this:", 
                    "could you summarize this:", 
                    "please summarize this:", 
                    "summarize the following:", 
                    "summary of this:", 
                    "provide an extractive summary for the following:",
                    "provide a summary of:"
                ]
                
                # Extract text to summarize based on prefixes
                text_to_summarize = raw_text
                for prefix in summarize_prefixes:
                    if raw_text.lower().startswith(prefix):
                        text_to_summarize = raw_text[len(prefix):].strip()
                        break
                        
            # Check if text is too short for meaningful summarization
            if len(text_to_summarize.split()) < 50:
                return {"text": "The provided text is too short for effective summarization. Please provide a longer text (at least 50 words)."}
            
            # Generate the summary
            try:
                logger.info(f"Summarizing text of length: {len(text_to_summarize)}")
                summary = self.news_extractive_summarizer.summarize(text_to_summarize)
                return {"text": "Extractive Summary:\n\n" + summary}
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
        """Handle fallback intent with improved detection of summarization requests."""
        # Get the raw text if available
        raw_text = intent_data.get("raw_text", "")

        # Check if this might be a summarization request
        if len(raw_text) > 250:
            # Check for summarization-related keywords
            summarization_keywords = [
                "summary", "summarize", "summarization", "extract", "extractive"
            ]
            
            if any(keyword in raw_text.lower() for keyword in summarization_keywords):
                # Check if this contains enough text to summarize
                if len(raw_text.split()) > 100:
                    # It likely contains both a request and content to summarize
                    return self._handle_extractive_summarization_news({
                        "raw_text": raw_text,
                        "is_fallback": True
                    })
                else:
                    # Likely just asking for summarization capability
                    return {
                        "text": "I'd be happy to create an extractive summary for you. Please enter the news article or text you want me to summarize in the chat.",
                        "awaiting_article": True
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