# intent_handler.py
import logging
import re
import os
from typing import Dict, Any, List, Optional
from google.protobuf.json_format import MessageToJson

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
            "extractive_summarization": self._handle_extractive_summarization,
            "QA_Document": self._handle_document_qa,
            "summarization_info": self._handle_summarization_info,
            "show_settings": self._handle_show_settings,
            "upload_document": self._handle_upload_document,
            "Document_Status": self._handle_document_status,
            "FallbackIntent": self._handle_fallback
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

    def extract_sys_any_protobuf(parameter):
        """
        Extract values from a Dialogflow sys.any parameter using Protobuf's JSON converter.
        
        Args:
            parameter: A protobuf message object (like from Dialogflow's sys.any)
            
        Returns:
            Extracted value(s) as Python native types
        """
        try:
            # Convert protobuf message to JSON string
            serialized = MessageToJson(parameter)
            
            # Parse JSON string back to Python dict/list
            parsed = json.loads(serialized)
            
            # Debug output
            logger.debug(f"Converted protobuf to: {parsed}")
            
            # Extract values from common Dialogflow structures
            if isinstance(parsed, dict):
                # Try various common field names in Dialogflow responses
                for field in ['values', 'items', 'listValue', 'value', 'stringValue']:
                    if field in parsed:
                        return parsed[field]
                
                # If there's only one key, return its value
                if len(parsed) == 1:
                    return list(parsed.values())[0]
                
                # Return the whole dict if we couldn't extract a specific field
                return parsed
            
            return parsed
        except Exception as e:
            logger.error(f"Error converting protobuf to JSON: {str(e)}")
            # Fallback to string representation if JSON conversion fails
            try:
                return str(parameter)
            except:
                return None
    
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
            
            # Use the summary with embedded citations and sources
            if "summary_with_sources" in summary_result:
                # This is the summary text with citations and source list at the end
                response_text = summary_result["summary_with_sources"]
            else:
                # Fallback to older format if the new field isn't available
                summary_text = summary_result["summary"]
                metadata = summary_result["metadata"]
                
                source_info = ""
                if "source_documents" in metadata and metadata["source_documents"]:
                    source_docs = metadata["source_documents"]
                    source_info = f"\n\n\nSource(s): {', '.join(source_docs)}"
                
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
        
    def _handle_extractive_summarization(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle extractive summarization intent."""
        return {
                "text": "Extractive summary not done."
            }
    
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

    def _handle_document_status(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle requests to check status of uploaded documents and their topics.
        
        Args:
            intent_data: Intent data from Dialogflow
            
        Returns:
            Response with document status information
        """
        if not self.session.documents_loaded:
            return {
                "text": "You haven't uploaded any documents yet. Use the file upload button to add documents."
            }
        
        # Get list of document names
        document_names = self.session.document_names if hasattr(self.session, 'document_names') else []
        
        # Get topics from the retrieval system
        topics = []
        try:
            if self.retrieval_system:
                topics = self.retrieval_system.get_available_topics()
        except Exception as e:
            logger.error(f"Error retrieving topics: {str(e)}")
        
        # Build the response
        doc_count = len(document_names)
        response_text = f"You have {doc_count} document{'s' if doc_count != 1 else ''} uploaded:\n\n"
        
        # List documents
        for i, doc_name in enumerate(document_names):
            response_text += f"{i+1}. {doc_name}\n"
        
        # Add topics section if any were found
        if topics:
            response_text += f"\nI've identified the following topics in your documents:\n"
            for topic in topics:
                response_text += f"- {topic}\n"
        else:
            response_text += "\nNo specific topics were identified in your documents."
        
        # Add instructions for what they can do next
        response_text += "\n\nYou can now:\n"
        response_text += "- Ask for an abstractive summary (short, medium, or long)\n"
        response_text += "- Ask questions about specific content in the documents\n"
        response_text += "- Request a summary focused on specific topics"
        
        return {
            "text": response_text
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
        # Get the fulfillment text from Dialogflow if available

        if not self.session.documents_loaded:
            response = "Hello! You can upload documents using the file button below, or ask me about summarization systems."
        else:
            try:
                # Try to handle as a document QA intent
                qa_response = self._handle_document_qa(intent_data)
                response += qa_response["text"]
            except:
                response = "I tried querying the document but was unable to find anything."
        
        return {
            "text": response
        }