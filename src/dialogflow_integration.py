# dialogflow_integration.py
import os
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
import requests
import uuid

load_dotenv()
logger = logging.getLogger(__name__)

class DialogflowIntegration:
    """
    Integration with Dialogflow ES for intent classification
    """
    def __init__(self, project_id: str = None):
        """
        Initialize Dialogflow ES integration.
        
        Args:
            project_id: Google Cloud project ID
        """
        self.project_id = project_id or os.getenv("DIALOGFLOW_PROJECT_ID")
        
        # Verify environment has been configured
        if not self.project_id:
            logger.warning("Dialogflow environment not fully configured. Set DIALOGFLOW_PROJECT_ID.")
    
    def detect_intent(self, text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Detect intent using Dialogflow ES.
        
        Args:
            text: The input text from the user
            session_id: A unique session ID for this conversation (will be generated if None)
            
        Returns:
            Dictionary with intent and other Dialogflow response information
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        try:
            # Use Dialogflow ES API to detect intent
            from google.cloud import dialogflow_v2 as dialogflow
            
            # Create session client
            session_client = dialogflow.SessionsClient()
            
            # Build the session path
            session_path = session_client.session_path(self.project_id, session_id)
            
            # Create text input
            text_input = dialogflow.TextInput(text=text, language_code="en")
            query_input = dialogflow.QueryInput(text=text_input)
            
            # Send the request
            response = session_client.detect_intent(
                request={"session": session_path, "query_input": query_input}
            )
            
            # Extract intent information
            query_result = response.query_result
            
            intent_result = {
                "intent": query_result.intent.display_name if query_result.intent else "fallback",
                "confidence": query_result.intent_detection_confidence,
                "response_text": query_result.fulfillment_text,
                "parameters": dict(query_result.parameters),
                "resolved_query": text,
                "fulfillment_text": query_result.fulfillment_text,
                "all_required_params_present": query_result.all_required_params_present
            }
            
            return intent_result
        
        except ImportError:
            logger.warning("Google Cloud Dialogflow libraries not installed. Using REST API fallback.")
            return self._detect_intent_rest(text, session_id)
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            # Return a fallback intent
            return {
                "intent": "fallback",
                "confidence": 0.0,
                "response_text": "I'm having trouble understanding. Could you rephrase that?",
                "parameters": {},
                "resolved_query": text,
                "fulfillment_text": "I'm having trouble understanding. Could you rephrase that?",
                "all_required_params_present": True
            }
    
    def _detect_intent_rest(self, text: str, session_id: str) -> Dict[str, Any]:
        """Fallback REST API implementation for Dialogflow ES if the Google Cloud library is not available."""
        try:
            from google.oauth2 import service_account
            import google.auth.transport.requests
            
            # Get credentials - assumes you've set GOOGLE_APPLICATION_CREDENTIALS env var
            credentials = service_account.Credentials.from_service_account_file(
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            
            # Create the request
            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': 'application/json; charset=utf-8'
            }
            
            # Dialogflow ES API endpoint
            url = f"https://dialogflow.googleapis.com/v2/projects/{self.project_id}/agent/sessions/{session_id}:detectIntent"
            
            data = {
                "queryInput": {
                    "text": {
                        "text": text,
                        "languageCode": "en"
                    }
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Extract intent information
            query_result = result.get("queryResult", {})
            intent = query_result.get("intent", {})
            
            return {
                "intent": intent.get("displayName", "fallback"),
                "confidence": query_result.get("intentDetectionConfidence", 0.0),
                "response_text": query_result.get("fulfillmentText", ""),
                "parameters": query_result.get("parameters", {}),
                "resolved_query": text,
                "fulfillment_text": query_result.get("fulfillmentText", ""),
                "all_required_params_present": query_result.get("allRequiredParamsPresent", True)
            }
            
        except Exception as e:
            logger.error(f"Error with REST API fallback: {str(e)}")
            
            # Local fallback if everything else fails - use simple heuristics
            return self._local_intent_detection(text)
    
    def _local_intent_detection(self, text: str) -> Dict[str, Any]:
        """Very basic local intent detection as a last resort fallback."""
        text_lower = text.lower()
        
        # Check for summarization intent
        if any(keyword in text_lower for keyword in ["summarize", "summary", "summarization", "abstract", "gist", "overview"]):
            if "abstractive" in text_lower:
                intent = "abstractive_summarization"
            elif "extractive" in text_lower:
                intent = "extractive_summarization"
            else:
                intent = "abstractive_summarization"  # default to abstractive
            
            return {
                "intent": intent,
                "confidence": 0.7,
                "response_text": "I'll create a summary for you.",
                "parameters": {},
                "resolved_query": text,
                "fulfillment_text": "I'll create a summary for you.",
                "all_required_params_present": True
            }
        
        # Check for document QA intent
        elif any(keyword in text_lower for keyword in ["question", "answer", "ask", "explain", "what is", "how does", "tell me about"]):
            return {
                "intent": "document_qa",
                "confidence": 0.7,
                "response_text": "I'll try to answer that based on the documents.",
                "parameters": {},
                "resolved_query": text,
                "fulfillment_text": "I'll try to answer that based on the documents.",
                "all_required_params_present": True
            }
        
        # Check for summarization info intent
        elif any(keyword in text_lower for keyword in ["what is summarization", "types of summarization", "summarization techniques", "about summarization"]):
            return {
                "intent": "summarization_info",
                "confidence": 0.7,
                "response_text": "Let me tell you about summarization techniques.",
                "parameters": {},
                "resolved_query": text,
                "fulfillment_text": "Let me tell you about summarization techniques.",
                "all_required_params_present": True
            }
        
        # Default fallback
        else:
            return {
                "intent": "fallback",
                "confidence": 0.0,
                "response_text": "I'm having trouble understanding. Could you rephrase that?",
                "parameters": {},
                "resolved_query": text,
                "fulfillment_text": "I'm having trouble understanding. Could you rephrase that?",
                "all_required_params_present": True
            }