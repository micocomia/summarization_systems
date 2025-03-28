# Key modifications for app.py

import streamlit as st
import json
import os
import base64
import time
import logging
import re
import hashlib
import uuid  # For session IDs
from typing import Dict, Any, List, Optional

# Import components
from document_processor import DocumentProcessor 
from vector_store import VectorStore
from retrieval import RetrievalSystem
from dialogflow_integration import DialogflowIntegration  
from intent_handler import IntentHandlerManager, SessionState
from abstractive_summarizer import AbstractiveSummarizer 
from document_qa import DocumentQA  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set avatars
user_avatar = None
assistant_avatar = None

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.documents = []
    st.session_state.document_names = []
    st.session_state.topics = []
    st.session_state.awaiting_response = False
    st.session_state.processing_type = None 
    st.session_state.processing_start_time = None
    st.session_state.show_processing = False  # Processing display flag
    st.session_state.processing_message = ""  # Message to show during processing
    st.session_state.session_id = str(uuid.uuid4())  # Unique session ID for Dialogflow

def initialize_systems():
    """Initialize all the required systems."""
    # Only initialize once
    if st.session_state.initialized:
        return
    
    print("Here")
    # Document Processor
    st.session_state.document_processor = DocumentProcessor(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=100,
    )

    print("Document Processor Initialized")

    # Vector store for document storage
    st.session_state.vector_store = VectorStore(
        collection_name="document_collection",
        persist_directory="./data/vector_store"
    )

    print("Vector Store Initialized")
    
    # Retrieval system for finding relevant content
    st.session_state.retrieval_system = RetrievalSystem(
        vector_store=st.session_state.vector_store
    )

    print("Retrieval System Initialized")

    # Dialogflow integration for intent classification
    st.session_state.dialogflow = DialogflowIntegration()
    
    print("Dialogflow Initialized")

    # Abstractive summarizer
    st.session_state.abstractive_summarizer = AbstractiveSummarizer(
        retrieval_system=st.session_state.retrieval_system,
        use_ollama=True
    )
    
    print("Abstractive Summarizer Initialized")

    # Document QA
    st.session_state.document_qa = DocumentQA(
        retrieval_system=st.session_state.retrieval_system,
        use_ollama=True
    )
        
    # Intent handler manager
    st.session_state.intent_handler = IntentHandlerManager(
        document_processor=st.session_state.document_processor,
        retrieval_system=st.session_state.retrieval_system,
        abstractive_summarizer=st.session_state.abstractive_summarizer,
        document_qa=st.session_state.document_qa
    )
    
    st.session_state.initialized = True
    logger.info("All systems initialized")

def add_message(role: str, avatar: str, content: str, **kwargs):
    """Add a message to the conversation history."""
    message_data = {"role": role, 
                    "avatar": avatar,
                    "content": content}
    
    # Add any additional data (like question data)
    for key, value in kwargs.items():
        message_data[key] = value
        
    st.session_state.messages.append(message_data)

def display_chat_messages():
    """Display chat messages with special formatting for questions."""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(name=message["role"], avatar=message["avatar"]):
            # Check if this message contains a question
            if message["role"] == "assistant" and "question" in message:
                question_data = message["question"]
                
                # Display the question text
                st.write(message["content"])
                
                # Special handling for multiple-choice questions
                if question_data.get("type") == "multiple-choice" and "options" in question_data:
                    # Create a container for options with better styling
                    options_container = st.container()
                    with options_container:
                        st.markdown("### Options:")
                        options = question_data["options"]
                        
                        # Display each option with a letter label
                        for idx, option in enumerate(options):
                            option_letter = chr(65 + idx)  # Convert to A, B, C, D
                            st.markdown(f"**{option_letter}.** {option}")
            else:
                # Regular message without question data
                st.write(message["content"])

def process_uploaded_file(uploaded_file, is_part_of_batch=False):
    """
    Process an uploaded document and provide feedback in the chat.
    
    Args:
        uploaded_file: The file to process
        is_part_of_batch: Whether this file is part of a batch upload (affects messaging)
    """
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("./uploads", exist_ok=True)
        
        # Save the file temporarily
        file_path = os.path.join("./uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Add an initial message to show upload started (only if not part of batch)
        if not is_part_of_batch:
            add_message("assistant", assistant_avatar, f"Processing '{uploaded_file.name}'...")
        
        # Process the document using the document processor
        processed_chunks = st.session_state.document_processor.process_document(file_path)
        
        # Check if we got valid results (non-empty list)
        if processed_chunks and isinstance(processed_chunks, list) and len(processed_chunks) > 0:
            # Add processed chunks to vector store
            st.session_state.vector_store.add_documents(processed_chunks)
            
            # Update session state
            st.session_state.documents.append(file_path)
            st.session_state.document_names.append(uploaded_file.name)
            
            # Extract topics from the first chunk's metadata
            if "metadata" in processed_chunks[0] and "topics" in processed_chunks[0]["metadata"]:
                topics = processed_chunks[0]["metadata"]["topics"]
                for topic in topics:
                    if topic not in st.session_state.topics:
                        st.session_state.topics.append(topic)
                    
            # Update the session to indicate documents are loaded
            st.session_state.intent_handler.session.documents_loaded = True
                
            # Update the latest assistant message with success info (only if not part of batch)
            if not is_part_of_batch:
                st.session_state.messages[-1]["content"] = (
                    f"I've successfully processed '{uploaded_file.name}'.\n\n"
                )
            else:
                # For batch processing, just log success without updating messages
                logger.info(f"Successfully processed '{uploaded_file.name}' with {len(processed_chunks)} chunks")
            
            return True
        else:
            # Handle processing failure due to empty results
            if not is_part_of_batch:
                st.session_state.messages[-1]["content"] = (
                    f"I couldn't process '{uploaded_file.name}'.\n\n"
                    f"No content was extracted from the document. Please check if it's a valid PDF or PPTX file."
                )
            else:
                # For batch processing, add a specific error message
                add_message("assistant", assistant_avatar, f"Failed to process '{uploaded_file.name}': No content extracted.")
            return False
            
    except Exception as e:
        # Handle processing failure
        logger.error(f"Error processing file: {str(e)}")
        if not is_part_of_batch:
            st.session_state.messages[-1]["content"] = (
                f"I couldn't process '{uploaded_file.name}'.\n\n"
                f"Error: {str(e)}"
            )
        else:
            # For batch processing, add a specific error message
            add_message("assistant", assistant_avatar, f"Failed to process '{uploaded_file.name}': {str(e)}")
        return False

def handle_user_input(user_input: str):
    """
    Process user input and generate a response.
    This function adds the user message to history and marks it for processing.
    """
    if not user_input:
        return
    
    # Add user message to history
    add_message("user", user_avatar, user_input)
    
    # Set up processing state to show a spinner
    st.session_state.show_processing = True
    
    # Force a rerun to update the UI and show the processing indicator
    st.rerun()

def generate_assistant_response():
    """
    Process the most recent user message and generate a response.
    This is called only when show_processing is True.
    """
    try:
        # Add a slight delay before responding
        time.sleep(0.3)

        # Get the most recent user message
        user_input = st.session_state.messages[-1]["content"]
        
        # Use Dialogflow to determine intent
        intent_data = st.session_state.dialogflow.detect_intent(
            user_input,
            st.session_state.session_id
        )
        
        intent_type = intent_data.get("intent", "fallback")
        
        # Process the intent
        response = st.session_state.intent_handler.handle_intent(intent_type, intent_data)
        
        # Check for special forward instruction (e.g., to extractive summarization)
        if "forward_to" in response:
            # Handle forwarding to external modules
            forward_to = response.get("forward_to")
            if forward_to == "extractive_summarization":
                # This would be handled by your groupmates' module
                # For now, return a placeholder message
                response_text = "The extractive summarization request has been forwarded to the external module."
            else:
                response_text = response.get("text", "Your request has been forwarded.")
        else:
            response_text = response.get("text", "I'm not sure how to respond to that.")
        
        # Prepare message kwargs for special data
        message_kwargs = {}
        
        # Include summary data if present
        if "summary" in response:
            message_kwargs["summary"] = response["summary"]
        
        # Include QA result if present
        if "qa_result" in response:
            message_kwargs["qa_result"] = response["qa_result"]
                
        # Add assistant message to history with any special data
        add_message("assistant", assistant_avatar, response_text, **message_kwargs)

        # Reset processing indicators
        st.session_state.show_processing = False

        # Force a rerun to update the UI with the message
        st.rerun()
        
    except Exception as e:
        # Handle errors gracefully
        logger.error(f"Error processing response: {str(e)}")
        add_message("assistant", assistant_avatar, f"I encountered an error while processing your request. Please try again or try rephrasing your message.")
    finally:
        # Reset processing indicators
        st.session_state.show_processing = False

def main():
    """Main Streamlit app function."""
    st.set_page_config(
        page_title="Summi",
        page_icon="📝",
        layout="wide"
    )

    # Initialize systems
    initialize_systems()
    
    # Special processing path
    if st.session_state.show_processing:
        display_chat_messages()
        
        with st.chat_message("assistant", avatar=assistant_avatar):
            typing_container = st.empty()
            typing_container.markdown("*Processing your request...*")
            generate_assistant_response()
            typing_container.empty()
        st.rerun()

    # Display chat messages or placeholder if no messages
    if st.session_state.messages:
        display_chat_messages()
    else:
        # Display a placeholder when no conversation has started
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 60vh; text-align: center;">
            <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f0e96e; color: #616054; max-width: 600px;">
                <h2>Start chatting with Summi</h2>
                <p>Upload your documents or ask a question about summarization systems to begin.</p>
                <p>For summaries, specify the type and length.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle file uploads directly from chat input
    user_input = st.chat_input(
        "Type your message here or upload files",
        accept_file="multiple",
        file_type=["pdf", "pptx"]
    )
    
    # Process text input if provided
    if user_input and user_input.text:
        handle_user_input(user_input.text)
    # Process uploaded files if any
    elif user_input and user_input["files"]:
        # Add a message showing that files were uploaded
        file_names = [file.name for file in user_input["files"]]
        files_str = ", ".join(file_names)
        
        # Process each uploaded file
        for uploaded_file in user_input["files"]:
            with st.chat_message("assistant", avatar=assistant_avatar):
                typing_container = st.empty()
                typing_container.markdown(f"*Processing {uploaded_file.name}...*")  # Or any subtle indicator you prefer

                success = process_uploaded_file(uploaded_file, is_part_of_batch=(len(user_input["files"]) > 1))
                if not success:
                    st.error(f"Failed to process {uploaded_file.name}")

                # Clear the typing indicator before rerun
                typing_container.empty()
        
        # Add a summary message for multiple files
        if len(user_input["files"]) > 1:
            add_message("assistant", assistant_avatar, f"Processed {len(user_input['files'])} files. You can now start a review session with 'Start review' command.")
        # For single file uploads, provide guidance if not already provided
        elif len(user_input["files"]) == 1 and success:
            # If the last message was just a processing confirmation, replace it with guidance
            if st.session_state.messages[-1]["role"] == "assistant" and "I've successfully processed" in st.session_state.messages[-1]["content"]:
                st.session_state.messages[-1]["content"] += "\n\nWhat would you like to do next? You can:\n- Ask questions about the document\n- Ask me to provide an abstractive or extractive summary of the document"
            # If it was a different kind of message, add a new guidance message
            else:
                add_message("assistant", assistant_avatar, "Now that your document is processed, you can:\n- Ask questions about the document\n- Ask me to provide an abstractive or extractive summary of the document")
        
        # Force a rerun to update the UI with new messages
        st.rerun()

if __name__ == "__main__":
    main()