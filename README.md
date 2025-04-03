# Summi - Document Summarization & QA Chatbot

Summi is an intelligent chat interface for document analysis, summarization, and question answering. It processes PDF and PowerPoint documents and helps users extract key information through natural conversation.

## Features

- **Document Processing**: Upload and process PDF and PPTX files
- **Abstractive Summarization**: Generate concise summaries that capture the essence of uploaded documents using AI
- **Extractive Summarization**: Extract key sentences from news articles or text shared in the chat
- **Document Q&A**: Ask questions about uploaded documents and get answers with citations
- **Intent-based Responses**: Integration with Dialogflow for handling common questions and requests

## System Components

- **Document Processor**: Handles PDF/PPTX extraction, text chunking, and embedding generation
- **Vector Store**: ChromaDB-based storage for document embeddings and efficient retrieval
- **Retrieval System**: Semantic search for finding relevant document sections
- **Abstractive Summarizer**: LLM-powered summarization with Ollama integration
- **Extractive Summarizer**: BART-based extractive summarization for news articles
- **Document QA**: Question answering with citation support
- **Dialogflow Integration**: Natural language understanding for intent classification

## Setup

### Prerequisites

- Python 3.8+
- Ollama (optional, for local LLM support)
- Google Cloud account (for Dialogflow integration)

### Installation

1. Clone the repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Configure environment variables in a `.env` file:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
   DIALOGFLOW_PROJECT_ID=your-dialogflow-project-id
   OLLAMA_ENDPOINT=http://localhost:11434/api/generate
   OLLAMA_MODEL=llama3.1:8b
   ```
4. The fine-tuned BART model could be accessed in: https://drive.google.com/drive/folders/1nUEZflHO5EkFtYAfAwrmOokTBkXVr-rV?usp=share_link

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```
2. Upload PDF or PPTX documents using the file upload button
3. Ask questions or request summaries through the chat interface

### Example Commands

- "Generate an abstractive summary of the document"
- "Provide an extractive summary of this article: [paste article text]"
- "What are the key topics in this document?"
- "What is the difference between abstractive and extractive summarization?"
- "What does [specific term] mean in the context of this document?"

## Customization

- **Summarization Parameters**: Adjust length settings ("short", "medium", "long") and focus on specific topics
- **Embedding Models**: Change the embedding model in `document_processor.py`
- **Chunking Settings**: Modify chunk size and overlap for document processing in `document_processor.py`

## Notes

- For optimal performance, Ollama LLM integration is recommended
- The system can handle multiple documents and maintains conversation context for follow-up questions
- Citation support helps track information sources within documents

## Troubleshooting

- If OCR is not working, ensure you have Tesseract installed
- For image processing capabilities, ensure required libraries are installed
- Check logs for detailed error information if summarization or QA fails
