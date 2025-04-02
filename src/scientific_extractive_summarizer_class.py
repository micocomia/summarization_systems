# scientific_extractive_summarizer_class.py

import joblib
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")

class scientific_extractive_summarizer:
    def __init__(self, model_path='random_forest_summary_model.joblib', vectorizer_path='tfidf_vectorizer.joblib'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def summarize(self, length='medium', focus_topics=None):
        # fallback if no documents are loaded
        try:
            from document_processor import DocumentProcessor
            documents = DocumentProcessor.get_active_document_texts()
            if not documents:
                return {"summary": "No documents loaded."}
            
            text = " ".join(documents)
            sentences = sent_tokenize(text)
            if not sentences:
                return {"summary": "No text available to summarize."}

            # Vectorize and score
            X = self.vectorizer.transform(sentences)
            probs = self.model.predict_proba(X)[:, 1]

            # Length control
            length_map = {"short": 3, "medium": 5, "long": 8}
            top_k = length_map.get(length.lower(), 5)

            # Select top sentences
            top_indices = probs.argsort()[::-1][:top_k]
            selected = [sentences[i] for i in sorted(top_indices)]

            summary = " ".join(selected)
            return summary
        
        except Exception as e:
            return {"summary": f"Error: {str(e)}", "metadata": {}}
