# random_forest_scientific_app_scientific.py

import streamlit as st
import joblib
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")

# Load model and vectorizer
model = joblib.load("random_forest_summary_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Summary function using Random Forest
def summarize_with_rf(text, model, vectorizer, top_k=3):
    sentences = sent_tokenize(text)
    if not sentences:
        return ""

    X = vectorizer.transform(sentences)
    probs = model.predict_proba(X)[:, 1]

    print("\n🔍 Sentence Probabilities:")
    
    selected = [s for s, p in zip(sentences, probs) if p >= 0.4]
    
    if not selected and top_k:
        selected = [sentences[i] for i in sorted(probs.argsort()[::-1][:top_k])]
    
    for s, p in zip(sentences, probs):
        print(f"{p:.4f} - {s}")

    top_indices = probs.argsort()[::-1][:top_k]
    selected = [sentences[i] for i in sorted(top_indices)]
    return " ".join(selected)


# Streamlit UI
st.title("Extractive Summarization using Random Forest")

st.write("""
This application uses a trained Random Forest model to perform extractive summarization on scientific text.
""")

# Text input for the article
input_text = st.text_area("Enter text to summarize:")

if st.button("Generate Summary"):
    if input_text.strip() != "":
        summary = summarize_with_rf(input_text, model, vectorizer, top_k=5)
        st.subheader("Generated Extractive Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
