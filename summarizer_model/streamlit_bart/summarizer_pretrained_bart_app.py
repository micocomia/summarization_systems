# app.py

import streamlit as st
from summarizer_bart import summarize  # Import the summarize function from summarizer.py

# Streamlit app layout
st.title('News Article Summarizer')
st.markdown("This app uses a pre-trained BART model to summarize news articles.")

# Input text box for the article
article = st.text_area("Enter a news article:", height=300)

if st.button('Generate Summary'):
    if article:
        # Call the summarize function from summarizer.py
        summary = summarize(article)
        st.subheader('Generated Summary:')
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
