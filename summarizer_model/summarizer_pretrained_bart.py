# summarizer.py

from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Summarization function
def summarize(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    
    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary
