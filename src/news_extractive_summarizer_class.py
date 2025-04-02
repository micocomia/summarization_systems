from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class news_extractive_summarizer:
    def __init__(self, model_path='fine_tuned_bart'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
    
    def summarize(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(self.device)
        summary_ids = self.model.generate(
            inputs['input_ids'], 
            max_length=150, 
            min_length=50, 
            length_penalty=1.5, 
            num_beams=6, 
            early_stopping=True
            )
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_text = summary_text.strip()  # Clean up the summary

        # Ensure the summary ends cleanly
        summary_text = self.ensure_end_with_period(summary_text)
        
        return summary_text
    
    def ensure_end_with_period(self, summary):
        # Check if the summary ends with a period or other punctuation, if not, append one.
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
        return summary