import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')

# Optimized Dataset Class
class ScientificSummarizationDataset(Dataset):
    def __init__(self, articles, summaries, tokenizer, model, max_length=512, max_sentences=50):
        self.articles = list(articles)
        self.summaries = list(summaries)
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.articles)

    def _get_sentence_importance(self, sentences, summary):
        summary_encoding = self.tokenizer(summary, return_tensors='pt', truncation=True, max_length=self.max_length)
        with torch.no_grad():
            summary_embedding = self.model(**summary_encoding).last_hidden_state.mean(dim=1)
        
        sent_encodings = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        with torch.no_grad():
            sent_embeddings = self.model(**sent_encodings).last_hidden_state.mean(dim=1)
        
        similarity_scores = torch.nn.functional.cosine_similarity(summary_embedding, sent_embeddings)
        if similarity_scores.numel() > 0:
            normalized_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min() + 1e-10)
        else:
            normalized_scores = torch.zeros_like(similarity_scores)
        return normalized_scores.tolist()
    
    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        sentences = nltk.sent_tokenize(str(article))[:self.max_sentences]
        
        sentence_encodings = self.tokenizer(sentences, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        importance_labels = self._get_sentence_importance(sentences, str(summary))

        return {
            'input_ids': sentence_encodings['input_ids'].squeeze(0),
            'attention_mask': sentence_encodings['attention_mask'].squeeze(0),
            'labels': torch.tensor(importance_labels, dtype=torch.float)
        }

# Custom Collate Function
def custom_collate(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

# BERT Summarizer Model
class BERTSummarizer(torch.nn.Module):
    def __init__(self, max_sentences=50):
        super(BERTSummarizer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output).squeeze()

# Training Function
def train_summarization_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Main Function
def main():
    model_path = "C:/Users/Kabi/Desktop/5125_project/beert_summarizer.pth"
    df = pd.read_csv("cleaned_scientific_dataset.csv").reset_index(drop=True)
    train_articles, val_articles, train_summaries, val_summaries = train_test_split(df['article'], df['summary'], test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_instance = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = SCientifcSummarizationDataset(train_articles, train_summaries, tokenizer, model_instance)
    val_dataset = ScientificSummarizationDataset(val_articles, val_summaries, tokenizer, model_instance)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=custom_collate)

    model = BERTSummarizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(10):
        train_loss = train_summarization_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/10, Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

# Summary Generation
def generate_summary(model, article, tokenizer, top_k=3, max_sentences=30):
    sentences = nltk.sent_tokenize(str(article))[:max_sentences]
    sentence_encodings = tokenizer(sentences, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    with torch.no_grad():
        scores = model(sentence_encodings['input_ids'], sentence_encodings['attention_mask']).squeeze()
    
    top_indices = scores.topk(min(top_k, len(sentences))).indices.tolist()
    summary_sentences = [sentences[i] for i in top_indices]
    return ' '.join(summary_sentences)
