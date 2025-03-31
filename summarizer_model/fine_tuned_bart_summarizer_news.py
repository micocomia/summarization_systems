import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

# Load your dataset
df = pd.read_csv('cleaned_data.csv')

# BART Tokenizer 
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Preprocessing function to tokenize the data
def preprocess_function(examples):
    # Tokenize both the input and target
    inputs = tokenizer(examples['article'], max_length=512, truncation=True, padding='max_length')
    outputs = tokenizer(examples['summary'], max_length=150, truncation=True, padding='max_length')
    
    # Return tokenized inputs and outputs
    inputs['labels'] = outputs['input_ids']
    return inputs

# Convert the DataFrame to the Dataset format
dataset = Dataset.from_pandas(df)

# Apply preprocessing function
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Training arguments 
training_args = TrainingArguments(
    output_dir='./results',          
    evaluation_strategy="no",        
    learning_rate=2e-5,              
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,    
    num_train_epochs=3,              
    weight_decay=0.01,               
    logging_dir='./logs',            
    gradient_accumulation_steps=4,   
    fp16=False,                      
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_bart')
tokenizer.save_pretrained('fine_tuned_bart')

# Summarization function using the fine-tuned model
def summarize(text):
    # Load the fine-tuned model
    model = BartForConditionalGeneration.from_pretrained('fine_tuned_bart')
    tokenizer = BartTokenizer.from_pretrained('fine_tuned_bart')

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    
    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary
