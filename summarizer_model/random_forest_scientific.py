import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_recall_curve, auc
)
from imblearn.over_sampling import RandomOverSampler
from rouge_score import rouge_scorer

nltk.download('punkt')

# Load dataset (limit for speed)
df = pd.read_csv("cleaned_scientific_dataset.csv")
df_sample = df.sample(n=20000, random_state=42) # ← speed optimized here

# Load Sentence-BERT
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Optimized sentence labeling
def label_sentences(article, summary, threshold=0.7):
    article_sents = sent_tokenize(article)
    summary_sents = sent_tokenize(summary)

    if not article_sents or not summary_sents:
        return article_sents, [0] * len(article_sents)

    article_embeds = embedder.encode(article_sents, convert_to_tensor=True, batch_size=16, show_progress_bar=False)
    summary_embeds = embedder.encode(summary_sents, convert_to_tensor=True, batch_size=16, show_progress_bar=False)

    sims = util.cos_sim(article_embeds, summary_embeds)
    labels = (sims.max(dim=1).values >= threshold).long().tolist()

    return article_sents, labels

# Process and label data
processed_data = []
for i, row in df_sample.iterrows():
    sents, labels = label_sentences(row['article'], row['summary'])
    for sent, label in zip(sents, labels):
        processed_data.append({'sentence': sent, 'label': label, 'article_index': i})
labeled_df = pd.DataFrame(processed_data)

# Vectorize sentences
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(labeled_df['sentence'])
y = labeled_df['label']

# Balance data
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# Generate summaries
def generate_summary(article, model, vectorizer):
    sentences = sent_tokenize(article)
    features = vectorizer.transform(sentences)
    predictions = model.predict(features)
    return ' '.join([sent for sent, pred in zip(sentences, predictions) if pred == 1])

# ROUGE evaluation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = []

for i, row in df_sample.iterrows():
    gen_summary = generate_summary(row['article'], clf, vectorizer)
    ref_summary = row['summary']
    score = scorer.score(ref_summary, gen_summary)
    rouge_scores.append(score)

# Average ROUGE
avg_rouge = {
    metric: np.mean([score[metric].fmeasure for score in rouge_scores])
    for metric in ['rouge1', 'rouge2', 'rougeL']
}
print("\nAverage ROUGE Scores:")
for metric, score in avg_rouge.items():
    print(f"{metric}: {score:.4f}")
