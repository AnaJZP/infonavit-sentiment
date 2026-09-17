"""
Phase 2: NLP Sentiment Analysis
=================================
Applies multilingual BERT sentiment classification to all headlines
and compares with manual tone labels.

Model: nlptown/bert-base-multilingual-uncased-sentiment (5-star → ternary)
Input:  data/infonavit_consolidated.csv
Output: data/infonavit_nlp_results.csv, results/sentiment_comparison_metrics.json

See Section 3.3 of the paper.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    cohen_kappa_score, accuracy_score
)

print("Loading consolidated dataset...")
df = pd.read_csv(CONSOLIDATED_CSV, parse_dates=['Fecha'])
print(f"Dataset: {len(df):,} rows")

# ── Sentiment Analysis ────────────────────────────────────────
print(f"\nLoading model: {SENTIMENT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

sentiment_pipe = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer,
    device=-1, truncation=True, max_length=128, batch_size=64
)

def map_stars_to_sentiment(label, score):
    """Map 1-5 star rating to ternary sentiment."""
    stars = int(label.split()[0])
    if stars <= 2:
        return 'Negativo', -score
    elif stars == 3:
        return 'Neutral', 0.0
    else:
        return 'Positivo', score

headlines = df['Encabezado'].tolist()
batch_size = 256
nlp_sentiments, nlp_scores = [], []

total_batches = (len(headlines) + batch_size - 1) // batch_size
print(f"Processing {len(headlines):,} headlines in {total_batches} batches...")

for i in range(0, len(headlines), batch_size):
    batch = headlines[i:i+batch_size]
    batch_clean = [str(h)[:200] if h else "sin título" for h in batch]
    try:
        results = sentiment_pipe(batch_clean)
        for r in results:
            sent, score = map_stars_to_sentiment(r['label'], r['score'])
            nlp_sentiments.append(sent)
            nlp_scores.append(score)
    except Exception:
        for h in batch_clean:
            try:
                r = sentiment_pipe([h])[0]
                sent, score = map_stars_to_sentiment(r['label'], r['score'])
                nlp_sentiments.append(sent)
                nlp_scores.append(score)
            except Exception:
                nlp_sentiments.append('Neutral')
                nlp_scores.append(0.0)
    if (i // batch_size + 1) % 25 == 0 or i + batch_size >= len(headlines):
        pct = min(100, (i + batch_size) / len(headlines) * 100)
        print(f"  Batch {i//batch_size + 1}/{total_batches} ({pct:.0f}%)")

df['NLP_Sentimiento'] = nlp_sentiments
df['NLP_Score'] = nlp_scores

print(f"\nNLP Sentiment distribution:")
print(df['NLP_Sentimiento'].value_counts())

# ── Comparison NLP vs. Manual ─────────────────────────────────
labels = ['Negativo', 'Neutral', 'Positivo']
y_manual = df['Tono']
y_nlp = df['NLP_Sentimiento']

report = classification_report(y_manual, y_nlp, labels=labels, output_dict=True)
print("\n" + classification_report(y_manual, y_nlp, labels=labels))

kappa = cohen_kappa_score(y_manual, y_nlp)
accuracy = accuracy_score(y_manual, y_nlp)
cm = confusion_matrix(y_manual, y_nlp, labels=labels)
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Accuracy: {accuracy:.4f}")

metrics = {
    'accuracy': accuracy,
    'cohens_kappa': kappa,
    'classification_report': report,
    'confusion_matrix': cm.tolist(),
    'nlp_distribution': df['NLP_Sentimiento'].value_counts().to_dict(),
    'manual_distribution': df['Tono'].value_counts().to_dict()
}
with open(os.path.join(RESULTS_DIR, 'sentiment_comparison_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# ── Descriptive Stats ─────────────────────────────────────────
df.groupby(['Mes', 'Tono']).size().unstack(fill_value=0).to_csv(
    os.path.join(RESULTS_DIR, 'monthly_tone_distribution.csv'))
df.groupby(['Mes', 'NLP_Sentimiento']).size().unstack(fill_value=0).to_csv(
    os.path.join(RESULTS_DIR, 'monthly_nlp_distribution.csv'))
df['Medio'].value_counts().head(30).to_csv(
    os.path.join(RESULTS_DIR, 'top_medios.csv'))
df.groupby(['Tipo', 'Tono']).size().unstack(fill_value=0).to_csv(
    os.path.join(RESULTS_DIR, 'sentiment_by_type.csv'))
df.groupby(['Tipo', 'NLP_Sentimiento']).size().unstack(fill_value=0).to_csv(
    os.path.join(RESULTS_DIR, 'nlp_sentiment_by_type.csv'))
df.groupby(['Estado_Origen', 'Tono']).size().unstack(fill_value=0).to_csv(
    os.path.join(RESULTS_DIR, 'sentiment_by_state.csv'))
df['Tema'].value_counts().head(30).to_csv(
    os.path.join(RESULTS_DIR, 'top_temas.csv'))

top20_medios = df['Medio'].value_counts().head(20).index
df[df['Medio'].isin(top20_medios)].groupby(['Medio', 'Tono']).size().unstack(fill_value=0).to_csv(
    os.path.join(RESULTS_DIR, 'sentiment_by_medio.csv'))

df.to_csv(NLP_RESULTS_CSV, index=False, encoding='utf-8-sig')
print(f"\nEnriched dataset saved: {NLP_RESULTS_CSV}")
print("Phase 2 complete!")
