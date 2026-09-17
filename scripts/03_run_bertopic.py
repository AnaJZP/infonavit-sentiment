"""
Phase 2b: Topic Modeling with BERTopic
========================================
Runs BERTopic on all headlines to discover latent themes.

Model: paraphrase-multilingual-MiniLM-L12-v2
Input:  data/infonavit_nlp_results.csv
Output: results/bertopic_topics.csv, results/topic_details.json,
        results/topic_sentiment_cross.csv

See Section 3.4 of the paper.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

print("Loading NLP results...")
df = pd.read_csv(NLP_RESULTS_CSV, parse_dates=['Fecha'])
print(f"Dataset: {len(df):,} rows")

print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

vectorizer_model = CountVectorizer(
    stop_words=SPANISH_STOPS,
    min_df=BERTOPIC_MIN_DF,
    ngram_range=(1, 2)
)

docs = df['Encabezado'].astype(str).tolist()
timestamps = df['Fecha'].tolist()

print(f"\nFitting BERTopic on {len(docs):,} documents...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    language="multilingual",
    min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
    nr_topics=BERTOPIC_NR_TOPICS,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)
df['Topic'] = topics
df['Topic_Prob'] = [p.max() if hasattr(p, 'max') else p for p in probs]

topic_info = topic_model.get_topic_info()
print(f"\nTopics discovered: {len(topic_info) - 1}")
topic_info.to_csv(os.path.join(RESULTS_DIR, 'bertopic_topics.csv'), index=False)

# Topic-Sentiment cross-tabulation
topic_sent = df[df['Topic'] != -1].groupby(['Topic', 'NLP_Sentimiento']).size().unstack(fill_value=0)
topic_sent.to_csv(os.path.join(RESULTS_DIR, 'topic_sentiment_cross.csv'))

# Topic details JSON
topics_data = {}
for topic_id in sorted(df['Topic'].unique()):
    if topic_id == -1:
        continue
    topic_df = df[df['Topic'] == topic_id]
    topics_data[str(topic_id)] = {
        'count': len(topic_df),
        'sentiment_distribution': topic_df['NLP_Sentimiento'].value_counts().to_dict(),
        'top_headlines': topic_df.nlargest(5, 'NLP_Score')['Encabezado'].tolist(),
        'date_range': {
            'start': str(topic_df['Fecha'].min()),
            'end': str(topic_df['Fecha'].max())
        }
    }
with open(os.path.join(RESULTS_DIR, 'topic_details.json'), 'w', encoding='utf-8') as f:
    json.dump(topics_data, f, indent=2, ensure_ascii=False)

df.to_csv(NLP_RESULTS_CSV, index=False, encoding='utf-8-sig')
print(f"\nDataset updated with topics.")
print("Phase 2b complete!")
