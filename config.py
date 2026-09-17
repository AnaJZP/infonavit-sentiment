"""
Centralized configuration for the INFONAVIT Sentiment Analysis pipeline.

All scripts import paths and parameters from this module instead of
hardcoding them, ensuring portability across machines.
"""
import os

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# Intermediate files
CONSOLIDATED_CSV = os.path.join(BASE_DIR, 'data', 'infonavit_consolidated.csv')
NLP_RESULTS_CSV = os.path.join(BASE_DIR, 'data', 'infonavit_nlp_results.csv')

# ── Model parameters ──────────────────────────────────────────
SENTIMENT_MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# ── RRS weights used by the ORIGINAL scripts/04_rrs_model.py (superseded) ──
# These do not match the manuscript. The published weights (0.35, 0.25, 0.25,
# 0.15) and Eqs. (1)-(5) are implemented in scripts/06_rrs_corrected.py.
RRS_WEIGHTS = {
    'volume':    0.20,   # Media volume (log-normalized)
    'neg_ratio': 0.35,   # Proportion of negative articles
    'nlp_score': 0.25,   # Mean NLP sentiment score (inverted)
    'diversity': 0.20,   # Number of unique outlets (diffusion)
}

# ── BERTopic parameters ──────────────────────────────────────
BERTOPIC_MIN_TOPIC_SIZE = 100
BERTOPIC_NR_TOPICS = 'auto'
BERTOPIC_MIN_DF = 10

# ── Spanish stopwords for topic modeling ─────────────────────
SPANISH_STOPS = [
    'de', 'la', 'el', 'en', 'y', 'los', 'del', 'las', 'un', 'por',
    'con', 'una', 'su', 'para', 'es', 'al', 'que', 'se', 'no', 'lo',
    'más', 'a', 'como', 'pero', 'sus', 'le', 'ha', 'me', 'si', 'ya',
    'sin', 'sobre', 'este', 'entre', 'cuando', 'muy', 'ser', 'son',
    'también', 'fue', 'había', 'todo', 'esta', 'son', 'dos', 'hay',
    'desde', 'está', 'mi', 'porque', 'qué', 'o', 'ni', 'cada',
    'infonavit', 'nan'
]

# ── Ensure output directories exist ──────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
