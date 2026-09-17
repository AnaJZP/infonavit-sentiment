"""
Phase 3: Reputational Risk Score (RRS)
========================================
Constructs a composite daily index of reputational risk combining:
  - Media volume (log-normalized)        weight = 0.20
  - Negative article ratio               weight = 0.35
  - Mean NLP sentiment score (inverted)   weight = 0.25
  - Media outlet diversity (diffusion)    weight = 0.20

Input:  data/infonavit_nlp_results.csv
Output: results/rrs_daily.csv, results/rrs_monthly.csv,
        results/rrs_stl_decomposition.csv, results/rrs_summary.json

See Section 3.5 and Equation (1) in the paper.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import zscore
from statsmodels.tsa.seasonal import STL

print("Loading NLP results...")
df = pd.read_csv(NLP_RESULTS_CSV, parse_dates=['Fecha'])
print(f"Dataset: {len(df):,} rows")

# ── Daily aggregation ─────────────────────────────────────────
daily = df.groupby(df['Fecha'].dt.date).agg(
    volume=('Encabezado', 'count'),
    neg_manual=('Tono', lambda x: (x == 'Negativo').sum()),
    pos_manual=('Tono', lambda x: (x == 'Positivo').sum()),
    neu_manual=('Tono', lambda x: (x == 'Neutral').sum()),
    neg_nlp=('NLP_Sentimiento', lambda x: (x == 'Negativo').sum()),
    pos_nlp=('NLP_Sentimiento', lambda x: (x == 'Positivo').sum()),
    mean_nlp_score=('NLP_Score', 'mean'),
    std_nlp_score=('NLP_Score', 'std'),
    n_medios=('Medio', 'nunique'),
    n_temas=('Tema', 'nunique'),
    n_estados=('Estado_Origen', 'nunique')
).reset_index()
daily.columns = ['Fecha'] + list(daily.columns[1:])
daily['Fecha'] = pd.to_datetime(daily['Fecha'])

daily['neg_ratio'] = daily['neg_nlp'] / daily['volume']
daily['pos_ratio'] = daily['pos_nlp'] / daily['volume']
daily['net_sentiment'] = daily['pos_ratio'] - daily['neg_ratio']

# ── Z-score normalization ─────────────────────────────────────
W = RRS_WEIGHTS
print(f"RRS weights: {W}")

daily['z_volume'] = zscore(np.log1p(daily['volume']))
daily['z_neg_ratio'] = zscore(daily['neg_ratio'])
daily['z_nlp_score'] = zscore(daily['mean_nlp_score']) * -1
daily['z_diversity'] = zscore(daily['n_medios'])

# ── Composite score ───────────────────────────────────────────
daily['RRS'] = (
    W['volume']    * daily['z_volume'] +
    W['neg_ratio'] * daily['z_neg_ratio'] +
    W['nlp_score'] * daily['z_nlp_score'] +
    W['diversity'] * daily['z_diversity']
)

rrs_min, rrs_max = daily['RRS'].min(), daily['RRS'].max()
daily['RRS_scaled'] = ((daily['RRS'] - rrs_min) / (rrs_max - rrs_min)) * 100

def classify_risk(rrs):
    if rrs >= 75: return 'Crítico'
    elif rrs >= 50: return 'Alto'
    elif rrs >= 25: return 'Moderado'
    else: return 'Bajo'

daily['Riesgo'] = daily['RRS_scaled'].apply(classify_risk)
print(f"\nRisk distribution:\n{daily['Riesgo'].value_counts()}")

# ── STL decomposition (monthly) ──────────────────────────────
monthly = daily.set_index('Fecha').resample('ME')['RRS_scaled'].mean().dropna()
print(f"Monthly series: {len(monthly)} observations")

if len(monthly) >= 24:
    stl = STL(monthly, period=12, robust=True)
    stl_result = stl.fit()
    stl_df = pd.DataFrame({
        'Fecha': monthly.index,
        'RRS_mensual': monthly.values,
        'Tendencia': stl_result.trend,
        'Estacional': stl_result.seasonal,
        'Residuo': stl_result.resid
    })
    stl_df.to_csv(os.path.join(RESULTS_DIR, 'rrs_stl_decomposition.csv'), index=False)
    print("STL decomposition saved.")

# ── Crisis detection ──────────────────────────────────────────
neg_mean = daily['neg_ratio'].mean()
neg_std = daily['neg_ratio'].std()
threshold = neg_mean + 2 * neg_std
crisis_days = daily[daily['neg_ratio'] > threshold].sort_values('neg_ratio', ascending=False)
crisis_days.to_csv(os.path.join(RESULTS_DIR, 'crisis_days.csv'), index=False)
print(f"Crisis days detected: {len(crisis_days)}")

# ── Save all outputs ──────────────────────────────────────────
daily.to_csv(os.path.join(RESULTS_DIR, 'rrs_daily.csv'), index=False)

weekly = daily.set_index('Fecha').resample('W')['RRS_scaled'].mean().reset_index()
weekly.columns = ['Fecha', 'RRS_semanal']
weekly.to_csv(os.path.join(RESULTS_DIR, 'rrs_weekly.csv'), index=False)

monthly_df = daily.set_index('Fecha').resample('ME').agg({
    'RRS_scaled': 'mean', 'volume': 'sum', 'neg_ratio': 'mean', 'n_medios': 'mean'
}).reset_index()
monthly_df.to_csv(os.path.join(RESULTS_DIR, 'rrs_monthly.csv'), index=False)

# ── Correlation matrix ────────────────────────────────────────
corr_vars = ['volume', 'neg_ratio', 'pos_ratio', 'mean_nlp_score', 'n_medios', 'RRS_scaled']
daily[corr_vars].corr().to_csv(os.path.join(RESULTS_DIR, 'correlation_matrix.csv'))

# ── Summary JSON ──────────────────────────────────────────────
summary = {
    'total_articles': int(daily['volume'].sum()),
    'date_range': f"{daily['Fecha'].min()} to {daily['Fecha'].max()}",
    'unique_media': int(df['Medio'].nunique()),
    'unique_topics': int(df['Tema'].nunique()),
    'rrs_mean': float(daily['RRS_scaled'].mean()),
    'rrs_median': float(daily['RRS_scaled'].median()),
    'rrs_std': float(daily['RRS_scaled'].std()),
    'rrs_max': float(daily['RRS_scaled'].max()),
    'crisis_days_count': len(crisis_days),
    'p95_threshold': float(daily['RRS_scaled'].quantile(0.95)),
    'p99_threshold': float(daily['RRS_scaled'].quantile(0.99)),
    'pct_negative': float((df['NLP_Sentimiento'] == 'Negativo').mean() * 100),
    'pct_positive': float((df['NLP_Sentimiento'] == 'Positivo').mean() * 100),
    'pct_neutral': float((df['NLP_Sentimiento'] == 'Neutral').mean() * 100),
    'weights': W
}
with open(os.path.join(RESULTS_DIR, 'rrs_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nPhase 3 complete!")
