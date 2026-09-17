"""
Phase 4: Publication-Quality Visualizations
=============================================
Generates 11 figures (PNG + PDF) for the JRFM manuscript.

Input:  data/infonavit_nlp_results.csv, results/*.csv
Output: figures/fig1–fig11 (.png and .pdf)

See Figures 1–11 in the paper.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

# Style
plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif'
})

def save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'))
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.pdf'))
    plt.close(fig)
    print(f'  Saved {name}')

# ── Load data ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(NLP_RESULTS_CSV, parse_dates=['Fecha'])
rrs = pd.read_csv(os.path.join(RESULTS_DIR, 'rrs_daily.csv'), parse_dates=['Fecha'])
rrs_monthly = pd.read_csv(os.path.join(RESULTS_DIR, 'rrs_monthly.csv'), parse_dates=['Fecha'])

try:
    stl = pd.read_csv(os.path.join(RESULTS_DIR, 'rrs_stl_decomposition.csv'), parse_dates=['Fecha'])
except FileNotFoundError:
    stl = None

try:
    corr = pd.read_csv(os.path.join(RESULTS_DIR, 'correlation_matrix.csv'), index_col=0)
except FileNotFoundError:
    corr = None

print(f"Main dataset: {len(df):,} rows")
print(f"RRS daily: {len(rrs)} observations")

# ── Fig 1: RRS Time Series ────────────────────────────────────
print("\nGenerating figures...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(rrs['Fecha'], rrs['RRS_scaled'], alpha=0.3, color='steelblue', linewidth=0.5)
ax.plot(rrs['Fecha'], rrs['RRS_scaled'].rolling(30).mean(), color='navy', linewidth=1.5, label='30-day MA')
ax.axhspan(75, 100, alpha=0.1, color='red', label='Critical')
ax.axhspan(50, 75, alpha=0.1, color='orange', label='High')
ax.axhspan(25, 50, alpha=0.1, color='yellow', label='Moderate')
ax.axhspan(0, 25, alpha=0.1, color='green', label='Low')
ax.set_ylabel('RRS (0–100)')
ax.set_xlabel('Date')
ax.set_title('Reputational Risk Score – Daily Time Series (2019–2024)')
ax.legend(loc='upper left', fontsize=8)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
save_fig(fig, 'fig1_rrs_timeseries')

# ── Fig 2: Sentiment Distribution ─────────────────────────────
monthly_tone = df.groupby([df['Fecha'].dt.to_period('M'), 'Tono']).size().unstack(fill_value=0)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
monthly_tone.plot(kind='bar', stacked=True, ax=axes[0], color=['#d32f2f', '#9e9e9e', '#388e3c'], width=0.8)
axes[0].set_title('Manual Tone Distribution')
axes[0].set_xlabel('Month')
axes[0].tick_params(axis='x', labelsize=6, rotation=90)
monthly_nlp = df.groupby([df['Fecha'].dt.to_period('M'), 'NLP_Sentimiento']).size().unstack(fill_value=0)
monthly_nlp.plot(kind='bar', stacked=True, ax=axes[1], color=['#d32f2f', '#9e9e9e', '#388e3c'], width=0.8)
axes[1].set_title('NLP Sentiment Distribution')
axes[1].set_xlabel('Month')
axes[1].tick_params(axis='x', labelsize=6, rotation=90)
plt.tight_layout()
save_fig(fig, 'fig2_sentiment_distribution')

# ── Fig 3: Confusion Matrix ──────────────────────────────────
labels = ['Negativo', 'Neutral', 'Positivo']
cm = confusion_matrix(df['Tono'], df['NLP_Sentimiento'], labels=labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_ylabel('Manual Label')
ax.set_xlabel('NLP Prediction')
ax.set_title('Normalized Confusion Matrix (NLP vs. Manual)')
save_fig(fig, 'fig3_confusion_matrix')

# ── Fig 4: Heatmap Type × Year ────────────────────────────────
df['Year'] = df['Fecha'].dt.year
pivot = df.groupby(['Tipo', 'Year']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('Media Coverage by Type and Year')
save_fig(fig, 'fig4_heatmap_type_year')

# ── Fig 5: Topics × Sentiment ─────────────────────────────────
if 'Topic' in df.columns:
    top_topics = df[df['Topic'] != -1]['Topic'].value_counts().head(20).index
    topic_df = df[df['Topic'].isin(top_topics)]
    pivot = topic_df.groupby(['Topic', 'NLP_Sentimiento']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot.plot(kind='barh', stacked=True, ax=ax, color=['#d32f2f', '#9e9e9e', '#388e3c'])
    ax.set_title('Top 20 Topics by Sentiment')
    ax.set_xlabel('Number of Articles')
    save_fig(fig, 'fig5_topics_sentiment')
else:
    print('  Skipping fig5 (no Topic column)')

# ── Fig 6: Word Clouds ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (sent, color) in zip(axes, [('Positivo', 'Greens'), ('Neutral', 'Greys'), ('Negativo', 'Reds')]):
    text = ' '.join(df[df['NLP_Sentimiento'] == sent]['Encabezado'].astype(str).tolist()[:5000])
    wc = WordCloud(width=800, height=400, background_color='white', colormap=color, max_words=100).generate(text)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(sent)
    ax.axis('off')
save_fig(fig, 'fig6_wordclouds')

# ── Fig 7: Monthly RRS ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(rrs_monthly['Fecha'], rrs_monthly['RRS_scaled'], width=25, color='steelblue', alpha=0.7)
ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
ax.set_ylabel('RRS (0–100)')
ax.set_title('Monthly Average Reputational Risk Score')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
save_fig(fig, 'fig7_rrs_monthly')

# ── Fig 8: STL Decomposition ─────────────────────────────────
if stl is not None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(stl['Fecha'], stl['RRS_mensual'], color='navy')
    axes[0].set_ylabel('Observed')
    axes[1].plot(stl['Fecha'], stl['Tendencia'], color='darkred')
    axes[1].set_ylabel('Trend')
    axes[2].plot(stl['Fecha'], stl['Estacional'], color='darkgreen')
    axes[2].set_ylabel('Seasonal')
    axes[3].plot(stl['Fecha'], stl['Residuo'], color='gray')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    fig.suptitle('STL Decomposition of Monthly RRS', fontsize=14)
    plt.tight_layout()
    save_fig(fig, 'fig8_stl_decomposition')
else:
    print('  Skipping fig8 (no STL data)')

# ── Fig 9: Geographic Distribution ────────────────────────────
state_counts = df['Estado_Origen'].value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 8))
state_counts.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Number of Articles')
ax.set_title('Top 20 States by Media Coverage')
ax.invert_yaxis()
save_fig(fig, 'fig9_geographic_distribution')

# ── Fig 10: Violin Plot NLP Scores ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x='Tono', y='NLP_Score', order=labels, palette=['#d32f2f', '#9e9e9e', '#388e3c'], ax=ax)
ax.set_title('NLP Score Distribution by Manual Tone Label')
ax.set_xlabel('Manual Tone')
ax.set_ylabel('NLP Score')
save_fig(fig, 'fig10_violin_nlp_scores')

# ── Fig 11: Correlation Matrix ────────────────────────────────
if corr is not None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix of RRS Components')
    save_fig(fig, 'fig11_correlation_matrix')
else:
    print('  Skipping fig11 (no correlation data)')

print("\nAll figures generated!")
