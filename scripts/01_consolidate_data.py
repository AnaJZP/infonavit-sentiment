"""
Phase 1: Data Consolidation and Cleaning
==========================================
Merges six yearly Excel files (2019–2024) into a single clean dataset.

Input:  data/raw/BD 2019.xlsx ... BDD 2024.xlsx
Output: data/infonavit_consolidated.csv

See Section 3.2 of the paper for details.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

import pandas as pd
import re

CORE_COLS = [
    'Fecha', 'Medio', 'Autor', 'Tipo', 'Cobertura', 'Estado_Origen',
    'Encabezado', 'Actor', 'Mencion', 'Tema', 'Tono', 'Origen',
    'Tipo_de_publicacion', 'Impulsado_por', 'Tiempo_aire', 'Ubicacion',
    'Equivalencia_publicitaria'
]
EXTRA_COLS = ['Pais', 'Municipio', 'Alcance', 'Testigo']

files = [
    'BD 2019.xlsx', 'BD 2020.xlsx', 'BD 2021.xlsx',
    'BD 2022.xlsx', 'BD 2023.xlsx', 'BDD 2024.xlsx'
]

frames = []
for f in files:
    filepath = os.path.join(DATA_DIR, f)
    print(f'Loading {f}...')
    df = pd.read_excel(filepath)
    
    # Normalize País/Pais column name
    col_rename = {}
    for col in df.columns:
        normalized = col.strip()
        if normalized in ['País', 'Pa\u00eds', 'Pais']:
            col_rename[col] = 'Pais'
    if col_rename:
        df = df.rename(columns=col_rename)
    
    for col in EXTRA_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    
    year_match = re.search(r'(\d{4})', f)
    df['Archivo_Origen'] = f
    df['Anio'] = int(year_match.group(1)) if year_match else None
    
    all_cols = CORE_COLS + EXTRA_COLS + ['Archivo_Origen', 'Anio']
    available = [c for c in all_cols if c in df.columns]
    df = df[available]
    frames.append(df)
    print(f'  -> {len(df)} rows')

print('\nConcatenating...')
df_all = pd.concat(frames, ignore_index=True)
print(f'Total before cleaning: {len(df_all):,} rows')

# --- Cleaning ---
# 1. Normalize Tono
df_all['Tono'] = df_all['Tono'].str.strip().str.capitalize()
tono_map = {'Positivo': 'Positivo', 'Negativo': 'Negativo', 'Neutral': 'Neutral', 'Neutro': 'Neutral'}
df_all['Tono'] = df_all['Tono'].map(tono_map).fillna('Neutral')

# 2. Clean headlines
df_all['Encabezado'] = df_all['Encabezado'].astype(str).str.strip()
mask_empty = (df_all['Encabezado'].isna()) | (df_all['Encabezado'] == '') | (df_all['Encabezado'] == 'nan')
df_all = df_all[~mask_empty].copy()

# 3. Parse dates
df_all['Fecha'] = pd.to_datetime(df_all['Fecha'], errors='coerce')
df_all = df_all.dropna(subset=['Fecha'])

# 4. Remove exact duplicates
n_before = len(df_all)
df_all = df_all.drop_duplicates(subset=['Fecha', 'Medio', 'Encabezado'], keep='first')
print(f'Duplicates removed: {n_before - len(df_all)}')

# 5. Normalize text fields
for col in ['Medio', 'Tipo', 'Cobertura', 'Estado_Origen', 'Tema', 'Actor', 'Mencion']:
    if col in df_all.columns:
        df_all[col] = df_all[col].astype(str).str.strip()

# 6. Temporal columns
df_all['Mes'] = df_all['Fecha'].dt.to_period('M').astype(str)
df_all['Semana'] = df_all['Fecha'].dt.to_period('W').astype(str)
df_all['DiaSemana'] = df_all['Fecha'].dt.day_name()
df_all['Trimestre'] = df_all['Fecha'].dt.to_period('Q').astype(str)

# --- Summary ---
print(f'\nFinal dataset: {len(df_all):,} rows')
print(f'Date range: {df_all["Fecha"].min()} to {df_all["Fecha"].max()}')
print(f'Tone distribution:\n{df_all["Tono"].value_counts()}')

df_all.to_csv(CONSOLIDATED_CSV, index=False, encoding='utf-8-sig')
print(f'\nSaved to: {CONSOLIDATED_CSV}')
