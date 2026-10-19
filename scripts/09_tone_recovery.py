"""
tone_recovery.py — Recuperación de etiquetas de tono (revisión R2)
===================================================================
El script original de consolidación mapeaba solo {Positivo, Negativo,
Neutral, Neutro} y enviaba todo lo demás a 'Neutral' mediante fillna.
Eso convirtió en neutrales 15,729 notas que sí tenían etiqueta
(variantes en femenino "Positiva"/"Negativa") y ocultó ~10% de registros
cuyo campo de tono contiene cadenas de tema (desfase de columnas).

Este script relee los seis archivos originales y reconstruye la etiqueta:
  - "Positiva"/"Positivo" → Positivo;  "Negativa"/"Negativo" → Negativo
  - "Neutral"/"Neutro"/"Neutra"        → Neutral
  - cualquier otro valor (cadenas de tema) → faltante

Salida: infonavit_consolidated_v2.csv  (mismas columnas; Tono corregido y
        nueva columna Tono_valido) y results_robustez/tone_recovery.json
Uso:    python scripts/tone_recovery.py
"""

import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILES = ["BD 2019.xlsx", "BD 2020.xlsx", "BD 2021.xlsx",
             "BD 2022.xlsx", "BD 2023.xlsx", "BDD 2024.xlsx"]
TONE_MAP = {"positivo": "Positivo", "positiva": "Positivo",
            "negativo": "Negativo", "negativa": "Negativo",
            "neutral": "Neutral", "neutro": "Neutral", "neutra": "Neutral"}
OUT_CSV = os.path.join(BASE, "data", "infonavit_consolidated_v2.csv")
OUT_JSON = os.path.join(BASE, "results", "revision_r2", "tone_recovery.json")


def key(fecha, medio, head):
    return (pd.to_datetime(fecha, errors="coerce").dt.strftime("%Y-%m-%d").fillna("") + "|" +
            medio.astype(str).str.strip() + "|" + head.astype(str).str.strip())


raw = pd.concat([pd.read_excel(os.path.join(BASE, "data", "raw", f),
                               usecols=["Fecha", "Medio", "Encabezado", "Tono"]).assign(archivo=f)
                 for f in RAW_FILES], ignore_index=True)
raw["tono_fix"] = raw["Tono"].astype(str).str.strip().str.lower().map(TONE_MAP)
raw["key"] = key(raw["Fecha"], raw["Medio"], raw["Encabezado"])
lookup = raw.dropna(subset=["tono_fix"]).drop_duplicates("key").set_index("key")["tono_fix"]

df = pd.read_csv(os.path.join(BASE, "data", "infonavit_consolidated.csv"),
                 encoding="utf-8-sig", low_memory=False)
df["Fecha"] = pd.to_datetime(df["Fecha"], format="mixed")
df["key"] = key(df["Fecha"], df["Medio"], df["Encabezado"])
fixed = df["key"].map(lookup)

stats = {
    "raw_records": int(len(raw)),
    "raw_recoverable": int(raw["tono_fix"].notna().sum()),
    "raw_topic_strings": int(raw["tono_fix"].isna().sum()),
    "corpus_records": int(len(df)),
    "before": df["Tono"].value_counts().to_dict(),
    "after": {k: int(v) for k, v in fixed.value_counts().items()},
    "after_missing": int(fixed.isna().sum()),
    "after_missing_pct": round(float(fixed.isna().mean() * 100), 1),
    "relabelled": int(((df["Tono"] != fixed) & fixed.notna()).sum()),
    "neutral_to_negative": int(((df["Tono"] == "Neutral") & (fixed == "Negativo")).sum()),
    "neutral_to_positive": int(((df["Tono"] == "Neutral") & (fixed == "Positivo")).sum()),
}
valid = fixed.notna()
stats["after_shares_pct"] = {k: round(float(v / valid.sum() * 100), 1)
                             for k, v in fixed[valid].value_counts().items()}
by_year = pd.DataFrame({"anio": df["Fecha"].dt.year, "valido": valid}).groupby("anio")["valido"]
stats["valid_share_by_year_pct"] = {int(k): round(float(v * 100), 1) for k, v in by_year.mean().items()}

df["Tono"] = fixed
df["Tono_valido"] = valid
df.drop(columns=["key"]).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(stats, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=1))
print(f"\n→ {OUT_CSV}")
