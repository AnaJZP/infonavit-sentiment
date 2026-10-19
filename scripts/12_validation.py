"""
validation_v2.py — Validación y cifras descriptivas con las etiquetas corregidas
=================================================================================
Recalcula, sobre las etiquetas recuperadas por tone_recovery.py y únicamente
en los registros con etiqueta válida:
  - matriz de confusión BERT vs. codificación humana, kappa, exactitud y F1
  - distribución de tono del corpus (Tabla 1) y comparación BERT/manual (Tabla 2)
  - negatividad por estado y por tipo de medio (Sección 4.6)

Salida: results_robustez/validation_v2.json
Uso:    python scripts/validation_v2.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = ["Negativo", "Neutral", "Positivo"]

df = pd.read_csv(os.path.join(BASE, "data", "infonavit_consolidated_v2.csv"), encoding="utf-8-sig",
                 low_memory=False, usecols=["Tono", "Tono_valido", "Tipo", "Cobertura",
                                            "Estado_Origen", "Encabezado"])
nlp = pd.read_csv(os.path.join(BASE, "data", "infonavit_nlp_results.csv"), encoding="utf-8-sig",
                  low_memory=False, usecols=["NLP_Sentimiento", "NLP_Score"])
df["NLP_Sentimiento"] = nlp["NLP_Sentimiento"].values
df["NLP_Score"] = nlp["NLP_Score"].values
v = df[df["Tono_valido"]].copy()
N = {"n_total": int(len(df)), "n_valid": int(len(v)),
     "valid_pct": round(float(len(v) / len(df) * 100), 1)}

# ── Tabla 1 y Tabla 2 ────────────────────────────────────────────────
N["manual_distribution"] = {c: {"n": int((v["Tono"] == c).sum()),
                                "pct_of_valid": round(float((v["Tono"] == c).mean() * 100), 1)}
                            for c in CATS}
N["bert_distribution_all"] = {c: {"n": int((df["NLP_Sentimiento"] == c).sum()),
                                  "pct": round(float((df["NLP_Sentimiento"] == c).mean() * 100), 1)}
                              for c in CATS}
N["bert_distribution_valid"] = {c: {"n": int((v["NLP_Sentimiento"] == c).sum()),
                                    "pct": round(float((v["NLP_Sentimiento"] == c).mean() * 100), 1)}
                                for c in CATS}
N["nlp_score_stats"] = {"mean": round(float(df["NLP_Score"].mean()), 3),
                        "median": round(float(df["NLP_Score"].median()), 3),
                        "sd": round(float(df["NLP_Score"].std()), 3)}

# ── Tabla 3 y métricas de acuerdo ────────────────────────────────────
cm = confusion_matrix(v["Tono"], v["NLP_Sentimiento"], labels=CATS)
N["confusion_manual_rows_bert_cols"] = {CATS[i]: {CATS[j]: int(cm[i, j]) for j in range(3)}
                                        for i in range(3)}
N["confusion_row_pct"] = {CATS[i]: {CATS[j]: round(float(cm[i, j] / cm[i].sum() * 100), 1)
                                    for j in range(3)} for i in range(3)}
p, r, f1, sup = precision_recall_fscore_support(v["Tono"], v["NLP_Sentimiento"], labels=CATS,
                                                zero_division=0)
N["agreement"] = {
    "kappa": round(float(cohen_kappa_score(v["Tono"], v["NLP_Sentimiento"])), 4),
    "accuracy_pct": round(float(accuracy_score(v["Tono"], v["NLP_Sentimiento"]) * 100), 1),
    "macro_f1": round(float(f1_score(v["Tono"], v["NLP_Sentimiento"], average="macro")), 3),
    "per_class_f1": {c: round(float(x), 3) for c, x in zip(CATS, f1)},
    "per_class_precision": {c: round(float(x), 3) for c, x in zip(CATS, p)},
    "per_class_recall": {c: round(float(x), 3) for c, x in zip(CATS, r)},
}
# Comparación con los valores publicados (etiquetas sin recuperar)
old = pd.read_csv(os.path.join(BASE, "data", "infonavit_consolidated.csv"), encoding="utf-8-sig",
                  usecols=["Tono"], low_memory=False)["Tono"]
N["agreement_previous_labels"] = {
    "kappa": round(float(cohen_kappa_score(old, df["NLP_Sentimiento"])), 4),
    "accuracy_pct": round(float(accuracy_score(old, df["NLP_Sentimiento"]) * 100), 1),
    "macro_f1": round(float(f1_score(old, df["NLP_Sentimiento"], average="macro")), 3),
}

# ── Sección 4.6: estados y tipos de medio ────────────────────────────
v["Estado_Origen"] = v["Estado_Origen"].replace({"CDMX": "Ciudad de México"})
df["Estado_Origen"] = df["Estado_Origen"].replace({"CDMX": "Ciudad de México"})
top_states = df["Estado_Origen"].value_counts().head(15)
st = v[v["Estado_Origen"].isin(top_states.index)].groupby("Estado_Origen").agg(
    valid=("Tono", "size"), neg=("Tono", lambda x: (x == "Negativo").sum())).reset_index()
st["pct_neg"] = (st["neg"] / st["valid"] * 100).round(1)
st = st.sort_values("pct_neg", ascending=False)
N["states"] = {"share_cdmx_pct": round(float(top_states.iloc[0] / len(df) * 100), 1),
               "top_shares_pct": {k: round(float(x / len(df) * 100), 1)
                                  for k, x in top_states.head(4).items()},
               "mean_pct_neg_top15": round(float(st["pct_neg"].mean()), 1),
               "ranking": st[["Estado_Origen", "pct_neg"]].to_dict("records")}

grp = {"Digital": ["Portal", "Portales"], "Print": ["Periódico", "Periódicos"]}
N["media_type"] = {}
for lab, vals in grp.items():
    m_all = df["Tipo"].astype(str).str.strip().isin(vals)
    m_val = v["Tipo"].astype(str).str.strip().isin(vals)
    N["media_type"][lab] = {
        "share_of_corpus_pct": round(float(m_all.mean() * 100), 1),
        "manual_neg_pct": round(float((v.loc[m_val, "Tono"] == "Negativo").mean() * 100), 1),
        "bert_neg_pct": round(float((df.loc[m_all, "NLP_Sentimiento"] == "Negativo").mean() * 100), 1),
    }

json.dump(N, open(os.path.join(BASE, "results", "revision_r2", "validation_v2.json"), "w",
                  encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(N, ensure_ascii=False, indent=1))
