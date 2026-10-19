"""
spanish_models.py — Sentimiento con modelos de dominio español (revisión R2)
============================================================================
El Revisor 2 pide comparar el modelo multilingüe usado en el artículo
(nlptown/bert-base-multilingual-uncased-sentiment) con modelos entrenados
en español (BETO, RoBERTuito).

Clasifica los 138,789 encabezados con:
  - finiteautomata/beto-sentiment-analysis      (BETO, español)
  - pysentimiento/robertuito-sentiment-analysis (RoBERTuito, español)

Para cada modelo guarda la etiqueta (NEG/NEU/POS) y un score continuo
s = P(POS) − P(NEG) ∈ [−1, 1], comparable con NLP_Score del artículo.

Salida: results_robustez/spanish_models_scores.csv
Uso:    python scripts/spanish_models.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "results", "revision_r2", "spanish_models_scores.csv")
MODELS = {
    "beto": "finiteautomata/beto-sentiment-analysis",
    "robertuito": "pysentimiento/robertuito-sentiment-analysis",
}
BATCH = 128
MAX_LEN = 96

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device} | torch {torch.__version__}", flush=True)

df = pd.read_csv(os.path.join(BASE_DIR, "data", "infonavit_consolidated.csv"),
                 encoding="utf-8-sig", usecols=["Encabezado"], low_memory=False)
texts = df["Encabezado"].astype(str).str.strip().tolist()
print(f"{len(texts):,} encabezados", flush=True)

# Ordenar por longitud acelera el padding; se restaura el orden al final
order = np.argsort([len(t) for t in texts])
out = pd.DataFrame(index=range(len(texts)))

for key, name in MODELS.items():
    t0 = time.time()
    print(f"\n== {key}: {name}", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    id2label = {int(k): v.upper() for k, v in model.config.id2label.items()}
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            batch = [texts[i] for i in idx]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
            if start % (BATCH * 100) == 0:
                done = start + len(idx)
                el = time.time() - t0
                print(f"  {done:,}/{len(texts):,}  {el/60:.1f} min "
                      f"(ETA {el/max(done,1)*(len(texts)-done)/60:.1f} min)", flush=True)

    cols = {id2label[i]: probs[:, i] for i in range(probs.shape[1])}
    pos = next(v for k, v in cols.items() if k.startswith("POS"))
    neg = next(v for k, v in cols.items() if k.startswith("NEG"))
    out[f"{key}_score"] = pos - neg
    out[f"{key}_label"] = [id2label[i] for i in probs.argmax(axis=1)]
    out[f"{key}_p_neg"] = neg
    print(f"  listo en {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_csv(OUT, index=False)
print(f"\nGuardado: {OUT}", flush=True)
