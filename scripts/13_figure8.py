"""
figure8_revision.py — Figura 8 corregida (revisión R2)
=======================================================
El archivo insertado en el manuscrito mostraba la distribución de los scores
por año, mientras que el pie describe la distribución por categoría de tono
manual (señalado por el Revisor 1). Este script genera la figura que el pie
describe: violines de los scores continuos de BERT dentro de cada categoría
de tono asignada por los codificadores del proveedor.

Salida: figures_revision/fig8_violin_by_tone.png (300 dpi)
Uso:    python scripts/figure8_revision.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "figures", "revision_r2")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 17,
    "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 14,
    "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.facecolor": "white",
})
C = {"Negativo": "#C0392B", "Neutral": "#2E86C1", "Positivo": "#239B56"}
EN = {"Negativo": "Negative", "Neutral": "Neutral", "Positivo": "Positive"}

df = pd.read_csv(os.path.join(BASE, "data", "infonavit_nlp_results.csv"), encoding="utf-8-sig",
                 usecols=["NLP_Score"], low_memory=False)
df["Tono"] = pd.read_csv(os.path.join(BASE, "data", "infonavit_consolidated_v2.csv"),
                         encoding="utf-8-sig", usecols=["Tono"], low_memory=False)["Tono"].values
cats = ["Negativo", "Neutral", "Positivo"]
data = [df.loc[df["Tono"] == c, "NLP_Score"].dropna().values for c in cats]

fig, ax = plt.subplots(figsize=(14.5, 6))
parts = ax.violinplot(data, showextrema=False, widths=0.85)
for body, c in zip(parts["bodies"], cats):
    body.set_facecolor(C[c]); body.set_alpha(0.55); body.set_edgecolor(C[c]); body.set_linewidth(1.5)
bp = ax.boxplot(data, widths=0.12, showfliers=False, patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                boxprops=dict(facecolor="white", alpha=0.9))
for i, (c, vals) in enumerate(zip(cats, data), start=1):
    ax.text(i, 1.06, f"n = {len(vals):,}\nmean = {vals.mean():.2f}", ha="center", fontsize=14)
ax.axhline(0, color="#1C2833", lw=0.9, ls="--")
ax.set_xticks(range(1, 4), [EN[c] for c in cats])
ax.set_xlabel("Manual tone label (monitoring service)")
ax.set_ylabel("BERT sentiment score")
ax.set_ylim(-1.15, 1.25)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_violin_by_tone.png"), bbox_inches="tight")
print("[OK] fig8_violin_by_tone.png")

# Cifras citadas en el texto
for c, vals in zip(cats, data):
    q = np.percentile(vals, [25, 50, 75])
    print(f"{c}: n={len(vals):,} mean={vals.mean():.3f} median={q[1]:.3f} "
          f"IQR=[{q[0]:.3f}, {q[2]:.3f}] share_below_0={100*(vals<0).mean():.1f}%")
