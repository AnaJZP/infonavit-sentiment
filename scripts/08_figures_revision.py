"""
figures_revision.py — Figuras 1-7 del manuscrito revisado (jrfm-4546218)
=======================================================================
Figs. 1, 4, 5, 6: recalculadas con el RRS corregido (ecs. 1-5, umbral 30).
Figs. 2, 3, 7:    mismos datos que la versión original, con etiquetas más
                  grandes y mayor resolución (Revisor 2).

Uso: python scripts/06_rrs_corrected.py && python scripts/07_revision_numbers.py && python scripts/08_figures_revision.py
Salida: figures/revision_r1/figN_*.png (300 dpi)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo
RES = os.path.join(BASE_DIR, "results", "revision_r1")
OUT = os.path.join(BASE_DIR, "figures", "revision_r1")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 17,
    "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.facecolor": "white",
})
C = {"neg": "#C0392B", "neu": "#2E86C1", "pos": "#239B56", "rrs": "#6C3483",
     "trend": "#1C2833", "grey": "#7F8C8D", "excl": "#D5D8DC"}
EXCL = (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-11-30"))

N = json.load(open(os.path.join(RES, "revision_numbers.json"), encoding="utf-8"))
base = pd.read_csv(os.path.join(RES, "rrs_base_revision.csv"), parse_dates=["Fecha"])
base["RRS100"] = base["RRS"] * 100
top_eps = N["top_episodes"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("  [OK]", name)


def shade_excluded(ax, label=False):
    ax.axvspan(*EXCL, color=C["excl"], alpha=0.6, lw=0,
               label="Excluded (incomplete delivery)" if label else None)


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ── Figure 1: daily RRS, volume, sentiment ───────────────────────────
s = base.set_index("Fecha")
full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
ma30 = s["RRS100"].reindex(full_idx).rolling(30, min_periods=15).mean()
fig, axes = plt.subplots(3, 1, figsize=(15, 10.7), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.4, 1.4]})
ax = axes[0]
shade_excluded(ax, label=True)
sg = s.reindex(full_idx)  # NaN en días sin dato: no se trazan líneas sobre huecos
ax.plot(sg.index, sg["RRS100"], color=C["grey"], lw=0.6, alpha=0.8, label="Daily RRS")
ax.plot(ma30.index, ma30, color=C["neu"], lw=2.2, label="30-day moving average")
for q, ls in zip([25, 50, 75], [":", "--", ":"]):
    ax.axhline(np.percentile(s["RRS100"], q), color=C["trend"], ls=ls, lw=1)
crisis = s[s["z"] > 2]
ax.scatter(crisis.index, crisis["RRS100"], color=C["neg"], s=22, zorder=3,
           label="Crisis day (z > 2)")
for e in top_eps:
    d = pd.Timestamp(e["peak_day"])
    ax.annotate(str(e["rank"]), (d, e["peak_rrs"]), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=14, fontweight="bold", color=C["neg"])
ax.set_ylabel("RRS (0–100)")
ax.set_ylim(15, 64)
ax.set_title("(a) Daily Reputational Risk Score (numbers mark the five highest-peak episodes)", loc="left")
ax.legend(loc="lower center", ncol=4, framealpha=0.95, fontsize=13)
ax = axes[1]
shade_excluded(ax)
ax.bar(s.index, s["n_notas"], width=1, color=C["neu"], alpha=0.7)
ax.set_ylabel("Articles")
ax.set_title("(b) Daily news volume (days with ≥ 30 articles)", loc="left")
ax = axes[2]
shade_excluded(ax)
ax.plot(sg.index, sg["mean_sent"], color=C["pos"], lw=0.6, alpha=0.8)
ax.axhline(0, color=C["trend"], lw=0.8)
ax.set_ylabel("Mean s")
ax.set_title("(c) Daily mean BERT sentiment score", loc="left")
date_axis(axes[2])
fig.tight_layout()
save(fig, "fig1_rrs_daily.png")

# ── Figure 4: monthly RRS ────────────────────────────────────────────
m = s["RRS100"].resample("MS").agg(["mean", "min", "max", "count"])
m.loc[m["count"] == 0, ["mean", "min", "max"]] = np.nan  # meses sin días válidos: hueco
fig, ax = plt.subplots(figsize=(15, 5.3))
shade_excluded(ax, label=True)
ax.fill_between(m.index, m["min"], m["max"], color=C["rrs"], alpha=0.15, step="mid",
                label="Monthly min–max")
ax.plot(m.index, m["mean"], color=C["rrs"], lw=2.4, marker="o", ms=5, label="Monthly mean")
ax.axhline(s["RRS100"].mean(), color=C["trend"], ls="--", lw=1, label="Sample mean")
ax.set_ylabel("RRS (0–100)")
date_axis(ax)
ax.legend(loc="upper left", ncol=4, framealpha=0.9)
fig.tight_layout()
save(fig, "fig4_rrs_monthly.png")

# ── Figure 5: STL (period 30) ────────────────────────────────────────
stl = pd.read_csv(os.path.join(RES, "stl_p30_x100.csv"), parse_dates=["Fecha"]).set_index("Fecha")
fig, axes = plt.subplots(4, 1, figsize=(15, 10.7), sharex=True)
for ax, col, title, colr in zip(axes, ["RRS", "Trend", "Seasonal", "Residual"],
                                ["Observed", "Trend", "Seasonal (period = 30)", "Residual"],
                                [C["grey"], C["trend"], C["neu"], C["neg"]]):
    shade_excluded(ax)
    ax.plot(stl.index, stl[col], color=colr, lw=0.9 if col != "Trend" else 2)
    ax.axvspan(pd.Timestamp("2022-10-01"), pd.Timestamp("2022-10-31"), color=C["excl"], alpha=0.6, lw=0)
    ax.set_ylabel(title, fontsize=14)
axes[3].axhline(0, color=C["trend"], lw=0.8)
date_axis(axes[3])
fig.tight_layout()
save(fig, "fig5_stl_p30.png")

# ── Figure 6: component correlation matrix ───────────────────────────
labels = ["C1\nSentiment", "C2\nVolume", "C3\nNegativity", "C4\nKeywords"]
corr = base[["C1", "C2", "C3", "C4"]].corr().values
fig, ax = plt.subplots(figsize=(9, 7.9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center", fontsize=17,
                color="white" if abs(corr[i, j]) > 0.6 else "black")
ax.set_xticks(range(4), labels)
ax.set_yticks(range(4), labels)
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
save(fig, "fig6_component_correlation.png")

# ── Datos a nivel artículo para Figs. 2, 3 y 7 ───────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, "data", "infonavit_nlp_results.csv"), encoding="utf-8-sig",
                 low_memory=False, usecols=["Tono", "NLP_Sentimiento", "Tema", "Estado_Origen", "Encabezado"])

# ── Figure 2: confusion matrix ───────────────────────────────────────
cats = ["Negativo", "Neutral", "Positivo"]
en = ["Negative", "Neutral", "Positive"]
cm = pd.crosstab(df["Tono"], df["NLP_Sentimiento"]).reindex(index=cats, columns=cats).values
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
for ax, mat, fmt, cmap, title in [(axes[0], cm, "{:,.0f}", "Blues", "(a) Counts"),
                                  (axes[1], cm_norm, "{:.1%}", "Oranges", "(b) Row-normalized")]:
    im = ax.imshow(mat, cmap=cmap)
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=18,
                    color="white" if v > mat.max() * 0.6 else "black")
    ax.set_xticks(range(3), en)
    ax.set_yticks(range(3), en)
    ax.set_xlabel("BERT prediction")
    ax.set_ylabel("Manual label (monitoring service)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[:].set_visible(False)
fig.tight_layout()
save(fig, "fig2_confusion_matrix.png")

# ── Figure 3: top-20 topics by sentiment (manual tone) ───────────────
# Traducción de categorías del proveedor (igual que en la versión original de la Figura 3)
TOPIC_EN = {
    'Aportaciones patronales': 'Employer Contributions',
    'Responsabilidad Compartida': 'Shared Responsibility',
    'Responsabilidad compartida': 'Shared Responsibility',
    'GOBIERNO': 'Government',
    'Colocación de créditos': 'Loan Placement',
    'Ferias': 'Housing Fairs',
    'Outsourcing': 'Outsourcing',
    'Reforma de ley': 'Legislative Reform',
    'Soluciones de pago': 'Payment Solutions',
    'Opciones de crédito': 'Credit Options',
    'CRÉDITOS': 'Loans',
    'Fraudes': 'Fraud',
    'Mi Cuenta Infonavit': 'My Infonavit Account',
    'Apoyos a damnificados': 'Disaster Relief',
    'Gobiernos Estatales': 'State Governments',
    'Mejoravit': 'Mejoravit',
    'Devolución de la SSV': 'SSV Fund Return',
    'Unamos Créditos': 'Combined Loans',
    'Seguro por daños': 'Damage Insurance',
    'Vivienda abandonada': 'Abandoned Housing',
    'TELRA': 'TELRA',
    'Liquidación anticipada': 'Early Payoff',
    'Crédito para la mujer': "Women's Credit",
    'Fondo de Pensiones para el Bienestar': 'Welfare Pension Fund',
    'Centro de atención': 'Service Center',
    'Desarrollos de vivienda': 'Housing Developments',
    'Vivienda económica': 'Affordable Housing',
    'PROGRAMAS': 'Programs',
    'Tasas de interés': 'Interest Rates',
    'Ética e Institucionalidad': 'Ethics and Governance',
}
df["Tema_EN"] = df["Tema"].map(lambda x: TOPIC_EN.get(x, x))
top_temas = df["Tema_EN"].value_counts().head(20).index
ts_ = df[df["Tema_EN"].isin(top_temas)].groupby(["Tema_EN", "Tono"]).size().unstack(fill_value=0)
ts_["pct_neg"] = ts_.get("Negativo", 0) / ts_[cats].sum(axis=1) * 100
ts_ = ts_.sort_values("pct_neg", ascending=True)
fig, ax = plt.subplots(figsize=(15, 10))
y = np.arange(len(ts_))
left = np.zeros(len(ts_))
for cat, lab, col in zip(cats, en, [C["neg"], C["neu"], C["pos"]]):
    ax.barh(y, ts_[cat], left=left, color=col, label=lab, alpha=0.85)
    left += ts_[cat].values
for yi, (tot, pn) in enumerate(zip(left, ts_["pct_neg"])):
    ax.text(tot, yi, f"  {pn:.1f}% neg.", va="center", fontsize=13)
ax.set_yticks(y, ts_.index, fontsize=15)
ax.set_xlabel("Number of articles")
ax.set_xlim(0, left.max() * 1.18)
ax.legend(loc="lower right", framealpha=0.95)
ax.grid(True, alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "fig3_topics_sentiment.png")

# ── Figure 7: geographic distribution (top 15 states) ────────────────
df["Estado_Origen"] = df["Estado_Origen"].replace({"CDMX": "Ciudad de México"})
st = df.groupby("Estado_Origen").agg(total=("Encabezado", "count"),
                                     neg=("Tono", lambda x: (x == "Negativo").sum())).reset_index()
st["pct_neg"] = st["neg"] / st["total"] * 100
st = st.sort_values("total", ascending=False).head(15)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
axes[0].barh(range(15), st["total"], color=C["neu"], alpha=0.85)
axes[0].set_title("(a) News volume by state", loc="left", fontweight="bold")
axes[0].set_xlabel("Number of articles")
mean_neg = st["pct_neg"].mean()
axes[1].barh(range(15), st["pct_neg"],
             color=[C["neg"] if v > mean_neg else C["neu"] for v in st["pct_neg"]], alpha=0.85)
axes[1].axvline(mean_neg, color="black", ls="--", lw=1.2)
axes[1].set_title(f"(b) Negative coverage rate (dashed = mean, {mean_neg:.1f}%)", loc="left", fontweight="bold")
axes[1].set_xlabel("% negative (manual label)")
for ax in axes:
    ax.set_yticks(range(15), st["Estado_Origen"], fontsize=15)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "fig7_geographic.png")
