"""
revision_numbers.py — Cifras para la revisión R1 de jrfm-4546218
=================================================================
Reproduce el RRS corregido de rrs_robustness.py (ecs. 1-5, umbral 30,
exclusión jun–nov 2020) y calcula todas las cifras reportadas en el
manuscrito revisado.

Entrada:  infonavit_consolidated.csv + infonavit_nlp_results.csv
Salida:   results/revision_r1/revision_numbers.json (+ rrs_base_revision.csv, stl_p30_x100.csv)
Uso:      python scripts/06_rrs_corrected.py && python scripts/07_revision_numbers.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, linregress
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo
DATA = os.path.join(BASE_DIR, "data")
RES = os.path.join(BASE_DIR, "results", "revision_r1")

# Misma configuración que rrs_robustness.py
W_BASE = (0.35, 0.25, 0.25, 0.15)
EXCL_START, EXCL_END = pd.Timestamp("2020-06-01"), pd.Timestamp("2020-11-30")
CRISIS_KEYWORDS = [
    "fraude", "corrupción", "corrupto", "escándalo", "denuncia",
    "demanda judicial", "demandado", "demandan",
    "irregularidad", "desvío", "robo", "abuso", "nepotismo",
    "conflicto", "protesta", "queja", "desalojo", "embargo",
    "crisis", "colapso", "impunidad", "investigación", "ilegal",
    "sanción", "multa", "pérdida", "deuda", "morosidad",
    "cartera vencida", "problema",
]
GRID_W, GRID_T, GRID_G = [7, 14, 21, 30], [1.5, 2.0, 2.5, 3.0], [1, 3, 5]
TOP_K = 5

# ── Carga ────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "infonavit_consolidated.csv"),
                 encoding="utf-8-sig", low_memory=False)
df["Fecha"] = pd.to_datetime(df["Fecha"], format="mixed")
nlp = pd.read_csv(os.path.join(DATA, "infonavit_nlp_results.csv"),
                  encoding="utf-8-sig", low_memory=False,
                  usecols=["Encabezado", "Tono", "NLP_Sentimiento", "NLP_Score", "Topic"])
assert (nlp["Encabezado"].fillna("").values == df["Encabezado"].fillna("").values).all()
for c in ["NLP_Sentimiento", "NLP_Score", "Topic"]:
    df[c] = nlp[c].values

daily_articles_all = df.groupby(df["Fecha"].dt.date).size()
df["es_crisis"] = df["Encabezado"].astype(str).str.lower().apply(
    lambda h: int(any(k in h for k in CRISIS_KEYWORDS)))
df = df[~((df["Fecha"] >= EXCL_START) & (df["Fecha"] <= EXCL_END))].copy()
df["Sent_Num"] = df["NLP_Score"].astype(float)
df["dia"] = df["Fecha"].dt.normalize()


def aggregate(d):
    out = d.groupby("dia").agg(
        n_notas=("Encabezado", "count"),
        mean_sent=("Sent_Num", "mean"),
        n_neg=("Tono", lambda x: (x == "Negativo").sum()),
        n_crisis=("es_crisis", "sum"),
    ).reset_index().rename(columns={"dia": "Fecha"})
    return out.sort_values("Fecha").reset_index(drop=True)


def components(d):
    d = d.copy()
    d["C1"] = (1.0 - d["mean_sent"]) / 2.0
    d["C2"] = norm.cdf((d["n_notas"] - d["n_notas"].mean()) / d["n_notas"].std())
    d["C3"] = d["n_neg"] / d["n_notas"]
    d["C4"] = d["n_crisis"] / d["n_notas"]
    return d


def rrs(d, w=W_BASE):
    return w[0] * d["C1"] + w[1] * d["C2"] + w[2] * d["C3"] + w[3] * d["C4"]


def detect(series, fechas, window=14, threshold=2.0, gap=3):
    s = series.reset_index(drop=True)
    f = fechas.reset_index(drop=True)
    m = s.shift(1).rolling(window, min_periods=window).mean()
    sd = s.shift(1).rolling(window, min_periods=window).std()
    z = (s - m) / (sd + 1e-8)
    mask = z > threshold
    idx = mask[mask].index.tolist()
    eps = []
    if idx:
        a = b = idx[0]
        for k in idx[1:]:
            if (f.iloc[k] - f.iloc[b]).days <= gap:
                b = k
            else:
                eps.append((a, b)); a = b = k
        eps.append((a, b))
    episodes = [{"start": f.iloc[a], "end": f.iloc[b],
                 "peak": float(s.iloc[a:b + 1].max()),
                 "peak_day": f.iloc[a + int(np.argmax(s.iloc[a:b + 1].values))]}
                for a, b in eps]
    return z, mask, episodes


def crisis_dates(fechas, mask):
    return set(fechas.reset_index(drop=True)[mask.values].dt.date)


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def covers(dates, ep, pad=1):
    rng = set(pd.date_range(ep["start"] - pd.Timedelta(days=pad),
                            ep["end"] + pd.Timedelta(days=pad)).date)
    return bool(dates & rng)


N = {}  # todas las cifras

# ── Serie base (umbral 30) y verificación contra rrs_robustness.py ──
daily_all = aggregate(df)
base = components(daily_all[daily_all["n_notas"] >= 30].reset_index(drop=True))
base["RRS"] = rrs(base)
ref = pd.read_csv(os.path.join(RES, "rrs_diario_u30.csv"), parse_dates=["Fecha"])
assert len(ref) == len(base) and np.allclose(ref["RRS"].values, base["RRS"].values), \
    "La serie no coincide con rrs_diario_u30.csv"

z, mask, eps = detect(base["RRS"], base["Fecha"])
base["z"] = z.values
base_dates = crisis_dates(base["Fecha"], mask)
top = sorted(eps, key=lambda e: e["peak"], reverse=True)[:TOP_K]

r100 = base["RRS"] * 100
N["corpus"] = {
    "articles_total": int(len(nlp)),
    "articles_after_exclusion": int(len(df)),
    "articles_excluded_block": int(len(nlp) - len(df)),
    "days_with_articles_after_exclusion": int(len(daily_all)),
    "days_base_case": int(len(base)),
    "days_dropped_threshold": int(len(daily_all) - len(base)),
    "mean_daily_articles_all": round(float(daily_articles_all.mean()), 1),
    "median_daily_articles_all": float(daily_articles_all.median()),
    "articles_in_base_days": int(base["n_notas"].sum()),
}
N["table5"] = {
    "mean": round(r100.mean(), 2), "median": round(r100.median(), 2),
    "sd": round(r100.std(), 2), "min": round(r100.min(), 2), "max": round(r100.max(), 2),
    "skew": round(float(r100.skew()), 3), "excess_kurtosis": round(float(r100.kurt()), 3),
    "p25": round(r100.quantile(.25), 2), "p75": round(r100.quantile(.75), 2),
    "p95": round(r100.quantile(.95), 2),
    "crisis_days": int(mask.sum()), "episodes": len(eps),
}
N["by_year_mean"] = base.groupby(base["Fecha"].dt.year)[["C1", "C2", "C3", "C4", "RRS"]] \
    .mean().mul(100).round(1).to_dict(orient="index")
N["component_corr"] = base[["C1", "C2", "C3", "C4"]].corr().round(3).to_dict()
N["component_means"] = base[["C1", "C2", "C3", "C4"]].mean().round(3).to_dict()

# ── STL (periodo 30 y 7) ─────────────────────────────────────────────
ts = base.set_index("Fecha")["RRS"].asfreq("D")
n_interp = int(ts.isna().sum())
ts = ts.interpolate("linear") * 100
N["stl"] = {"interpolated_days": n_interp, "total_days": int(len(ts))}
for p in [30, 7]:
    fit = STL(ts, period=p, robust=True).fit()
    obs = ts[base.set_index("Fecha").index]  # solo días observados para residuos
    res_obs = fit.resid[obs.index]
    N["stl"][f"p{p}"] = {
        "trend_min": round(fit.trend.min(), 2), "trend_max": round(fit.trend.max(), 2),
        "trend_max_date": str(fit.trend.idxmax().date()),
        "seasonal_min": round(fit.seasonal.min(), 2), "seasonal_max": round(fit.seasonal.max(), 2),
        "resid_max": round(res_obs.max(), 2), "resid_max_date": str(res_obs.idxmax().date()),
        "var_share_trend": round(fit.trend.var() / ts.var(), 3),
        "var_share_seasonal": round(fit.seasonal.var() / ts.var(), 3),
        "var_share_resid": round(fit.resid.var() / ts.var(), 3),
        "trend_by_year": fit.trend.groupby(fit.trend.index.year).mean().round(1).to_dict(),
    }
    if p == 30:
        stl30 = fit

# ── Top-5 episodios: contenido ───────────────────────────────────────
topic_names = pd.read_csv(os.path.join(BASE_DIR, "results", "bertopic_topics.csv"),
                          usecols=["Topic", "Name"]).set_index("Topic")["Name"].to_dict()
ep_rows = []
for rank, e in enumerate(top, 1):
    in_ep = (df["dia"] >= e["start"]) & (df["dia"] <= e["end"])
    arts = df[in_ep]
    peak_arts = df[df["dia"] == e["peak_day"]]
    topics = arts.loc[arts["Topic"] != -1, "Topic"].value_counts()
    brow = base[base["Fecha"] == e["peak_day"]].iloc[0]
    ep_rows.append({
        "rank": rank, "start": str(e["start"].date()), "end": str(e["end"].date()),
        "duration": (e["end"] - e["start"]).days + 1,
        "peak_day": str(e["peak_day"].date()), "peak_rrs": round(e["peak"] * 100, 2),
        "z_peak": round(float(brow["z"]), 2),
        "articles_peak_day": int(brow["n_notas"]),
        "articles_episode": int(len(arts)),
        "neg_manual_episode": int((arts["Tono"] == "Negativo").sum()),
        "neg_bert_episode": int((arts["NLP_Sentimiento"] == "Negativo").sum()),
        "C1": round(float(brow["C1"]), 3), "C2": round(float(brow["C2"]), 3),
        "C3": round(float(brow["C3"]), 3), "C4": round(float(brow["C4"]), 3),
        "top_headlines": peak_arts["Encabezado"].astype(str).str.strip()
            .value_counts().head(5).to_dict(),
        "top_topics": {topic_names.get(int(k), str(k)): int(v) for k, v in topics.head(3).items()},
    })
N["top_episodes"] = ep_rows

# ── Estabilidad de los top-5 en la malla de detección ────────────────
grid_hits = np.zeros(TOP_K, dtype=int)
grid_rows = []
n_cfg = 0
for w in GRID_W:
    for t in GRID_T:
        for g in GRID_G:
            _, m, _ = detect(base["RRS"], base["Fecha"], w, t, g)
            dts = crisis_dates(base["Fecha"], m)
            hits = np.array([covers(dts, e) for e in top])
            grid_hits += hits
            grid_rows.append({"w": w, "tau": t, "g": g, "crisis_days": int(m.sum()),
                              "episodes": len(detect(base["RRS"], base["Fecha"], w, t, g)[2]),
                              "top5_detected": int(hits.sum())})
            n_cfg += 1
N["grid"] = {"n_configs": n_cfg, "rows": grid_rows,
             "share_configs_detecting_top_episode": [round(h / n_cfg, 3) for h in grid_hits]}


def variant_summary(name, series, fechas):
    """Comparación de una variante del índice contra la serie base."""
    _, m, ep = detect(series, fechas)
    dts = crisis_dates(fechas, m)
    merged = pd.DataFrame({"Fecha": fechas.values, "v": series.values}).merge(
        base[["Fecha", "RRS"]], on="Fecha")
    return {
        "variant": name, "days": int(len(series)),
        "r_with_base": round(float(merged["v"].corr(merged["RRS"])), 3),
        "crisis_days": int(m.sum()), "episodes": len(ep),
        "jaccard_crisis_days": round(jaccard(dts, base_dates), 3),
        "top5_detected": int(sum(covers(dts, e) for e in top)),
    }


# ── Pesos alternativos (incluye PCA) ─────────────────────────────────
X = base[["C1", "C2", "C3", "C4"]]
Xs = (X - X.mean()) / X.std()
eigval, eigvec = np.linalg.eigh(np.cov(Xs.T))
pc1 = eigvec[:, np.argmax(eigval)]
pc1 = pc1 * np.sign(pc1.sum())
pca_w = np.abs(pc1) / np.abs(pc1).sum()
N["pca"] = {"loadings_pc1": dict(zip(X.columns, np.round(pc1, 3))),
            "explained_share_pc1": round(float(eigval.max() / eigval.sum()), 3),
            "weights_abs_normalized": dict(zip(X.columns, np.round(pca_w, 3)))}
schemes = {
    "Baseline (0.35, 0.25, 0.25, 0.15)": W_BASE,
    "Equal (0.25, 0.25, 0.25, 0.25)": (0.25, 0.25, 0.25, 0.25),
    "Sentiment-heavy (0.50, 0.20, 0.20, 0.10)": (0.50, 0.20, 0.20, 0.10),
    "Volume-heavy (0.25, 0.35, 0.25, 0.15)": (0.25, 0.35, 0.25, 0.15),
    "Sentiment-excluded (0.00, 0.40, 0.35, 0.25)": (0.00, 0.40, 0.35, 0.25),
    "PCA-based (%.2f, %.2f, %.2f, %.2f)" % tuple(pca_w): tuple(pca_w),
}
N["weights"] = []
for name, w in schemes.items():
    s = rrs(base, w)
    row = variant_summary(name, s, base["Fecha"])
    row.update({"mean": round(s.mean() * 100, 2), "sd": round(s.std() * 100, 2)})
    N["weights"].append(row)

# ── Umbral mínimo de notas ───────────────────────────────────────────
N["thresholds"] = []
for u in [30, 10, 1]:
    d = components(daily_all[daily_all["n_notas"] >= u].reset_index(drop=True))
    N["thresholds"].append(variant_summary(f"min {u} articles/day", rrs(d), d["Fecha"]))

# ── Medios: top-20 y titulares deduplicados ──────────────────────────
top20 = df.groupby("Medio").size().nlargest(20).index
d20 = aggregate(df[df["Medio"].isin(top20)])
d20 = components(d20[d20["n_notas"] >= 10].reset_index(drop=True))
N["media"] = [variant_summary("Top-20 outlets (min 10/day)", rrs(d20), d20["Fecha"])]
N["media"][0]["share_articles_top20"] = round(float(df["Medio"].isin(top20).mean()), 3)

dedup = df.assign(h=df["Encabezado"].astype(str).str.strip().str.lower()) \
    .drop_duplicates(subset=["dia", "h"])
N["media"].append({"syndication_share_removed": round(1 - len(dedup) / len(df), 3)})
dd = aggregate(dedup)
dd = components(dd[dd["n_notas"] >= 30].reset_index(drop=True))
N["media"].append(variant_summary("Unique headlines per day (min 30/day)", rrs(dd), dd["Fecha"]))

# ── Estabilidad temporal de los clasificadores ───────────────────────
mon = df.groupby(df["Fecha"].dt.to_period("M")).agg(
    bert_neg=("NLP_Sentimiento", lambda x: (x == "Negativo").mean() * 100),
    bert_score=("NLP_Score", "mean"),
    manual_neg=("Tono", lambda x: (x == "Negativo").mean() * 100),
).reset_index()
mon["t"] = (mon["Fecha"].dt.year * 12 + mon["Fecha"].dt.month)
mon["t"] -= mon["t"].min()  # meses calendario (respeta el hueco de 2020)
N["drift"] = {}
for col in ["bert_neg", "bert_score", "manual_neg"]:
    lr = linregress(mon["t"], mon[col])
    N["drift"][col] = {"mean": round(float(mon[col].mean()), 3),
                       "sd": round(float(mon[col].std()), 3),
                       "min": round(float(mon[col].min()), 3),
                       "max": round(float(mon[col].max()), 3),
                       "slope_per_month": round(lr.slope, 4), "p": round(lr.pvalue, 4),
                       "r2": round(lr.rvalue ** 2, 3), "months": int(len(mon))}
# ¿Sesgo BERT constante? tasa BERT-negativo dentro de notas manualmente no negativas
nn = df[df["Tono"] != "Negativo"]
mon2 = nn.groupby(nn["Fecha"].dt.to_period("M"))["NLP_Sentimiento"] \
    .apply(lambda x: (x == "Negativo").mean() * 100).reset_index()
mon2["t"] = mon2["Fecha"].dt.year * 12 + mon2["Fecha"].dt.month
lr = linregress(mon2["t"] - mon2["t"].min(), mon2["NLP_Sentimiento"])
N["drift"]["bert_neg_among_manual_nonneg"] = {
    "mean": round(float(mon2["NLP_Sentimiento"].mean()), 2),
    "slope_per_month": round(lr.slope, 4), "p": round(lr.pvalue, 4)}

# ── Contenido de días de crisis vs. resto (BERTopic, fuera del índice) ─
crisis_days_ts = pd.to_datetime(sorted(base_dates))
in_base = df["dia"].isin(base["Fecha"])
is_crisis = df["dia"].isin(crisis_days_ts)
topic_share = []
for tid in [3, 9, 10, 2, 1, 16, 0]:
    a = (df.loc[in_base & is_crisis, "Topic"] == tid).mean()
    b = (df.loc[in_base & ~is_crisis, "Topic"] == tid).mean()
    topic_share.append({"topic": topic_names.get(tid, str(tid)),
                        "share_crisis_days_pct": round(a * 100, 2),
                        "share_other_days_pct": round(b * 100, 2),
                        "ratio": round(a / b, 2) if b > 0 else None})
N["topic_enrichment"] = topic_share
N["bert_neg_share"] = {
    "crisis_days_pct": round(float((df.loc[in_base & is_crisis, "NLP_Sentimiento"] == "Negativo").mean() * 100), 1),
    "other_days_pct": round(float((df.loc[in_base & ~is_crisis, "NLP_Sentimiento"] == "Negativo").mean() * 100), 1),
}

# ── Cobertura mensual (para describir la exclusión) ──────────────────
full = pd.read_csv(os.path.join(DATA, "infonavit_consolidated.csv"),
                   encoding="utf-8-sig", usecols=["Fecha"])
fd = pd.to_datetime(full["Fecha"], format="mixed")
N["coverage_2020"] = full.groupby(fd.dt.to_period("M")).size().loc["2020-03":"2021-01"] \
    .rename(index=str).to_dict()
N["coverage_2022_10"] = int((fd.dt.to_period("M") == pd.Period("2022-10")).sum())
N["lexicon_hits_no_accents"] = int(df["es_crisis"].sum())

# ── Guardar ──────────────────────────────────────────────────────────
base.to_csv(os.path.join(RES, "rrs_base_revision.csv"), index=False)
pd.DataFrame({"Fecha": ts.index, "RRS": ts.values, "Trend": stl30.trend.values,
              "Seasonal": stl30.seasonal.values, "Residual": stl30.resid.values}) \
    .to_csv(os.path.join(RES, "stl_p30_x100.csv"), index=False)
with open(os.path.join(RES, "revision_numbers.json"), "w", encoding="utf-8") as f:
    json.dump(N, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(N, ensure_ascii=False, indent=1, default=str))
