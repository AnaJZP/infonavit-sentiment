"""
revision_r2_numbers.py — Análisis adicionales para la segunda ronda (jrfm-4626849)
===================================================================================
Responde con datos a los puntos del Revisor 2 (y al Revisor 1 donde aplica):

  A. Auditoría de etiquetas de tono: tasas de etiquetas corruptas/faltantes por
     año y por medio, leídas de los seis archivos originales.
  B. Especificaciones del índice sin componentes de sentimiento:
     sin C1, sin C3, y sin C1 ni C3 (solo volumen y palabras clave).
  C. C2 reconstruido solo con información pasada (expansiva y ventana móvil),
     ajustado por día de la semana y por tamaño del universo de medios.
  D. Detección: umbral empírico, z robusto (MAD), puntos de cambio,
     ventana en días calendario y efecto de incluir el día corriente.
  E. Composición de medios: universo por mes, duplicados por similitud,
     índice ponderado por medio e índices por cobertura y tipo de medio.
  F. Pesos: PCA con signos, doble conteo (traslape léxico-tópicos) y
     descomposición de la varianza del compuesto.
  G. Tópicos: coherencia NPMI de los tópicos BERTopic y concentración.
  H. Episodios: tabla completa de los 54 episodios con su contenido.

Entrada: infonavit_consolidated.csv, infonavit_nlp_results.csv, BD *.xlsx,
         results/bertopic_topics.csv y (si existe) results_robustez/
         spanish_models_scores.csv para la comparación de modelos.
Salida:  results_robustez/r2/*.csv y results_robustez/r2/r2_numbers.json
Uso:     python revision_r2_numbers.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import norm, linregress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "revision_r2")
os.makedirs(OUT, exist_ok=True)

W_BASE = (0.35, 0.25, 0.25, 0.15)
EXCL = (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-11-30"))
MIN_ARTICLES = 30
CRISIS_KEYWORDS = [
    "fraude", "corrupción", "corrupto", "escándalo", "denuncia",
    "demanda judicial", "demandado", "demandan",
    "irregularidad", "desvío", "robo", "abuso", "nepotismo",
    "conflicto", "protesta", "queja", "desalojo", "embargo",
    "crisis", "colapso", "impunidad", "investigación", "ilegal",
    "sanción", "multa", "pérdida", "deuda", "morosidad",
    "cartera vencida", "problema",
]
RAW_FILES = ["BD 2019.xlsx", "BD 2020.xlsx", "BD 2021.xlsx",
             "BD 2022.xlsx", "BD 2023.xlsx", "BDD 2024.xlsx"]
VALID_TONE = {"Positivo", "Negativo", "Neutral", "Neutro"}
N = {}

# ══════════════════════════════════════════════════════════════════
#  Datos base
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(BASE, "data", "infonavit_consolidated_v2.csv"),
                 encoding="utf-8-sig", low_memory=False)
df["Fecha"] = pd.to_datetime(df["Fecha"], format="mixed")
nlp = pd.read_csv(os.path.join(BASE, "data", "infonavit_nlp_results.csv"), encoding="utf-8-sig",
                  low_memory=False, usecols=["Encabezado", "NLP_Score", "NLP_Sentimiento", "Topic"])
assert (nlp["Encabezado"].fillna("").values == df["Encabezado"].fillna("").values).all()
for c in ["NLP_Score", "NLP_Sentimiento", "Topic"]:
    df[c] = nlp[c].values
df["head_l"] = df["Encabezado"].astype(str).str.strip().str.lower()
df["es_crisis"] = df["head_l"].apply(lambda h: int(any(k in h for k in CRISIS_KEYWORDS)))
df["dia"] = df["Fecha"].dt.normalize()

# ══════════════════════════════════════════════════════════════════
#  A. Auditoría de etiquetas de tono (R2: tasas por año y medio)
# ══════════════════════════════════════════════════════════════════
raw = []
for f in RAW_FILES:
    r = pd.read_excel(os.path.join(BASE, "data", "raw", f), usecols=["Fecha", "Medio", "Encabezado", "Tono"])
    r["archivo"] = f
    raw.append(r)
raw = pd.concat(raw, ignore_index=True)
raw["Fecha"] = pd.to_datetime(raw["Fecha"], errors="coerce")
tono_norm = raw["Tono"].astype(str).str.strip().str.capitalize()
raw["tono_valido"] = tono_norm.isin(VALID_TONE) & raw["Tono"].notna()
raw["anio"] = raw["Fecha"].dt.year

by_year = raw.groupby("anio").agg(registros=("Tono", "size"),
                                  validos=("tono_valido", "sum")).reset_index()
by_year["faltantes"] = raw.groupby("anio")["Tono"].apply(lambda s: s.isna().sum()).values
by_year["invalidos"] = by_year["registros"] - by_year["validos"]
by_year["pct_invalidos"] = (by_year["invalidos"] / by_year["registros"] * 100).round(1)
by_year.to_csv(os.path.join(OUT, "tono_auditoria_por_anio.csv"), index=False)

top_medios = raw["Medio"].value_counts().head(20).index
by_outlet = raw[raw["Medio"].isin(top_medios)].groupby("Medio").agg(
    registros=("Tono", "size"), validos=("tono_valido", "sum")).reset_index()
by_outlet["pct_invalidos"] = ((1 - by_outlet["validos"] / by_outlet["registros"]) * 100).round(1)
by_outlet = by_outlet.sort_values("registros", ascending=False)
by_outlet.to_csv(os.path.join(OUT, "tono_auditoria_por_medio.csv"), index=False)

ejemplos = tono_norm[~raw["tono_valido"] & raw["Tono"].notna()].value_counts().head(10)
N["tone_audit"] = {
    "raw_records": int(len(raw)),
    "invalid_total": int((~raw["tono_valido"]).sum()),
    "invalid_pct": round(float((~raw["tono_valido"]).mean() * 100), 1),
    "missing_total": int(raw["Tono"].isna().sum()),
    "by_year": by_year.to_dict("records"),
    "worst_outlets": by_outlet.nlargest(5, "pct_invalidos")[["Medio", "registros", "pct_invalidos"]].to_dict("records"),
    "example_invalid_values": {str(k): int(v) for k, v in ejemplos.items()},
    "note": "consolidate_data.py recodificó estos valores como 'Neutral' (fillna), "
            "no como faltantes; la sección 3.1.2 debe corregirse.",
}

# Marca de validez para cada nota del corpus consolidado (match por fecha+medio+encabezado)
raw["key"] = (raw["Fecha"].dt.strftime("%Y-%m-%d").fillna("") + "|" +
              raw["Medio"].astype(str).str.strip() + "|" +
              raw["Encabezado"].astype(str).str.strip())
valid_map = raw.groupby("key")["tono_valido"].max()
df["key"] = (df["Fecha"].dt.strftime("%Y-%m-%d") + "|" +
             df["Medio"].astype(str).str.strip() + "|" +
             df["Encabezado"].astype(str).str.strip())
df["tono_valido"] = df["Tono_valido"].astype(bool)   # de tone_recovery.py
N["tone_audit"]["corpus_matched_pct"] = round(float(df["key"].isin(valid_map.index).mean() * 100), 1)
N["tone_audit"]["corpus_valid_pct"] = round(float(df["tono_valido"].mean() * 100), 1)

# ══════════════════════════════════════════════════════════════════
#  Funciones del índice
# ══════════════════════════════════════════════════════════════════
d_all = df[~((df["Fecha"] >= EXCL[0]) & (df["Fecha"] <= EXCL[1]))].copy()


def aggregate(d, sent_col="NLP_Score", valid_only=False):
    g = d.groupby("dia")
    out = pd.DataFrame({
        "n_notas": g.size(),
        "mean_sent": g[sent_col].mean(),
        "n_neg": g["Tono"].apply(lambda x: (x == "Negativo").sum()),
        "n_valid": g["tono_valido"].sum(),
        "n_crisis": g["es_crisis"].sum(),
    })
    return out.reset_index().rename(columns={"dia": "Fecha"}).sort_values("Fecha").reset_index(drop=True)


def components(d, c3_denom=None):
    d = d.copy()
    d["C1"] = (1 - d["mean_sent"]) / 2
    d["C2"] = norm.cdf((d["n_notas"] - d["n_notas"].mean()) / d["n_notas"].std())
    denom = d[c3_denom] if c3_denom else d["n_valid"]
    d["C3"] = (d["n_neg"] / denom.replace(0, np.nan)).fillna(0)
    d["C4"] = d["n_crisis"] / d["n_notas"]
    return d


def rrs(d, w=W_BASE):
    return w[0] * d["C1"] + w[1] * d["C2"] + w[2] * d["C3"] + w[3] * d["C4"]


def detect(series, fechas, window=14, threshold=2.0, gap=3, include_current=False,
           robust=False, calendar=False):
    s = series.reset_index(drop=True).astype(float)
    f = pd.Series(pd.to_datetime(fechas)).reset_index(drop=True)
    if calendar:  # ventana de w días calendario
        ser = pd.Series(s.values, index=f.values)
        roll = ser.rolling(f"{window}D", closed="left")
        mu, sd = roll.mean().to_numpy(copy=True), roll.std().to_numpy(copy=True)
        cnt = roll.count().to_numpy()
        mu[cnt < 5], sd[cnt < 5] = np.nan, np.nan
    else:
        base = s if include_current else s.shift(1)
        if robust:
            med = base.rolling(window, min_periods=window).median()
            mad = base.rolling(window, min_periods=window).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True)
            mu, sd = med.values, (1.4826 * mad).values
        else:
            mu = base.rolling(window, min_periods=window).mean().values
            sd = base.rolling(window, min_periods=window).std().values
    z = (s.values - mu) / (sd + 1e-8)
    mask = pd.Series(z > threshold).fillna(False)
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
    episodes = [{"start": f.iloc[a], "end": f.iloc[b], "peak": float(s.iloc[a:b + 1].max()),
                 "peak_day": f.iloc[a + int(np.argmax(s.iloc[a:b + 1].values))]} for a, b in eps]
    return pd.Series(z), mask, episodes


def cdates(fechas, mask):
    return set(pd.Series(pd.to_datetime(fechas)).reset_index(drop=True)[mask.values].dt.date)


def jac(a, b):
    return round(len(a & b) / len(a | b), 3) if (a | b) else float("nan")


def covers(dates, ep, pad=1):
    return bool(dates & set(pd.date_range(ep["start"] - pd.Timedelta(days=pad),
                                          ep["end"] + pd.Timedelta(days=pad)).date))


daily_all = aggregate(d_all, valid_only=True)
base = components(daily_all[daily_all["n_notas"] >= MIN_ARTICLES].reset_index(drop=True))
base["RRS"] = rrs(base)
z0, m0, eps0 = detect(base["RRS"], base["Fecha"])
base["z"] = z0.values
base_dates = cdates(base["Fecha"], m0)
top5 = sorted(eps0, key=lambda e: e["peak"], reverse=True)[:5]
N["baseline"] = {"days": len(base), "crisis_days": int(m0.sum()), "episodes": len(eps0),
                 "mean": round(float(base["RRS"].mean() * 100), 2)}


def summarize(label, series, fechas, **kw):
    _, m, ep = detect(series, fechas, **kw)
    dts = cdates(fechas, m)
    merged = pd.DataFrame({"Fecha": pd.to_datetime(fechas).values, "v": np.asarray(series)}).merge(
        base[["Fecha", "RRS"]], on="Fecha")
    return {"variant": label, "days": int(len(series)),
            "r_with_base": round(float(merged["v"].corr(merged["RRS"])), 3),
            "crisis_days": int(m.sum()), "episodes": len(ep),
            "jaccard": jac(dts, base_dates),
            "top5_detected": int(sum(covers(dts, e) for e in top5)),
            "all54_detected": int(sum(covers(dts, e) for e in eps0))}


# ══════════════════════════════════════════════════════════════════
#  B. Especificaciones sin componentes de sentimiento
# ══════════════════════════════════════════════════════════════════
specs = []
specs.append(summarize("Baseline (0.35, 0.25, 0.25, 0.15)", base["RRS"], base["Fecha"]))
specs.append(summarize("Without C1 (0, 0.40, 0.35, 0.25)", rrs(base, (0, .40, .35, .25)), base["Fecha"]))
specs.append(summarize("Without C3 (0.45, 0.35, 0, 0.20)", rrs(base, (.45, .35, 0, .20)), base["Fecha"]))
specs.append(summarize("Without C1 and C3 (0, 0.625, 0, 0.375)", rrs(base, (0, .625, 0, .375)), base["Fecha"]))
specs.append(summarize("Volume only (C2)", base["C2"], base["Fecha"]))
specs.append(summarize("Crisis keywords only (C4)", base["C4"], base["Fecha"]))
# C3 calculado solo sobre etiquetas válidas
bv = components(daily_all[daily_all["n_notas"] >= MIN_ARTICLES].reset_index(drop=True), c3_denom="n_valid")
specs.append(summarize("C3 over all articles (labels as delivered)", rrs(components(daily_all[daily_all["n_notas"] >= MIN_ARTICLES].reset_index(drop=True), c3_denom="n_notas")), base["Fecha"]))
N["specs_no_sentiment"] = specs

# ══════════════════════════════════════════════════════════════════
#  C. C2 con información pasada, día de la semana y universo de medios
# ══════════════════════════════════════════════════════════════════
b = base.copy()
n = b["n_notas"].astype(float)
exp_mu, exp_sd = n.shift(1).expanding(90).mean(), n.shift(1).expanding(90).std()
b["C2_exp"] = norm.cdf((n - exp_mu) / exp_sd)
roll_mu, roll_sd = n.shift(1).rolling(365, min_periods=90).mean(), n.shift(1).rolling(365, min_periods=90).std()
b["C2_roll"] = norm.cdf((n - roll_mu) / roll_sd)
b["dow"] = b["Fecha"].dt.dayofweek
dow_stats = b.groupby("dow")["n_notas"].agg(["mean", "std"])
b["C2_dow"] = norm.cdf((n - b["dow"].map(dow_stats["mean"])) / b["dow"].map(dow_stats["std"]))
outlets = d_all.groupby("dia")["Medio"].nunique().reindex(b["Fecha"]).values
b["outlets"] = outlets
rate = n.values / np.maximum(outlets, 1)
b["C2_rate"] = norm.cdf((rate - pd.Series(rate).shift(1).expanding(90).mean()) /
                        pd.Series(rate).shift(1).expanding(90).std())
for col, lab in [("C2_exp", "C2 past-only (expanding)"), ("C2_roll", "C2 past-only (365-day)"),
                 ("C2_dow", "C2 day-of-week adjusted"), ("C2_rate", "C2 articles per active outlet")]:
    v = b.copy(); v["C2"] = b[col].fillna(b["C2"])
    specs_c2 = summarize(lab, rrs(v), v["Fecha"])
    N.setdefault("c2_variants", []).append(specs_c2)
mon_outlets = d_all.groupby(d_all["dia"].dt.to_period("M")).agg(
    outlets=("Medio", "nunique"), articles=("Medio", "size")).reset_index()
mon_outlets["articles_per_outlet"] = (mon_outlets["articles"] / mon_outlets["outlets"]).round(2)
mon_outlets["dia"] = mon_outlets["dia"].astype(str)
mon_outlets.to_csv(os.path.join(OUT, "universo_medios_mensual.csv"), index=False)
lr = linregress(range(len(mon_outlets)), mon_outlets["outlets"])
N["outlet_universe"] = {"outlets_first_month": int(mon_outlets["outlets"].iloc[0]),
                        "outlets_last_month": int(mon_outlets["outlets"].iloc[-1]),
                        "slope_outlets_per_month": round(lr.slope, 2), "p": round(lr.pvalue, 4),
                        "articles_per_outlet_first": float(mon_outlets["articles_per_outlet"].iloc[0]),
                        "articles_per_outlet_last": float(mon_outlets["articles_per_outlet"].iloc[-1])}

# ══════════════════════════════════════════════════════════════════
#  D. Reglas de detección alternativas
# ══════════════════════════════════════════════════════════════════
det = []
z_series = base["z"].dropna()
emp_q = float(np.nanquantile(z_series, 0.977))
_, m_emp, ep_emp = detect(base["RRS"], base["Fecha"], threshold=emp_q)
det.append({"rule": f"Empirical 97.7th percentile of z (τ = {emp_q:.2f})",
            "crisis_days": int(m_emp.sum()), "episodes": len(ep_emp),
            "jaccard": jac(cdates(base["Fecha"], m_emp), base_dates),
            "top5": int(sum(covers(cdates(base["Fecha"], m_emp), e) for e in top5))})
_, m_mad, ep_mad = detect(base["RRS"], base["Fecha"], robust=True)
det.append({"rule": "Robust z (median / MAD), τ = 2", "crisis_days": int(m_mad.sum()),
            "episodes": len(ep_mad), "jaccard": jac(cdates(base["Fecha"], m_mad), base_dates),
            "top5": int(sum(covers(cdates(base["Fecha"], m_mad), e) for e in top5))})
_, m_cal, ep_cal = detect(base["RRS"], base["Fecha"], calendar=True)
det.append({"rule": "14 calendar days (not valid observations)", "crisis_days": int(m_cal.sum()),
            "episodes": len(ep_cal), "jaccard": jac(cdates(base["Fecha"], m_cal), base_dates),
            "top5": int(sum(covers(cdates(base["Fecha"], m_cal), e) for e in top5))})
_, m_inc, ep_inc = detect(base["RRS"], base["Fecha"], include_current=True)
det.append({"rule": "Current day included in the baseline window", "crisis_days": int(m_inc.sum()),
            "episodes": len(ep_inc), "jaccard": jac(cdates(base["Fecha"], m_inc), base_dates),
            "top5": int(sum(covers(cdates(base["Fecha"], m_inc), e) for e in top5))})
algo = rpt.Pelt(model="rbf", min_size=14).fit(base["RRS"].values * 100)
cps = algo.predict(pen=20)[:-1]
cp_dates = [str(base["Fecha"].iloc[i].date()) for i in cps]
near = sum(any(abs((pd.Timestamp(c) - e["peak_day"]).days) <= 7 for c in cp_dates) for e in top5)
N["detection_rules"] = det
N["changepoints"] = {"n": len(cps), "dates": cp_dates, "top5_within_7_days": int(near),
                     "penalty": 20, "model": "rbf (Pelt)"}
N["z_distribution"] = {"empirical_q977": round(emp_q, 2),
                       "share_above_2": round(float((z_series > 2).mean() * 100), 2),
                       "normal_expectation_pct": 2.28,
                       "skew": round(float(z_series.skew()), 2),
                       "excess_kurtosis": round(float(z_series.kurt()), 2),
                       "lag1_autocorr_rrs": round(float(base["RRS"].autocorr(1)), 3),
                       "lag1_autocorr_z": round(float(z_series.autocorr(1)), 3)}

# ══════════════════════════════════════════════════════════════════
#  E. Composición de medios
# ══════════════════════════════════════════════════════════════════
# Duplicados por similitud semántica (TF-IDF de caracteres, dentro del mismo día)
dup_flag = pd.Series(False, index=d_all.index)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
for day, grp in d_all.groupby("dia"):
    if len(grp) < 2:
        continue
    try:
        X = vec.fit_transform(grp["head_l"])
    except ValueError:
        continue
    sim = cosine_similarity(X, dense_output=False)
    keep, seen = [], set()
    for i in range(sim.shape[0]):
        if i in seen:
            continue
        keep.append(i)
        row = sim.getrow(i).toarray().ravel()
        seen.update(np.where(row >= 0.85)[0].tolist())
    drop_idx = grp.index[[i for i in range(len(grp)) if i not in keep]]
    dup_flag.loc[drop_idx] = True
d_sem = d_all[~dup_flag]
N["dedup"] = {"exact_same_day_pct": round(float(d_all.duplicated(subset=["dia", "head_l"]).mean() * 100), 1),
              "semantic_085_pct": round(float(dup_flag.mean() * 100), 1),
              "articles_after_semantic_dedup": int(len(d_sem))}
ds = components(aggregate(d_sem).query(f"n_notas >= {MIN_ARTICLES}").reset_index(drop=True))
N.setdefault("media_composition", []).append(summarize("Semantic de-duplication (cos ≥ 0.85)", rrs(ds), ds["Fecha"]))

# Índice con igual peso por medio (promedio de medios, no de notas)
om = d_all.groupby(["dia", "Medio"]).agg(sent=("NLP_Score", "mean"),
                                         neg=("Tono", lambda x: (x == "Negativo").sum() / max(x.notna().sum(), 1)),
                                         cri=("es_crisis", "mean"), k=("Medio", "size")).reset_index()
ow = om.groupby("dia").agg(mean_sent=("sent", "mean"), neg_share=("neg", "mean"),
                           cri_share=("cri", "mean"), n_outlets=("Medio", "size"),
                           n_notas=("k", "sum")).reset_index().rename(columns={"dia": "Fecha"})
ow = ow[ow["n_notas"] >= MIN_ARTICLES].reset_index(drop=True)
ow["C1"] = (1 - ow["mean_sent"]) / 2
ow["C2"] = norm.cdf((ow["n_outlets"] - ow["n_outlets"].mean()) / ow["n_outlets"].std())
ow["C3"], ow["C4"] = ow["neg_share"], ow["cri_share"]
N["media_composition"].append(summarize("Outlet-weighted index (equal weight per outlet)", rrs(ow), ow["Fecha"]))

# Subconjuntos por cobertura y tipo de medio
subsets = {
    "National coverage only": d_all[d_all["Cobertura"].astype(str).str.strip() == "Nacional"],
    "State and local coverage": d_all[d_all["Cobertura"].astype(str).str.strip().isin(["Estatal", "Local"])],
    "Print outlets": d_all[d_all["Tipo"].astype(str).str.strip().isin(["Periódico", "Periódicos"])],
    "Digital outlets": d_all[d_all["Tipo"].astype(str).str.strip().isin(["Portal", "Portales"])],
}
for lab, sub in subsets.items():
    agg = aggregate(sub)
    agg = agg[agg["n_notas"] >= 10].reset_index(drop=True)
    s = components(agg)
    row = summarize(f"{lab} (≥ 10 articles/day)", rrs(s), s["Fecha"])
    row["share_of_articles"] = round(float(len(sub) / len(d_all) * 100), 1)
    N["media_composition"].append(row)

# ══════════════════════════════════════════════════════════════════
#  F. Pesos: PCA con signos, doble conteo y varianza del compuesto
# ══════════════════════════════════════════════════════════════════
X = base[["C1", "C2", "C3", "C4"]]
Z = (X - X.mean()) / X.std()
vals, vecs = np.linalg.eigh(np.cov(Z.T))
o = np.argsort(vals)[::-1]
vals, vecs = vals[o], vecs[:, o]
pc1 = vecs[:, 0] * np.sign(vecs[:, 0].sum())
pc1_scores = Z.values @ pc1
N["pca_signed"] = {
    "loadings": dict(zip(X.columns, np.round(pc1, 3))),
    "explained": [round(float(v / vals.sum()), 3) for v in vals],
    "index_corr_with_base": round(float(np.corrcoef(pc1_scores, base["RRS"])[0, 1]), 3),
}
pcs = pd.Series(pc1_scores)
N["pca_signed"].update(summarize("PC1 score (signed loadings)", pcs, base["Fecha"]))

w = np.array(W_BASE)
cov = np.cov(X.T)
contrib = {c: round(float(w[i] * np.sum(w * cov[i]) / (w @ cov @ w)), 3) for i, c in enumerate(X.columns)}
N["variance_contributions"] = contrib

kw_hits = d_all[d_all["es_crisis"] == 1]
N["double_counting"] = {
    "corr_C1_C3": round(float(base["C1"].corr(base["C3"])), 3),
    "corr_C1_C4": round(float(base["C1"].corr(base["C4"])), 3),
    "keyword_hits": int(len(kw_hits)),
    "share_of_keyword_hits_in_fraud_topics_pct": round(float(kw_hits["Topic"].isin([3, 16]).mean() * 100), 1),
    "share_of_keyword_hits_bert_negative_pct": round(float((kw_hits["NLP_Sentimiento"] == "Negativo").mean() * 100), 1),
    "share_all_articles_bert_negative_pct": round(float((d_all["NLP_Sentimiento"] == "Negativo").mean() * 100), 1),
}

# ══════════════════════════════════════════════════════════════════
#  G. Tópicos: coherencia NPMI
# ══════════════════════════════════════════════════════════════════
topics = pd.read_csv(os.path.join(BASE, "results", "bertopic_topics.csv"))
docs = d_all["head_l"].tolist()
doc_sets = [set(t.split()) for t in docs]
Ndoc = len(doc_sets)


def npmi_topic(words, k=10):
    words = [w for w in words[:k] if w and " " not in w]
    if len(words) < 2:
        return np.nan
    cnt = {w: sum(1 for s in doc_sets if w in s) for w in words}
    vals_ = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            wi, wj = words[i], words[j]
            if cnt[wi] == 0 or cnt[wj] == 0:
                continue
            co = sum(1 for s in doc_sets if wi in s and wj in s)
            if co == 0:
                vals_.append(-1.0); continue
            pij, pi, pj = co / Ndoc, cnt[wi] / Ndoc, cnt[wj] / Ndoc
            vals_.append(np.log(pij / (pi * pj)) / -np.log(pij))
    return float(np.mean(vals_)) if vals_ else np.nan


import ast
coh = []
for _, r in topics[topics["Topic"] >= 0].head(15).iterrows():
    try:
        words = ast.literal_eval(r["Representation"])
    except Exception:
        continue
    coh.append({"topic": int(r["Topic"]), "n": int(r["Count"]), "npmi": round(npmi_topic(words), 3)})
pd.DataFrame(coh).to_csv(os.path.join(OUT, "topic_coherence_npmi.csv"), index=False)
tot = topics["Count"].sum()
N["topics"] = {"npmi_mean_top15": round(float(np.nanmean([c["npmi"] for c in coh])), 3),
               "npmi_min": round(float(np.nanmin([c["npmi"] for c in coh])), 3),
               "npmi_max": round(float(np.nanmax([c["npmi"] for c in coh])), 3),
               "outlier_share_pct": round(float(topics.loc[topics["Topic"] == -1, "Count"].iloc[0] / tot * 100), 1),
               "largest_topic_share_of_classified_pct": round(
                   float(topics.loc[topics["Topic"] == 0, "Count"].iloc[0] /
                         topics.loc[topics["Topic"] >= 0, "Count"].sum() * 100), 1),
               "n_topics": int((topics["Topic"] >= 0).sum()),
               "per_topic": coh}

# ══════════════════════════════════════════════════════════════════
#  H. Los 54 episodios con su contenido
# ══════════════════════════════════════════════════════════════════
names = pd.read_csv(os.path.join(BASE, "results", "bertopic_topics.csv"),
                    usecols=["Topic", "Name"]).set_index("Topic")["Name"].to_dict()
rows = []
for i, e in enumerate(sorted(eps0, key=lambda x: x["peak"], reverse=True), 1):
    arts = d_all[(d_all["dia"] >= e["start"]) & (d_all["dia"] <= e["end"])]
    tp = arts.loc[arts["Topic"] != -1, "Topic"].value_counts()
    head = arts[arts["dia"] == e["peak_day"]]["Encabezado"].astype(str).str.strip().value_counts()
    rows.append({"rank": i, "start": str(e["start"].date()), "end": str(e["end"].date()),
                 "days": (e["end"] - e["start"]).days + 1, "peak_rrs": round(e["peak"] * 100, 2),
                 "articles": len(arts),
                 "neg_manual": int((arts["Tono"] == "Negativo").sum()),
                 "top_topic": names.get(int(tp.index[0]), "—") if len(tp) else "—",
                 "top_headline": head.index[0][:120] if len(head) else "—"})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "episodios_todos.csv"), index=False)
N["all_episodes_file"] = "results_robustez/r2/episodios_todos.csv"

# ══════════════════════════════════════════════════════════════════
#  I. Modelos en español (si spanish_models.py ya terminó)
# ══════════════════════════════════════════════════════════════════
sp_path = os.path.join(BASE, "results", "revision_r2", "spanish_models_scores.csv")
if os.path.exists(sp_path):
    sp = pd.read_csv(sp_path)
    for c in sp.columns:
        df[c] = sp[c].values
    d2 = df[~((df["Fecha"] >= EXCL[0]) & (df["Fecha"] <= EXCL[1]))].copy()
    man = d2["Tono"]
    res = {}
    from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
    lab_map = {"NEG": "Negativo", "NEU": "Neutral", "POS": "Positivo"}
    for key in ["beto", "robertuito"]:
        if f"{key}_label" not in d2:
            continue
        pred = d2[f"{key}_label"].str.upper().str[:3].map(lab_map)
        ok = pred.notna() & man.notna()
        res[key] = {
            "kappa_vs_manual": round(float(cohen_kappa_score(man[ok], pred[ok])), 4),
            "accuracy_vs_manual": round(float(accuracy_score(man[ok], pred[ok])), 3),
            "macro_f1_vs_manual": round(float(f1_score(man[ok], pred[ok], average="macro")), 3),
            "pct_negative": round(float((pred == "Negativo").mean() * 100), 1),
            "corr_score_with_nlptown": round(float(d2[f"{key}_score"].corr(d2["NLP_Score"])), 3),
            "agreement_with_nlptown_pct": round(float((pred == d2["NLP_Sentimiento"]).mean() * 100), 1),
        }
        # Índice recalculado con este modelo en C1
        dv = d2.copy()
        agg = aggregate(dv, sent_col=f"{key}_score")
        s = components(agg[agg["n_notas"] >= MIN_ARTICLES].reset_index(drop=True))
        res[key].update({"index_" + k: v for k, v in summarize(f"C1 from {key}", rrs(s), s["Fecha"]).items()})
        # Deriva mensual del sesgo
        mon = dv.groupby(dv["Fecha"].dt.to_period("M"))[f"{key}_label"].apply(
            lambda x: (x.str.upper().str[:3] == "NEG").mean() * 100)
        t = np.arange(len(mon))
        lr2 = linregress(t, mon.values)
        res[key].update({"monthly_neg_mean": round(float(mon.mean()), 1),
                         "drift_slope_pp_per_month": round(lr2.slope, 4), "drift_p": round(lr2.pvalue, 3)})
    kappa_models = None
    if {"beto_label", "robertuito_label"} <= set(d2.columns):
        kappa_models = round(float(cohen_kappa_score(d2["beto_label"].str.upper().str[:3],
                                                     d2["robertuito_label"].str.upper().str[:3])), 3)
    N["spanish_models"] = {"models": res, "kappa_beto_vs_robertuito": kappa_models}
else:
    N["spanish_models"] = "pendiente: ejecutar spanish_models.py"

base.to_csv(os.path.join(OUT, "rrs_base_r2.csv"), index=False)
with open(os.path.join(OUT, "r2_numbers.json"), "w", encoding="utf-8") as f:
    json.dump(N, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(N, ensure_ascii=False, indent=1, default=str)[:6000])
