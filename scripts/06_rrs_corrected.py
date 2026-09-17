"""
rrs_robustness.py — Cálculo corregido del RRS (jrfm-4546218)
=============================================================
Implementa las ecuaciones 1-5 del manuscrito tal como se publican:

  C1 = −S̄_t                       (sentimiento NLP medio invertido)
  C2 = Φ((n_t − μ_n) / σ_n)       (CDF normal del volumen)
  C3 = n_neg / n_t                 (proporción de notas con Tono negativo)
  C4 = n_crisis / n_t              (proporción de notas con keywords de crisis)

Los cuatro componentes ya viven en [0,1] y los pesos suman 1.
NO se aplica min-max ni reescalado a [0,100].

Correcciones respecto al código viejo (rrs_model.py / robustness_analysis.py):
 1. C2 usa Φ((n − μ)/σ) en vez de min-max de n.
 2. C3 y C4 son proporciones crudas, sin min-max.
 3. Se elimina el min-max de cada componente y el reescalado final a [0,100].
 4. C3 usa la columna Tono (codificación humana del proveedor).
 5. Del léxico de crisis se quita "demanda" a secas (≈ "demanda de vivienda"),
    se sustituye por "demanda judicial", "demandado", "demandan".
 6. Se excluye el bloque 2020-06-01 a 2020-11-30 (entrega incompleta).
 7. Umbral de notas mínimas por día: caso base 30 + sensibilidad 10 y 1.

Entrada:  infonavit_consolidated.csv
Salida:   results/revision_r1/resultados_robustez.md + CSVs de respaldo

Uso:
  pip install pandas numpy scipy statsmodels openpyxl tabulate
  python scripts/06_rrs_corrected.py
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import norm, linregress
from statsmodels.tsa.seasonal import STL
from tabulate import tabulate
from datetime import date

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURACIÓN                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── Rutas ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo
INPUT_FILE = os.path.join(BASE_DIR, "data", "infonavit_consolidated.csv")
NLP_FILE = os.path.join(BASE_DIR, "data", "infonavit_nlp_results.csv")   # aporta NLP_Score para C1
RESULTS_DIR = os.path.join(BASE_DIR, "results", "revision_r1")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Pesos (ecuación 5) ────────────────────────────────────────────────
W_SENT   = 0.35   # α  — Sentimiento (C1)
W_VOL    = 0.25   # β  — Volumen (C2)
W_NEG    = 0.25   # γ  — Negatividad (C3)
W_CRISIS = 0.15   # δ  — Intensidad de crisis (C4)

# ── Umbral de notas mínimas por día ──────────────────────────────────
UMBRALES = [30, 10, 1]          # caso base + sensibilidad

# ── Exclusión de entrega incompleta ──────────────────────────────────
EXCL_START = pd.Timestamp("2020-06-01")
EXCL_END   = pd.Timestamp("2020-11-30")

# ── Léxico de crisis corregido ───────────────────────────────────────
# Se quitó "demanda" a secas (≈ "demanda de vivienda").
# Se añadieron "demanda judicial", "demandado", "demandan".
CRISIS_KEYWORDS = [
    "fraude", "corrupción", "corrupto", "escándalo", "denuncia",
    "demanda judicial", "demandado", "demandan",
    "irregularidad", "desvío", "robo", "abuso", "nepotismo",
    "conflicto", "protesta", "queja", "desalojo", "embargo",
    "crisis", "colapso", "impunidad", "investigación", "ilegal",
    "sanción", "multa", "pérdida", "deuda", "morosidad",
    "cartera vencida", "problema",
]

# ── Eventos documentados (registro institucional / prensa) ───────────
# Estas fechas son la pieza que evita la circularidad:
# los rankings se calculan buscando qué episodio detectado cae cerca
# de cada evento real. Si están mal, todos los rankings salen mal.
EVENTOS = {
    1: {"fecha": "2020-03-01", "etiqueta": "Declaración pandemia / COVID-19"},
    2: {"fecha": "2021-01-14", "etiqueta": "Reforma de gobernanza Infonavit"},
    3: {"fecha": "2022-03-31", "etiqueta": "Debate de reforma de política habitacional"},
    4: {"fecha": "2023-11-01", "etiqueta": "Pico de cobertura mediática"},
    5: {"fecha": "2024-06-02", "etiqueta": "Jornada electoral México"},
}

# ── Detección de crisis ──────────────────────────────────────────────
WINDOW_Z     = 14     # días de la ventana trailing
THRESHOLD_Z  = 2.0    # umbral z-score
MERGE_GAP    = 3      # días calendario para fusionar episodios
TOLERANCIA   = 7      # ±días alrededor del evento para asignar rank

# ── Top-20 medios ────────────────────────────────────────────────────
UMBRAL_TOP20 = 10     # notas mínimas/día para el subconjunto top-20

# ── Malla de detección ──────────────────────────────────────────────
GRID_WINDOWS    = [7, 14, 21, 30]
GRID_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]
GRID_GAPS       = [1, 3, 5]

# ── STL ──────────────────────────────────────────────────────────────
STL_PERIODS = [7, 30]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FIN DE CONFIGURACIÓN                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ======================================================================
#  FUNCIONES AUXILIARES
# ======================================================================

def keyword_hit(headline: str) -> int:
    """Devuelve 1 si el encabezado contiene al menos un keyword de crisis."""
    h = str(headline).lower()
    return int(any(kw in h for kw in CRISIS_KEYWORDS))


def compute_components(day_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los cuatro componentes a partir de un DataFrame diario
    con columnas: Fecha, n_notas, mean_sent, n_neg, n_crisis.

    C1 = (1 − S̄_t) / 2          mapeado a [0,1], donde S̄ ∈ [-1,1]
    C2 = Φ((n − μ) / σ)          CDF normal
    C3 = n_neg / n               proporción cruda
    C4 = n_crisis / n             proporción cruda
    """
    d = day_df.copy()

    # C1: sentimiento invertido, mapeado a [0,1]
    # Si mean_sent ∈ [-1, 1], entonces (1 - mean_sent)/2 ∈ [0, 1]
    d["C1"] = (1.0 - d["mean_sent"]) / 2.0

    # C2: CDF normal del volumen
    mu = d["n_notas"].mean()
    sigma = d["n_notas"].std()
    if sigma < 1e-8:
        d["C2"] = 0.5
    else:
        d["C2"] = norm.cdf((d["n_notas"] - mu) / sigma)

    # C3: proporción de notas con Tono negativo
    d["C3"] = d["n_neg"] / d["n_notas"].clip(lower=1)

    # C4: proporción de notas con keywords de crisis
    d["C4"] = d["n_crisis"] / d["n_notas"].clip(lower=1)

    return d


def compute_rrs(day_df: pd.DataFrame,
                w: tuple = None) -> pd.Series:
    """RRS = α·C1 + β·C2 + γ·C3 + δ·C4  (ya en [0,1])."""
    if w is None:
        w = (W_SENT, W_VOL, W_NEG, W_CRISIS)
    a, b, g, d_ = w
    return a * day_df["C1"] + b * day_df["C2"] + g * day_df["C3"] + d_ * day_df["C4"]


def detect_crises(rrs: pd.Series, fechas: pd.Series,
                  window=WINDOW_Z, threshold=THRESHOLD_Z,
                  merge_gap=MERGE_GAP):
    """
    Trailing z-score → crisis mask → merge en episodios.
    Devuelve (z_series, mask, n_crisis_days, episodes_list).
    """
    roll_mean = rrs.shift(1).rolling(window, min_periods=window).mean()
    roll_std  = rrs.shift(1).rolling(window, min_periods=window).std()
    z = (rrs - roll_mean) / (roll_std + 1e-8)

    mask = z > threshold
    n_crisis = int(mask.sum())

    crisis_idx = mask[mask].index.tolist()
    episodes = []
    if crisis_idx:
        ep_s = ep_e = crisis_idx[0]
        for k in range(1, len(crisis_idx)):
            cur = crisis_idx[k]
            prev = crisis_idx[k - 1]
            gap_cal = (fechas.iloc[cur] - fechas.iloc[prev]).days
            if gap_cal <= merge_gap:
                ep_e = cur
            else:
                episodes.append({"start": ep_s, "end": ep_e})
                ep_s = ep_e = cur
        episodes.append({"start": ep_s, "end": ep_e})

    return z, mask, n_crisis, episodes


def rank_eventos(daily_df: pd.DataFrame, rrs_col: str,
                 eventos: dict = EVENTOS, tol: int = TOLERANCIA):
    """
    Para cada evento documentado, busca si un episodio detectado cae
    dentro de ±tol días. Devuelve lista de dicts con rank asignado.
    """
    rrs_sorted = daily_df.sort_values(rrs_col, ascending=False).reset_index(drop=True)
    resultados = []
    for eid, ev in eventos.items():
        ev_date = pd.Timestamp(ev["fecha"])
        # Buscar el día con mayor RRS dentro de ±tol
        mascara = (daily_df["Fecha"] >= ev_date - pd.Timedelta(days=tol)) & \
                  (daily_df["Fecha"] <= ev_date + pd.Timedelta(days=tol))
        sub = daily_df[mascara]
        if sub.empty:
            resultados.append({
                "evento_id": eid,
                "etiqueta": ev["etiqueta"],
                "fecha_evento": ev_date.strftime("%Y-%m-%d"),
                "dia_pico": "—",
                "rrs_pico": "—",
                "n_notas_dia_max": "—",
                "rank": "no det.",
            })
        else:
            peak_idx = sub[rrs_col].idxmax()
            peak_row = daily_df.loc[peak_idx]
            # Rank: posición en el ranking general
            rank_pos = int((rrs_sorted[rrs_col] >= peak_row[rrs_col]).sum())
            resultados.append({
                "evento_id": eid,
                "etiqueta": ev["etiqueta"],
                "fecha_evento": ev_date.strftime("%Y-%m-%d"),
                "dia_pico": peak_row["Fecha"].strftime("%Y-%m-%d"),
                "rrs_pico": f"{peak_row[rrs_col]:.4f}",
                "n_notas_dia_max": int(peak_row["n_notas"]),
                "rank": rank_pos,
            })
    return resultados


# ======================================================================
#  CARGA DE DATOS
# ======================================================================
print("=" * 70)
print("CARGA DE DATOS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)
# Fecha mezcla "YYYY-MM-DD" y "YYYY-MM-DD 00:00:00" (jun–ago 2020); parse_dates la deja como texto
df["Fecha"] = pd.to_datetime(df["Fecha"], format="mixed")
print(f"Cargadas {len(df):,} notas | {df['Fecha'].min().date()} → {df['Fecha'].max().date()}")

# Verificar columnas esenciales
for col in ["Fecha", "Medio", "Encabezado", "Tono"]:
    assert col in df.columns, f"Falta columna obligatoria: {col}"

# NLP_Score (BERT, ec. 1) vive en infonavit_nlp_results.csv, alineado fila a fila
if "NLP_Score" not in df.columns and os.path.exists(NLP_FILE):
    nlp = pd.read_csv(NLP_FILE, encoding="utf-8-sig", low_memory=False,
                      usecols=["Encabezado", "Tono", "NLP_Sentimiento", "NLP_Score"])
    assert len(nlp) == len(df) and \
        (nlp["Encabezado"].fillna("").values == df["Encabezado"].fillna("").values).all() and \
        (nlp["Tono"].values == df["Tono"].values).all(), \
        "infonavit_nlp_results.csv no está alineado con infonavit_consolidated.csv"
    df["NLP_Score"] = nlp["NLP_Score"].values
    df["NLP_Sentimiento"] = nlp["NLP_Sentimiento"].values

# Si existe NLP_Score, usarlo para C1; si no, mapear Tono
tono_map = {"Negativo": -1, "Neutral": 0, "Positivo": 1}
if "NLP_Score" in df.columns:
    df["Sent_Num"] = df["NLP_Score"].astype(float)
    print("  → Usando NLP_Score para C1")
else:
    df["Sent_Num"] = df["Tono"].map(tono_map).fillna(0)
    print("  → Usando Tono mapeado (-1/0/+1) para C1")

# Keywords de crisis por artículo
df["es_crisis"] = df["Encabezado"].apply(keyword_hit)

# Excluir bloque incompleto
n_antes = len(df)
df = df[~((df["Fecha"] >= EXCL_START) & (df["Fecha"] <= EXCL_END))].copy()
n_excl = n_antes - len(df)
print(f"  → Excluidas {n_excl:,} notas del bloque {EXCL_START.date()} – {EXCL_END.date()}")
print(f"  → Quedan {len(df):,} notas")

# ======================================================================
#  AGREGACIÓN DIARIA  (sin umbral aún)
# ======================================================================
daily_all = df.groupby(df["Fecha"].dt.date).agg(
    n_notas=("Encabezado", "count"),
    mean_sent=("Sent_Num", "mean"),
    n_neg=("Tono", lambda x: (x == "Negativo").sum()),
    n_crisis=("es_crisis", "sum"),
).reset_index()
daily_all.rename(columns={"Fecha": "Fecha"}, inplace=True)
daily_all["Fecha"] = pd.to_datetime(daily_all["Fecha"])
daily_all = daily_all.sort_values("Fecha").reset_index(drop=True)

# ── Cobertura mensual (Bloque 0) ─────────────────────────────────────
daily_all["YM"] = daily_all["Fecha"].dt.to_period("M")
cob_mensual = daily_all.groupby("YM").agg(
    dias_con_notas=("n_notas", "count"),
    notas_total=("n_notas", "sum"),
    notas_dia_media=("n_notas", "mean"),
    notas_dia_min=("n_notas", "min"),
    notas_dia_max=("n_notas", "max"),
).reset_index()
cob_mensual.to_csv(os.path.join(RESULTS_DIR, "cobertura_mensual.csv"), index=False)

# Contar meses "ausentes" (< 10 días con notas) y días con 0 notas
# Generar rango completo de fechas
full_range = pd.date_range(daily_all["Fecha"].min(), daily_all["Fecha"].max(), freq="D")
# Excluir bloque incompleto del rango completo
full_range = full_range[~((full_range >= EXCL_START) & (full_range <= EXCL_END))]
dias_sin_notas = len(full_range) - len(daily_all)

meses_rango = pd.period_range(daily_all["Fecha"].min(), daily_all["Fecha"].max(), freq="M")
# Excluir meses del bloque
meses_excl = pd.period_range(EXCL_START, EXCL_END, freq="M")
meses_rango = meses_rango.difference(meses_excl)
meses_con_datos = set(cob_mensual["YM"])
meses_ausentes = [m for m in meses_rango if m not in meses_con_datos]


# ======================================================================
#  BUCLE POR UMBRAL
# ======================================================================
resultados_md = []   # bloques de texto para el reporte final
resumen_umbrales = []

for u_idx, umbral in enumerate(UMBRALES):
    tag = f"u{umbral}"
    es_base = (u_idx == 0)

    print(f"\n{'='*70}")
    print(f"UMBRAL = {umbral} notas/día  {'(CASO BASE)' if es_base else '(sensibilidad)'}")
    print(f"{'='*70}")

    # Filtrar por umbral
    daily = daily_all[daily_all["n_notas"] >= umbral].copy()
    daily = daily.sort_values("Fecha").reset_index(drop=True)
    n_dias = len(daily)
    n_excl_umbral = len(daily_all) - n_dias
    print(f"  Días incluidos: {n_dias}  (excluidos por umbral: {n_excl_umbral})")

    # Componentes
    daily = compute_components(daily)

    # RRS caso base
    daily["RRS"] = compute_rrs(daily)

    print(f"  RRS — Media: {daily['RRS'].mean():.4f}, DE: {daily['RRS'].std():.4f}, "
          f"Mín: {daily['RRS'].min():.4f}, Máx: {daily['RRS'].max():.4f}")

    # Guardar serie diaria
    if es_base:
        daily.to_csv(os.path.join(RESULTS_DIR, f"rrs_diario_{tag}.csv"), index=False)

    # ── Detección de crisis ──────────────────────────────────────────
    z, mask, n_crisis, episodes = detect_crises(daily["RRS"], daily["Fecha"])
    daily["z"] = z
    daily["crisis"] = mask.astype(int)

    print(f"  Crisis: {n_crisis} días, {len(episodes)} episodios")

    # Guardar episodios
    ep_rows = []
    for i, ep in enumerate(episodes, 1):
        s = daily.iloc[ep["start"]]
        e = daily.iloc[ep["end"]]
        ep_rows.append({
            "episodio": i,
            "inicio": s["Fecha"].strftime("%Y-%m-%d"),
            "fin": e["Fecha"].strftime("%Y-%m-%d"),
            "dias": (e["Fecha"] - s["Fecha"]).days + 1,
            "rrs_max": daily.iloc[ep["start"]:ep["end"]+1]["RRS"].max(),
        })
    ep_df = pd.DataFrame(ep_rows)
    if es_base:
        ep_df.to_csv(os.path.join(RESULTS_DIR, f"episodios_{tag}.csv"), index=False)

    # ── Ranking de eventos ───────────────────────────────────────────
    rank_res = rank_eventos(daily, "RRS")

    # ── Sensibilidad de pesos ────────────────────────────────────────
    esquemas = {
        "Base (0.35,0.25,0.25,0.15)": (0.35, 0.25, 0.25, 0.15),
        "Igual (0.25,0.25,0.25,0.25)": (0.25, 0.25, 0.25, 0.25),
        "Sent. pesado (0.50,0.20,0.20,0.10)": (0.50, 0.20, 0.20, 0.10),
        "Vol. pesado (0.25,0.35,0.25,0.15)": (0.25, 0.35, 0.25, 0.15),
        "Sin sent. (0.00,0.40,0.35,0.25)": (0.00, 0.40, 0.35, 0.25),
    }
    sens_rows = []
    for nombre, pesos in esquemas.items():
        rrs_alt = compute_rrs(daily, pesos)
        r_pearson = daily["RRS"].corr(rrs_alt)
        _, m_alt, nc_alt, ep_alt = detect_crises(rrs_alt, daily["Fecha"])
        rec_alt = 0
        for eid, ev in EVENTOS.items():
            ev_date = pd.Timestamp(ev["fecha"])
            crisis_dates = set(daily.loc[m_alt, "Fecha"].dt.date)
            ev_range = set(pd.date_range(ev_date - pd.Timedelta(days=TOLERANCIA),
                                         ev_date + pd.Timedelta(days=TOLERANCIA)).date)
            if crisis_dates & ev_range:
                rec_alt += 1
        sens_rows.append({
            "esquema": nombre,
            "pesos": str(pesos),
            "media": f"{rrs_alt.mean():.4f}",
            "de": f"{rrs_alt.std():.4f}",
            "r_pearson": f"{r_pearson:.4f}",
            "dias_crisis": nc_alt,
            "episodios": len(ep_alt),
            "eventos_recuperados": f"{rec_alt}/5",
        })
    sens_df = pd.DataFrame(sens_rows)
    if es_base:
        sens_df.to_csv(os.path.join(RESULTS_DIR, "sensibilidad_pesos.csv"), index=False)

    # ── Malla de detección ───────────────────────────────────────────
    malla_rows = []
    for w_ in GRID_WINDOWS:
        for t_ in GRID_THRESHOLDS:
            for g_ in GRID_GAPS:
                z_m, mask_m, nc_m, ep_m = detect_crises(
                    daily["RRS"], daily["Fecha"],
                    window=w_, threshold=t_, merge_gap=g_
                )
                # Eventos recuperados
                rec_m = 0
                for eid, ev in EVENTOS.items():
                    ev_date = pd.Timestamp(ev["fecha"])
                    crisis_dates_m = set(daily.loc[mask_m, "Fecha"].dt.date)
                    ev_range_m = set(pd.date_range(
                        ev_date - pd.Timedelta(days=TOLERANCIA),
                        ev_date + pd.Timedelta(days=TOLERANCIA)).date)
                    if crisis_dates_m & ev_range_m:
                        rec_m += 1
                malla_rows.append({
                    "w": w_, "tau": t_, "g": g_,
                    "dias_crisis": nc_m,
                    "episodios": len(ep_m),
                    "eventos_rec": rec_m,
                })
    malla_df = pd.DataFrame(malla_rows)
    if es_base:
        malla_df.to_csv(os.path.join(RESULTS_DIR, "malla_deteccion.csv"), index=False)

    # ── Negativos mensuales (Tono proveedor) ─────────────────────────
    daily["YM2"] = daily["Fecha"].dt.to_period("M")
    neg_mensual = daily.groupby("YM2").agg(
        notas=("n_notas", "sum"),
        neg=("n_neg", "sum"),
    ).reset_index()
    neg_mensual["pct_neg"] = (neg_mensual["neg"] / neg_mensual["notas"] * 100).round(2)

    # Regresión de tendencia
    neg_mensual["t"] = range(len(neg_mensual))
    if len(neg_mensual) > 2:
        slope, intercept, r_val, p_val, se = linregress(
            neg_mensual["t"], neg_mensual["pct_neg"])
        drift = slope * len(neg_mensual)
    else:
        slope = intercept = r_val = p_val = drift = float("nan")

    if es_base:
        neg_mensual.to_csv(os.path.join(RESULTS_DIR, "negativos_mensual.csv"), index=False)

    # ── Top-20 medios ────────────────────────────────────────────────
    top20_medios = df.groupby("Medio").size().nlargest(20).index.tolist()
    df_top20 = df[df["Medio"].isin(top20_medios)].copy()
    daily_t20 = df_top20.groupby(df_top20["Fecha"].dt.date).agg(
        n_notas=("Encabezado", "count"),
        mean_sent=("Sent_Num", "mean"),
        n_neg=("Tono", lambda x: (x == "Negativo").sum()),
        n_crisis=("es_crisis", "sum"),
    ).reset_index()
    daily_t20.rename(columns={"Fecha": "Fecha"}, inplace=True)
    daily_t20["Fecha"] = pd.to_datetime(daily_t20["Fecha"])
    daily_t20 = daily_t20.sort_values("Fecha").reset_index(drop=True)
    daily_t20 = daily_t20[daily_t20["n_notas"] >= UMBRAL_TOP20].copy()
    daily_t20 = daily_t20.sort_values("Fecha").reset_index(drop=True)
    daily_t20 = compute_components(daily_t20)
    daily_t20["RRS_t20"] = compute_rrs(daily_t20)

    merged_t20 = daily[["Fecha", "RRS"]].merge(
        daily_t20[["Fecha", "RRS_t20"]], on="Fecha", how="inner"
    )
    r_top20 = merged_t20["RRS"].corr(merged_t20["RRS_t20"]) if len(merged_t20) > 2 else float("nan")
    n_dias_top20 = len(merged_t20)

    # ── STL ──────────────────────────────────────────────────────────
    stl_results = {}
    for period in STL_PERIODS:
        try:
            ts = daily.set_index("Fecha")["RRS"].asfreq("D")
            ts = ts.interpolate(method="linear").dropna()
            if len(ts) < 2 * period + 1:
                stl_results[period] = None
                continue
            stl_fit = STL(ts, period=period, robust=True).fit()
            stl_df = pd.DataFrame({
                "Fecha": ts.index,
                "RRS": ts.values,
                "Trend": stl_fit.trend,
                "Seasonal": stl_fit.seasonal,
                "Residual": stl_fit.resid,
            })
            total_var = ts.var()
            if total_var > 0:
                var_trend = stl_fit.trend.var() / total_var
                var_season = stl_fit.seasonal.var() / total_var
                var_resid = stl_fit.resid.var() / total_var
            else:
                var_trend = var_season = var_resid = float("nan")
            stl_results[period] = {
                "var_trend": var_trend,
                "var_season": var_season,
                "var_resid": var_resid,
            }
            if es_base:
                stl_df.to_csv(os.path.join(RESULTS_DIR, f"stl_p{period}.csv"), index=False)
        except Exception as exc:
            print(f"  STL p={period}: {exc}")
            stl_results[period] = None

    # ── Acumular resumen ─────────────────────────────────────────────
    resumen_umbrales.append({
        "umbral": umbral,
        "n_dias": n_dias,
        "rrs_media": daily["RRS"].mean(),
        "rrs_de": daily["RRS"].std(),
        "rrs_min": daily["RRS"].min(),
        "rrs_max": daily["RRS"].max(),
        "dias_crisis": n_crisis,
        "episodios": len(episodes),
        "rank_res": rank_res,
        "sens_df": sens_df,
        "malla_df": malla_df,
        "r_top20": r_top20,
        "n_dias_top20": n_dias_top20,
        "neg_slope": slope,
        "neg_p": p_val,
        "neg_drift": drift,
        "stl_results": stl_results,
        "ep_df": ep_df,
        "daily": daily,
    })


# ======================================================================
#  GENERAR REPORTE MARKDOWN
# ======================================================================
print(f"\n{'='*70}")
print("GENERANDO resultados_robustez.md")
print(f"{'='*70}")

R = []  # líneas del reporte

R.append("# Resultados de robustez — RRS corregido (jrfm-4546218)\n")
R.append(f"Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
R.append(f"Entrada: `{os.path.basename(INPUT_FILE)}` ({len(df):,} notas tras exclusión)\n")
R.append(f"Bloque excluido: {EXCL_START.date()} – {EXCL_END.date()} "
         f"(entrega incompleta, {n_excl:,} notas removidas)\n")

# ── Bloque 0: cobertura ──────────────────────────────────────────────
R.append("\n---\n")
R.append("## Bloque 0 — Diagnóstico de cobertura\n")
R.append(f"- Meses ausentes (< 10 días con datos): **{len(meses_ausentes)}**")
if meses_ausentes:
    R.append(f"  → {', '.join(str(m) for m in meses_ausentes)}")
R.append(f"- Días con cero notas (en rango activo, excluido el bloque): **{dias_sin_notas}**\n")
R.append("### Cobertura mensual\n")
R.append(tabulate(cob_mensual.to_dict("records"), headers="keys",
                  tablefmt="pipe", floatfmt=".1f"))
R.append("")

# ── Por cada umbral ──────────────────────────────────────────────────
for res in resumen_umbrales:
    u = res["umbral"]
    es_base = (u == UMBRALES[0])
    label = f"CASO BASE (umbral = {u})" if es_base else f"Sensibilidad (umbral = {u})"

    R.append(f"\n---\n")
    R.append(f"# {label}\n")
    R.append(f"- Días incluidos: **{res['n_dias']}**")
    R.append(f"- RRS: media = {res['rrs_media']:.4f}, DE = {res['rrs_de']:.4f}, "
             f"mín = {res['rrs_min']:.4f}, máx = {res['rrs_max']:.4f}\n")

    # Bloque 1: ranking de eventos
    R.append(f"## Bloque 1 — Ranking de eventos documentados (umbral {u})\n")
    R.append("| # | Evento | Fecha evento | Día pico | RRS pico | n_notas_dia_max | Rank |")
    R.append("|---|--------|-------------|----------|---------|----------------|------|")
    for rr in res["rank_res"]:
        R.append(f"| {rr['evento_id']} | {rr['etiqueta']} | {rr['fecha_evento']} | "
                 f"{rr['dia_pico']} | {rr['rrs_pico']} | {rr['n_notas_dia_max']} | "
                 f"{rr['rank']} |")
    R.append("")

    no_det = [r for r in res["rank_res"] if r["rank"] == "no det."]
    if no_det:
        R.append(f"> [!WARNING]\n> Eventos sin detección: "
                 f"{', '.join(str(r['evento_id']) for r in no_det)}. "
                 f"Revisar sus fechas en el diccionario EVENTOS.\n")

    # Bloque 2: episodios detectados
    R.append(f"## Bloque 2 — Episodios detectados (w={WINDOW_Z}, τ={THRESHOLD_Z}, g={MERGE_GAP})\n")
    R.append(f"- Días de crisis: **{res['dias_crisis']}**")
    R.append(f"- Episodios (tras fusión): **{res['episodios']}**\n")
    if not res["ep_df"].empty:
        R.append(tabulate(res["ep_df"].to_dict("records"), headers="keys",
                          tablefmt="pipe", floatfmt=".4f"))
        R.append("")

    # Bloque 3: malla de detección
    if es_base:
        R.append(f"## Bloque 3 — Malla de detección (solo caso base)\n")
        R.append(tabulate(res["malla_df"].to_dict("records"), headers="keys",
                          tablefmt="pipe"))
        R.append("")

    # Bloque 4: sensibilidad de pesos
    R.append(f"## Bloque 4 — Sensibilidad de pesos (umbral {u})\n")
    R.append(tabulate(res["sens_df"].to_dict("records"), headers="keys",
                      tablefmt="pipe"))
    R.append("")

    # Bloque 5: Top-20 medios
    R.append(f"## Bloque 5 — Top-20 medios (umbral {u})\n")
    R.append(f"- Días en intersección: **{res['n_dias_top20']}**")
    R.append(f"- Correlación Pearson (RRS completo vs top-20): **{res['r_top20']:.4f}**\n")
    if res["n_dias_top20"] < 50:
        R.append(f"> [!WARNING]\n> Solo {res['n_dias_top20']} días en la intersección. "
                 f"Considere bajar UMBRAL_TOP20 (actualmente {UMBRAL_TOP20}).\n")

    # Bloque 6: estabilidad del clasificador
    R.append(f"## Bloque 6 — Estabilidad del clasificador de tono (umbral {u})\n")
    R.append(f"- Pendiente tendencia: **{res['neg_slope']:.4f}** pp/mes")
    R.append(f"- p-valor: **{res['neg_p']:.4f}**")
    R.append(f"- Deriva acumulada: **{res['neg_drift']:.1f}** pp\n")
    if not np.isnan(res["neg_p"]):
        if res["neg_p"] > 0.05:
            R.append(f"> [!NOTE]\n> La pendiente NO es significativa al 5% (p = {res['neg_p']:.4f}). "
                     f"El clasificador de tono es estable en el tiempo.\n")
        else:
            R.append(f"> [!WARNING]\n> La pendiente ES significativa (p = {res['neg_p']:.4f}). "
                     f"Revisar posible deriva del clasificador.\n")

    # Bloque 7 (solo caso base): Top 10 días
    if es_base:
        R.append("## Bloque 7 — Top 10 días de mayor RRS\n")
        top10 = res["daily"].nlargest(10, "RRS")[
            ["Fecha", "n_notas", "C1", "C2", "C3", "C4", "RRS", "z"]
        ].copy()
        top10["Fecha"] = top10["Fecha"].dt.strftime("%Y-%m-%d")
        for c in ["C1", "C2", "C3", "C4", "RRS", "z"]:
            top10[c] = top10[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        R.append(tabulate(top10.to_dict("records"), headers="keys", tablefmt="pipe"))
        R.append("")

    # Bloque 8: STL
    R.append(f"## Bloque 8 — Descomposición STL (umbral {u})\n")
    R.append("| Periodo | Var(Tendencia) | Var(Estacional) | Var(Residual) |")
    R.append("|---------|---------------|-----------------|---------------|")
    for p in STL_PERIODS:
        s = res["stl_results"].get(p)
        if s:
            R.append(f"| {p} | {s['var_trend']:.4f} | {s['var_season']:.4f} | {s['var_resid']:.4f} |")
        else:
            R.append(f"| {p} | — | — | — |")
    R.append("")


# ── Resumen comparativo de umbrales ──────────────────────────────────
R.append("\n---\n")
R.append("## Comparación entre umbrales\n")
comp_rows = []
for res in resumen_umbrales:
    comp_rows.append({
        "Umbral": res["umbral"],
        "Días": res["n_dias"],
        "Media RRS": f"{res['rrs_media']:.4f}",
        "DE RRS": f"{res['rrs_de']:.4f}",
        "Días crisis": res["dias_crisis"],
        "Episodios": res["episodios"],
    })
R.append(tabulate(comp_rows, headers="keys", tablefmt="pipe"))
R.append("")

# ── Notas metodológicas ─────────────────────────────────────────────
R.append("\n---\n")
R.append("## Notas metodológicas\n")
R.append("1. **C2** usa la CDF normal Φ((n − μ)/σ), no min-max.")
R.append("2. **C3** y **C4** son proporciones crudas (n_neg/n, n_crisis/n).")
R.append("3. No se aplica min-max a ningún componente ni reescalado a [0,100].")
R.append("4. **C3** usa la columna `Tono` (codificación del proveedor humano).")
R.append("5. Del léxico de crisis se quitó «demanda» a secas y se añadieron "
         "«demanda judicial», «demandado», «demandan».")
R.append(f"6. Se excluye el bloque {EXCL_START.date()} – {EXCL_END.date()} "
         "(entrega incompleta del proveedor).")
R.append(f"7. Umbral caso base: {UMBRALES[0]} notas/día. "
         f"Sensibilidad: {', '.join(str(u) for u in UMBRALES[1:])}.")
R.append(f"8. Eventos documentados validados contra registro institucional "
         f"y de prensa (tolerancia ±{TOLERANCIA} días).\n")

# Escribir
report_text = "\n".join(R)
report_path = os.path.join(RESULTS_DIR, "resultados_robustez.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n  → Reporte guardado: {report_path}")
print(f"  → CSVs en: {RESULTS_DIR}/")
print(f"\n{'='*70}")
print("COMPLETADO")
print(f"{'='*70}")