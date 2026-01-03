import pandas as pd
import numpy as np

BASIC_COLS = [
    "hemoglobin", "wbc", "platelets",
    "rbc", "mcv", "mch", "mchc"
]

def build_features(df: pd.DataFrame):
    stats = {}

    df = df.copy()

    # ---------- ENSURE NUMERIC ----------
    for col in BASIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------- Z-SCORES ----------
    abnormal_z = 0
    z_scores = {}

    for col in BASIC_COLS:
        if col in df and df[col].std(ddof=0) > 0:
            z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
            z_scores[col] = float(abs(z.iloc[-1]))
            if abs(z.iloc[-1]) > 2:
                abnormal_z += 1
        else:
            z_scores[col] = 0.0

    stats["abnormal_z_count"] = abnormal_z
    stats["z_scores"] = z_scores

    # ---------- UBHS ----------
    variability = df[BASIC_COLS].pct_change().abs().mean(axis=1).fillna(0)
    ubhs = max(0, 100 - variability.iloc[-1] * 100)
    stats["UBHS"] = round(float(ubhs), 2)

    # ---------- NSP ----------
    nsp = min(1.0, variability.iloc[-1] * 3)
    stats["NSP"] = round(float(nsp), 2)

    # ---------- REQUIRE ≥2 CBC ----------
    if len(df) < 2:
        return None, stats

    # ---------- MODEL FEATURES ----------
    df_model = pd.DataFrame()
    df_model["HB"] = df["hemoglobin"]
    df_model["PLT"] = df["platelets"]
    df_model["MCV"] = df["mcv"]
    df_model["MCH"] = df["mch"]
    df_model["MCHC"] = df["mchc"]

    # 🚨 SAFETY FIXES
    df_model["LEUKO"] = df["wbc"].replace(0, np.nan)
    df_model["ERY"] = df["rbc"]

    df_model["RDW"] = 13.0

    df_model["hb_mcv_ratio"] = df_model["HB"] / df_model["MCV"]
    df_model["rdw_mcv_ratio"] = df_model["RDW"] / df_model["MCV"]

    # 🚨 SAFE DIVISION
    df_model["plt_leuko_ratio"] = (
        df_model["PLT"] / df_model["LEUKO"]
    ).replace([np.inf, -np.inf], np.nan)

    dates = pd.to_datetime(df["date"], errors="coerce")
    df_model["cbc_variability"] = variability
    df_model["days_since_last_test"] = (
        dates - dates.shift(1)
    ).dt.days.fillna(0)

    df_model["UBHS"] = stats["UBHS"]
    df_model["NSP"] = stats["NSP"]
    df_model["HK"] = 0

    # Final safety
    df_model = df_model.dropna()

    if df_model.empty:
        return None, stats

    return df_model.tail(1), stats
