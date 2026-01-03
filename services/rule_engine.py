import pandas as pd
import numpy as np

CBC_MARKERS = ["HB", "PLT", "MCV", "MCH", "MCHC", "LEUKO", "ERY"]

def compute_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Z-scores for CBC markers across records.
    """
    z_df = pd.DataFrame()
    for col in CBC_MARKERS:
        if col in df:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0 or pd.isna(std):
                z_df[col] = 0
            else:
                z_df[col] = (df[col] - mean) / std
    return z_df


def evaluate_blood_status(df_model: pd.DataFrame) -> dict:
    """
    Lightweight rule-based blood status evaluation.
    """
    status_flags = []

    # Rule 1: UBHS
    if "UBHS" in df_model and df_model["UBHS"].iloc[-1] < 40:
        status_flags.append("UBHS below normal range")

    # Rule 2: NSP
    if "NSP" in df_model and df_model["NSP"].iloc[-1] > 0.7:
        status_flags.append("Physiological stress indicator elevated")

    # Rule 3: Z-score abnormality
    z_scores = compute_z_scores(df_model)
    abnormal_markers = (z_scores.abs() > 2).sum(axis=1).iloc[-1]

    if abnormal_markers >= 2:
        status_flags.append("Multiple CBC parameters significantly abnormal")

    # Final blood status
    if len(status_flags) == 0:
        blood_status = "Normal"
    elif abnormal_markers >= 2:
        blood_status = "Abnormal"
    else:
        blood_status = "Borderline"

    return {
        "blood_status": blood_status,
        "flags": status_flags
    }
