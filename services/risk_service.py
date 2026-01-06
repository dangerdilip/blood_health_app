import pandas as pd
import logging
from features.feature_builder import build_features
from models.interface import predict_risk

logger = logging.getLogger(__name__)

def calculate_risk(cbc_records: list) -> dict:
    # ---------- INPUT TO DATAFRAME ----------
    df = pd.DataFrame([r.model_dump() for r in cbc_records])

    features, stats = build_features(df)

    flags = []

    # ---------- RULE ENGINE ----------
    if stats["UBHS"] < 40:
        flags.append("Low blood stability index (UBHS)")

    if stats["NSP"] > 0.7:
        flags.append("Physiological stress pattern detected (NSP)")

    if stats["abnormal_z_count"] >= 2:
        flags.append("Multiple abnormal CBC markers detected")

    # ---------- BLOOD STATUS (FIXED) ----------
    blood_status = (
        "Abnormal but stable" if flags else "Clinically stable"
    )

    # ---------- SINGLE CBC (NO ML) ----------
    if features is None:
        return {
            "blood_status": blood_status,
            "future_risk": "Insufficient data for trend-based prediction",
            "recommendation": "Periodic monitoring advised",
            "flags": flags
        }

    # ---------- ML MODEL (SAFE) ----------
    try:
        risk_score = float(predict_risk(features))
    except Exception as e:
        logger.error(f"Cox model inference failed: {e}")
        return {
            "blood_status": blood_status,
            "future_risk": "Risk model unavailable; rule-based assessment applied",
            "recommendation": "Periodic monitoring advised",
            "flags": flags + ["ML risk model unavailable"]
        }

        # ---------- RISK INTERPRETATION ----------
    if risk_score < 0.15:
        future_risk = "Low probability of further deterioration in the next 30–60 days"
    elif risk_score < 0.30:
        future_risk = "Moderate probability of deterioration in the next 30–60 days"
    else:
        future_risk = "High probability of deterioration in the next 30–60 days"

    recommendation = (
        "Immediate clinical follow-up advised"
        if risk_score >= 0.3
        else "Periodic monitoring advised"
    )

    # ---------- SAFE OVERRIDE (DOES NOT REMOVE LOGIC) ----------
    if flags:
        blood_status = "Critical abnormalities detected"
        recommendation = (
            "Extreme or clinically dangerous values detected. "
            "Please consult a doctor immediately, even if symptoms are mild."
        )

    return {
        "blood_status": blood_status,
        "future_risk": future_risk,
        "recommendation": recommendation,
        "risk_score": round(risk_score, 2),
        "flags": flags
    }
