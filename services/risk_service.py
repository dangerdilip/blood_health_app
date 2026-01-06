import pandas as pd
import logging
from features.feature_builder import build_features
from models.interface import predict_risk

logger = logging.getLogger(__name__)

def calculate_risk(cbc_records: list, alerts: list | None = None) -> dict:
    """
    Core clinical risk evaluation logic.
    - alerts: extreme-value alerts injected from API layer
    """

    alerts = alerts or []

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

    has_extreme_alerts = len(alerts) > 0

    # ---------- BLOOD STATUS (FIXED & MEDICALLY CORRECT) ----------
    if has_extreme_alerts:
        blood_status = "Critical abnormalities detected"
    elif flags:
        blood_status = "Abnormal but stable"
    else:
        blood_status = "Clinically stable"

    # ---------- SINGLE CBC (NO ML) ----------
    if features is None:
        recommendation = (
            "Extreme or clinically dangerous values detected. "
            "Please consult a doctor immediately."
            if has_extreme_alerts
            else "Periodic monitoring advised"
        )

        return {
            "blood_status": blood_status,
            "future_risk": "Insufficient data for trend-based prediction",
            "recommendation": recommendation,
            "flags": flags
        }

    # ---------- ML MODEL (SAFE) ----------
    try:
        risk_score = float(predict_risk(features))
    except Exception as e:
        logger.error(f"Cox model inference failed: {e}")

        recommendation = (
            "Extreme or clinically dangerous values detected. "
            "Please consult a doctor immediately."
            if has_extreme_alerts
            else "Periodic monitoring advised"
        )

        return {
            "blood_status": blood_status,
            "future_risk": "Risk model unavailable; rule-based assessment applied",
            "recommendation": recommendation,
            "flags": flags + ["ML risk model unavailable"]
        }

    # ---------- RISK INTERPRETATION ----------
    if risk_score < 0.15:
        future_risk = "Low probability of further deterioration in the next 30–60 days"
    elif risk_score < 0.30:
        future_risk = "Moderate probability of deterioration in the next 30–60 days"
    else:
        future_risk = "High probability of deterioration in the next 30–60 days"

    # ---------- RECOMMENDATION ----------
    if has_extreme_alerts:
        recommendation = (
            "Extreme or clinically dangerous values detected. "
            "Please consult a doctor immediately, even if symptoms are mild."
        )
    elif risk_score >= 0.3:
        recommendation = "Immediate clinical follow-up advised"
    else:
        recommendation = "Periodic monitoring advised"

    return {
        "blood_status": blood_status,
        "future_risk": future_risk,
        "recommendation": recommendation,
        "risk_score": round(risk_score, 2),
        "flags": flags
    }
