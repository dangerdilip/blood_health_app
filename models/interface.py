import os
import pickle
import logging
import pandas as pd
import numpy as np

# ---------------- LOGGING ----------------
logger = logging.getLogger(__name__)

# ---------------- GLOBAL MODEL CACHE ----------------
_COX_MODEL = None

# ---------------- MODEL LOADER ----------------
def _load_cox_model():
    """
    Lazy-load Cox model.
    Ensures:
    - Model loads once per worker
    - No import-time side effects
    - Safe for Gunicorn & Docker
    """
    global _COX_MODEL

    if _COX_MODEL is not None:
        return

    model_path = os.path.join(
        os.path.dirname(__file__),
        "cox_model.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cox model file not found: {model_path}")

    try:
        with open(model_path, "rb") as f:
            _COX_MODEL = pickle.load(f)
        logger.info("✅ Cox model loaded successfully")
         # ✅ NEW LOG LINE
    logger.info(f"Cox model type: {type(_COX_MODEL)}")
    
    except Exception as e:
        logger.exception("❌ Failed to load Cox model")
        raise RuntimeError("Cox model load failed") from e


# ---------------- RISK PREDICTION ----------------
def predict_risk(features_df: pd.DataFrame) -> float:
    """
    Predict risk score using Cox Proportional Hazards model.

    Uses survival function math:
        risk = 1 - survival probability (≈ 90 days)

    Returns:
        float in [0, 1]
    """

    # ---------- SAFETY ----------
    if features_df is None or features_df.empty:
        logger.warning("Empty features received — returning 0 risk")
        return 0.0

    _load_cox_model()

    # Replace infinities and NaNs defensively
    X = features_df.replace([np.inf, -np.inf], np.nan)

    if X.isnull().any().any():
        logger.warning("NaN detected in features — returning 0 risk")
        return 0.0

    try:
        # Predict survival function
        surv_df = _COX_MODEL.predict_survival_function(X)

        # Handle DataFrame output
        if isinstance(surv_df, pd.DataFrame):
            surv_series = surv_df.iloc[:, 0]
        else:
            surv_series = surv_df

        # Survival at ~90 days (fallback to last available)
        if 90 in surv_series.index:
            survival_prob = float(surv_series.loc[90])
        else:
            survival_prob = float(surv_series.iloc[-1])

        # Risk = 1 - survival
        risk_score = 1.0 - survival_prob

        # Clamp strictly
        risk_score = max(0.0, min(float(risk_score), 1.0))

        return risk_score

    except Exception as e:
        logger.exception("❌ Cox inference failed — returning safe fallback")
        return 0.0
