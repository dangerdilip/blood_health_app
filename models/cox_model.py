import logging
import pandas as pd
from lifelines import CoxPHFitter
import joblib
import os

logger = logging.getLogger(__name__)

MODEL_FILE = os.path.join(os.path.dirname(__file__), "cox_model.pkl")

logger.info("cox_model module loaded")

def train_and_save_model(data_csv_path):
    logger.info("train_and_save_model() called")
    logger.info("CSV path received: %s", data_csv_path)

    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"CSV not found at: {data_csv_path}")

    df = pd.read_csv(data_csv_path)
    logger.info("CSV loaded, shape: %s", df.shape)

    if "duration" not in df.columns or "event" not in df.columns:
        raise ValueError("CSV must contain 'duration' and 'event' columns")

    # Convert duration to numeric days
    df["duration"] = pd.to_timedelta(df["duration"]).dt.total_seconds() / (24 * 3600)
    df = df[df["duration"] > 0]

    logger.info("Training Cox model...")

    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    joblib.dump(cph, MODEL_FILE)

    logger.info("Cox model trained and saved at: %s", MODEL_FILE)
