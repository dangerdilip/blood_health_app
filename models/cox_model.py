import pandas as pd
from lifelines import CoxPHFitter
import joblib
import os

print("cox_model.py loaded")  # <-- THIS MUST PRINT

MODEL_FILE = os.path.join(os.path.dirname(__file__), "cox_model.pkl")

def train_and_save_model(data_csv_path):
    print("train_and_save_model() called")
    print("CSV path received:", data_csv_path)

    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"CSV not found at: {data_csv_path}")

    df = pd.read_csv(data_csv_path)
    print("CSV loaded, shape:", df.shape)

    if "duration" not in df.columns or "event" not in df.columns:
        raise ValueError("CSV must contain 'duration' and 'event' columns")

    # Convert duration to numeric days
    df["duration"] = pd.to_timedelta(df["duration"]).dt.total_seconds() / (24 * 3600)
    df = df[df["duration"] > 0]

    print("Training Cox model...")

    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    joblib.dump(cph, MODEL_FILE)

    print("Cox model trained and saved at:", MODEL_FILE)
