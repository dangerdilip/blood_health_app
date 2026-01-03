# train_model.py
import pandas as pd
from lifelines import CoxPHFitter
import pickle
import os

def train_and_save_model(data_csv_path="data/cbc_ubhnss_for_cox.csv",
                         save_path="models/cox_model.pkl"):

    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"CSV not found at: {data_csv_path}")

    print("Loading CSV:", data_csv_path)
    df = pd.read_csv(data_csv_path)
    print("CSV loaded, shape:", df.shape)

    # Ensure required columns exist
    if "duration" not in df.columns or "event" not in df.columns:
        raise ValueError("CSV must contain 'duration' and 'event' columns")

    # Drop identifier / non-predictive columns
    drop_cols = ["patient_id", "ward_id", "SampleNum"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop highly correlated / redundant columns (z-scores or deltas)
    redundant_cols = [c for c in df.columns if "_z" in c or "_delta" in c]
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = df[numeric_cols]

    # Ensure duration and event are included
    if "duration" not in df.columns or "event" not in df.columns:
        raise ValueError("After cleaning, 'duration' or 'event' column is missing")

    # Drop NaNs
    df = df.dropna()
    print("Final shape after cleaning:", df.shape)
    print("Columns used:", df.columns.tolist())

    # Correct types
    df["duration"] = df["duration"].astype(float)
    df["event"] = df["event"].astype(int)

    # Train Cox model
    print("Training Cox model...")
    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")
    print(cph.summary)
    print("Concordance index:", cph.concordance_index_)

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(cph, f)
    print("Model saved at:", save_path)


if __name__ == "__main__":
    train_and_save_model()
