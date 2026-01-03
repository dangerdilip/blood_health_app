import pandas as pd

# Load final CSV
df = pd.read_csv("data/cbc_ubhnss_final.csv")

# Ensure timestamp is datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Compute duration in days from first test per patient
df["duration"] = (df["timestamp"] - df.groupby("patient_id")["timestamp"].transform("first")).dt.total_seconds() / (24*3600)

# Create event column (1 if UBHS < 30, else 0)
df["event"] = (df["UBHS"] < 30).astype(int)

# Take last record per patient for survival analysis
survival_df = df.groupby("patient_id").tail(1)

# Save CSV for Cox model
survival_df.to_csv("data/cbc_ubhnss_for_cox.csv", index=False)

print("CSV created successfully:", survival_df.shape)