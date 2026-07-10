from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from services.risk_service import calculate_risk

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="Blood Health Risk API",
    version="1.2.1"
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://blood-health-frontend.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- EXTREME CLINICAL LIMITS ----------------
EXTREME_LIMITS = {
    "hemoglobin": (5.0, 20.0),    # g/dL - below 5 = life threatening anemia, above 20 = polycythemia
    "wbc": (1.0, 50.0),           # ×10³/µL - below 1 = severe neutropenia, above 50 = leukemic range
    "platelets": (20.0, 1000.0),  # ×10³/µL - below 20 = spontaneous bleeding, above 1000 = thrombocytosis
    "rbc": (1.5, 8.0),            # million/µL
    "mcv": (50.0, 130.0),         # fL
    "mch": (15.0, 45.0),          # pg
    "mchc": (25.0, 40.0),         # g/dL
}

# ---------------- NORMAL REFERENCE RANGES ----------------
NORMAL_RANGES = {
    "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL"},
    "wbc": {"min": 4.5, "max": 11.0, "unit": "×10³/µL"},
    "platelets": {"min": 150.0, "max": 400.0, "unit": "×10³/µL"},
    "rbc": {"min": 4.0, "max": 6.0, "unit": "million/µL"},
    "mcv": {"min": 80.0, "max": 100.0, "unit": "fL"},
    "mch": {"min": 27.0, "max": 33.0, "unit": "pg"},
    "mchc": {"min": 32.0, "max": 36.0, "unit": "g/dL"},
}

# ---------------- SCHEMAS ----------------
class CBCRecord(BaseModel):
    date: str
    hemoglobin: float
    wbc: float
    platelets: float
    rbc: float
    mcv: float
    mch: float
    mchc: float


class PatientCBCRequest(BaseModel):
    patient_id: str
    records: list[CBCRecord]


# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"status": "ok", "service": "blood-health-backend"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- PREDICTION ----------------
@app.post("/risk/predict")
def predict_risk(payload: PatientCBCRequest):
    logger.info(
        f"Risk prediction request | patient_id={payload.patient_id} "
        f"| records={len(payload.records)}"
    )

    # Validate: reject records with zero or missing values
    for idx, record in enumerate(payload.records):
        data = record.model_dump()
        zero_fields = [f for f in ["hemoglobin", "wbc", "platelets", "rbc", "mcv", "mch", "mchc"] if data.get(f, 0) <= 0]
        if zero_fields:
            return {
                "patient_id": payload.patient_id,
                "error": True,
                "blood_status": "Invalid data",
                "future_risk": "Cannot assess — incomplete data provided",
                "recommendation": f"Record {idx+1} has invalid or missing values for: {', '.join(f.upper() for f in zero_fields)}. Please enter valid CBC values.",
                "alerts": [f"Record {idx+1}: {f.upper()} must be greater than zero" for f in zero_fields],
                "normal_ranges": NORMAL_RANGES,
                "flags": []
            }

    alerts = []

    # Generate alerts for extreme clinical values
    for idx, record in enumerate(payload.records):
        data = record.model_dump()

        for field, value in data.items():
            if field == "date":
                continue

            if value < 0:
                alerts.append(
                    f"Record {idx+1}: {field.upper()} = {value} "
                    f"is invalid (negative values are not allowed)."
                )
                continue

            if field in EXTREME_LIMITS:
                low, high = EXTREME_LIMITS[field]
                if value < low or value > high:
                    alerts.append(
                        f"Record {idx+1}: {field.upper()} = {value} "
                        f"is in an extreme clinical range. "
                        f"Please consult a doctor at the earliest."
                    )

    # Pass alerts into risk engine
    result = calculate_risk(
        payload.records,
        alerts=alerts
    )

    return {
        "patient_id": payload.patient_id,
        **result,
        "alerts": alerts,
        "normal_ranges": NORMAL_RANGES,
        "records_submitted": len(payload.records),
        "min_records_for_trend": 2
    }
