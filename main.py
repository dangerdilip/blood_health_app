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
    version="1.2.0"
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- EXTREME CLINICAL LIMITS ----------------
EXTREME_LIMITS = {
    "rbc": (2.0, 8.0),               # million/µL
    "hematocrit": (15.0, 65.0),      # %
    "wbc": (1.0, 100.0),             # x10^3/µL
    "hemoglobin": (6.0, 22.0),       # g/dL
    "platelets": (20.0, 1000.0),     # x10^3/µL
    "mcv": (60.0, 120.0),            # fL
    "mch": (20.0, 40.0),             # pg
    "mchc": (28.0, 38.0),            # g/dL
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
    return {
        "status": "ok",
        "service": "blood-health-backend"
    }


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

    alerts = []

    for idx, record in enumerate(payload.records):
        data = record.dict()

        for field, value in data.items():
            if field == "date":
                continue

            if field not in EXTREME_LIMITS:
                continue

            low, high = EXTREME_LIMITS[field]

            if value < low or value > high:
                alerts.append(
                    f"Record {idx+1}: {field.upper()} = {value} "
                    f"is in an extreme clinical range. "
                    f"Please consult a doctor at the earliest."
                )

    result = calculate_risk(payload.records)

    return {
        "patient_id": payload.patient_id,
        **result,
        "alerts": alerts
    }
