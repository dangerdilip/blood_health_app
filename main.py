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

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="Blood Health Risk API")

# ---------------- CORS ----------------
# (Frontend domain can be restricted later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

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


# ---------------- ENDPOINTS ----------------
@app.post("/risk/predict")
def predict_risk(payload: PatientCBCRequest):
    result = calculate_risk(payload.records)
    return {
        "patient_id": payload.patient_id,
        **result
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
