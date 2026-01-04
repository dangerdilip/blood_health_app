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
    version="1.0.0"
)

# ---------------- CORS ----------------
# Allow all for now; restrict after frontend deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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


# ---------------- ROOT / HEALTH ----------------
@app.get("/")
def root():
    """
    Render + browser health check.
    Prevents confusing 404 logs.
    """
    return {
        "status": "ok",
        "service": "blood-health-backend",
        "message": "API is running"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------- MAIN API ----------------
@app.post("/risk/predict")
def predict_risk(payload: PatientCBCRequest):
    """
    Main risk prediction endpoint.
    ML model is lazy-loaded inside service layer.
    """
    logger.info(
        f"Received risk request for patient_id={payload.patient_id}, "
        f"records={len(payload.records)}"
    )

    result = calculate_risk(payload.records)

    return {
        "patient_id": payload.patient_id,
        **result
    }
