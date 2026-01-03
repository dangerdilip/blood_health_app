from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_single_cbc():
    payload = {
        "patient_id": "T1",
        "records": [{
            "date": "2024-01-01",
            "hemoglobin": 14,
            "wbc": 6000,
            "platelets": 240000,
            "rbc": 4.7,
            "mcv": 90,
            "mch": 30,
            "mchc": 34
        }]
    }
    r = client.post("/risk/predict", json=payload)
    assert "risk_score" not in r.json()
