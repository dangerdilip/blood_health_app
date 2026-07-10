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
            "wbc": 6.0,         # ×10³/µL
            "platelets": 240,    # ×10³/µL
            "rbc": 4.7,
            "mcv": 90,
            "mch": 30,
            "mchc": 34
        }]
    }
    r = client.post("/risk/predict", json=payload)
    data = r.json()
    assert r.status_code == 200
    assert "blood_status" in data
    assert "normal_ranges" in data

def test_zero_values_rejected():
    """Submitting all-zero values should return an error, not 'Critical abnormalities'."""
    payload = {
        "patient_id": "T2",
        "records": [{
            "date": "2024-01-01",
            "hemoglobin": 0,
            "wbc": 0,
            "platelets": 0,
            "rbc": 0,
            "mcv": 0,
            "mch": 0,
            "mchc": 0
        }]
    }
    r = client.post("/risk/predict", json=payload)
    data = r.json()
    assert r.status_code == 200
    assert data["error"] is True
    assert data["blood_status"] == "Invalid data"
    assert "HEMOGLOBIN" in data["alerts"][0]
