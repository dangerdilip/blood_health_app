from services.risk_service import calculate_risk

class CBC:
    def __init__(self, d):
        self._d = d
    def dict(self):
        return self._d

cbc1 = CBC({
    "date": "2026-01-01",
    "hemoglobin": 9.5,
    "wbc": 11.5,
    "platelets": 110000,
    "rbc": 3.2,
    "mcv": 72,
    "mch": 22,
    "mchc": 30
})

cbc2 = CBC({
    "date": "2026-01-05",
    "hemoglobin": 8.9,
    "wbc": 12.2,
    "platelets": 95000,
    "rbc": 3.0,
    "mcv": 70,
    "mch": 21,
    "mchc": 29
})

print("\n--- SINGLE CBC RECORD ---")
print(calculate_risk([cbc1]))

print("\n--- TWO CBC RECORDS ---")
print(calculate_risk([cbc1, cbc2]))
