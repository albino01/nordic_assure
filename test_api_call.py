# test_api_call.py
# Run with: pytest -q

# Notes:
# - This uses FastAPI's TestClient (no network, no deployed URL required).
# - It assumes your FastAPI app is defined in api.py as: app = FastAPI(...)
# - It assumes you have artifacts/ (fraud_model.pkl + model_meta.json) present.

import pytest
from fastapi.testclient import TestClient
#import nordic_assure._conftest as _conftest

from api import app

client = TestClient(app)


import requests
BASE_URL = "http://127.0.0.1:8000"

def test_health_live():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_endpoint_returns_required_fields():
    # Minimal payload: include only fields you are confident exist in your trained dataset.
    # IMPORTANT: these must match column names used during training.
    payload = {
        "claim_id": "CLM-12345",
        "Month": "Aug",
        "WeekOfMonth": 2,
        "Make": "Honda",
        "AccidentArea": "Urban",
        "Age": 31,
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # Required fields
    assert data["claim_id"] == "CLM-12345"
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "action" in data

    # Type / range checks
    assert isinstance(data["fraud_probability"], float)
    assert 0.0 <= data["fraud_probability"] <= 1.0

    # Value checks
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert data["action"] in {
        "AUTO_APPROVE_PAYMENT_QUEUE",
        "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER",
        "BLOCK_AND_INVESTIGATE",
    }


@pytest.mark.parametrize(
    "prob, expected_risk, expected_action",
    [
        (0.10, "LOW", "AUTO_APPROVE_PAYMENT_QUEUE"),
        (0.30, "MEDIUM", "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER"),
        (0.50, "MEDIUM", "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER"),
        (0.70, "MEDIUM", "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER"),
        (0.71, "HIGH", "BLOCK_AND_INVESTIGATE"),
    ],
)
def test_routing_logic_via_monkeypatch(prob, expected_risk, expected_action, monkeypatch):
    """
    This test isolates routing correctness even if the model output changes.
    We monkeypatch model.predict_proba to return a controlled probability.
    """
    from api import model as loaded_model

    def fake_predict_proba(X):
        # Return shape (n, 2): [P(class0), P(class1)]
        import numpy as np
        n = len(X)
        return np.tile([1.0 - prob, prob], (n, 1))

    monkeypatch.setattr(loaded_model, "predict_proba", fake_predict_proba)

    payload = {
        "claim_id": "CLM-ROUTE",
        "Month": "Aug",
        "WeekOfMonth": 2,
        "Make": "Honda",
        "AccidentArea": "Urban",
        "Age": 31,
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["risk_level"] == expected_risk
    assert data["action"] == expected_action
    assert abs(data["fraud_probability"] - prob) < 1e-9


def test_predict_rejects_bad_payload():
    # Missing all features: should fail (most likely inside sklearn pipeline)
    r = client.post("/predict", json={"claim_id": "CLM-BAD"})
    assert r.status_code == 400
    assert "Bad request or model error" in r.json()["detail"]



def _payload_minimal():
    # Must match training column names (use a small, known-good subset)
    return {
        "claim_id": "CLM-ROUTE",
        "Month": "Aug",
        "WeekOfMonth": 2,
        "Make": "Honda",
        "AccidentArea": "Urban",
        "Age": 31,
    }


def _monkeypatch_prob(monkeypatch, prob: float):
    """Patch api.model.predict_proba to return a controlled fraud probability."""
    from api import model as loaded_model
    import numpy as np

    def fake_predict_proba(X):
        n = len(X)
        # columns: [P(class0), P(class1)]
        return np.tile([1.0 - prob, prob], (n, 1))

    monkeypatch.setattr(loaded_model, "predict_proba", fake_predict_proba)


def test_predict_routes_low(monkeypatch):
    _monkeypatch_prob(monkeypatch, prob=0.10)

    r = client.post("/predict", json=_payload_minimal())
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["risk_level"] == "LOW"
    assert data["action"] == "AUTO_APPROVE_PAYMENT_QUEUE"
    assert abs(data["fraud_probability"] - 0.10) < 1e-9


def test_predict_routes_medium(monkeypatch):
    # pick something clearly in the middle of [LOW, HIGH]
    _monkeypatch_prob(monkeypatch, prob=0.50)

    r = client.post("/predict", json=_payload_minimal())
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["risk_level"] == "MEDIUM"
    assert data["action"] == "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER"
    assert abs(data["fraud_probability"] - 0.50) < 1e-9


def test_predict_routes_high(monkeypatch):
    _monkeypatch_prob(monkeypatch, prob=0.90)

    r = client.post("/predict", json=_payload_minimal())
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["risk_level"] == "HIGH"
    assert data["action"] == "BLOCK_AND_INVESTIGATE"
    assert abs(data["fraud_probability"] - 0.90) < 1e-9


# tests/test_api_data_cases.py
# Run: pytest -q
#
# These payloads are real rows from fraud_oracle.csv (diverse/orthogonal coverage).
# They include all feature columns (except target) + claim_id.
# One case intentionally omits some fields to verify missing-value handling.

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


ORTHOGONAL_PAYLOADS = [
    # Fraud row (FraudFound_P=1) — source row index: 2141
    {
        "AccidentArea": "Urban",
        "AddressChange_Claim": "no change",
        "Age": 35,
        "AgeOfPolicyHolder": "31 to 35",
        "AgeOfVehicle": "6 years",
        "AgentType": "External",
        "BasePolicy": "All Perils",
        "DayOfWeek": "Friday",
        "DayOfWeekClaimed": "Friday",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "Deductible": 400,
        "DriverRating": 2,
        "Fault": "Policy Holder",
        "Make": "Chevrolet",
        "MaritalStatus": "Single",
        "Month": "Mar",
        "MonthClaimed": "Apr",
        "NumberOfCars": "1 vehicle",
        "NumberOfSuppliments": "none",
        "PolicyNumber": 2142,
        "PolicyType": "Sedan - All Perils",
        "RepNumber": 4,
        "Sex": "Male",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "WeekOfMonth": 3,
        "WeekOfMonthClaimed": 1,
        "WitnessPresent": "No",
        "Year": 1994,
        "claim_id": "ROW-2141",
    },

    # Fraud row (FraudFound_P=1) — source row index: 5355
    {
        "AccidentArea": "Rural",
        "AddressChange_Claim": "no change",
        "Age": 24,
        "AgeOfPolicyHolder": "21 to 25",
        "AgeOfVehicle": "new",
        "AgentType": "Internal",
        "BasePolicy": "Collision",
        "DayOfWeek": "Monday",
        "DayOfWeekClaimed": "Monday",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "Deductible": 400,
        "DriverRating": 1,
        "Fault": "Third Party",
        "Make": "Honda",
        "MaritalStatus": "Married",
        "Month": "Nov",
        "MonthClaimed": "Nov",
        "NumberOfCars": "2 vehicles",
        "NumberOfSuppliments": "more than 5",
        "PolicyNumber": 5356,
        "PolicyType": "Sedan - Collision",
        "RepNumber": 5,
        "Sex": "Female",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "WeekOfMonth": 2,
        "WeekOfMonthClaimed": 2,
        "WitnessPresent": "Yes",
        "PoliceReportFiled": "Yes",
        "Year": 1994,
        "claim_id": "ROW-5355",
    },

    # Fraud row (FraudFound_P=1) — source row index: 13447
    {
        "AccidentArea": "Urban",
        "AddressChange_Claim": "under 6 months",
        "Age": 41,
        "AgeOfPolicyHolder": "41 to 50",
        "AgeOfVehicle": "3 years",
        "AgentType": "External",
        "BasePolicy": "Liability",
        "DayOfWeek": "Thursday",
        "DayOfWeekClaimed": "Friday",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "Deductible": 400,
        "DriverRating": 2,
        "Fault": "Policy Holder",
        "Make": "Mazda",
        "MaritalStatus": "Single",
        "Month": "Dec",
        "MonthClaimed": "Dec",
        "NumberOfCars": "1 vehicle",
        "NumberOfSuppliments": "none",
        "PolicyNumber": 13448,
        "PolicyType": "Sedan - Liability",
        "RepNumber": 12,
        "Sex": "Male",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "WeekOfMonth": 4,
        "WeekOfMonthClaimed": 4,
        "WitnessPresent": "No",
        "PoliceReportFiled": "No",
        "Year": 1996,
        "claim_id": "ROW-13447",
    },

    # Non-fraud row (FraudFound_P=0) — source row index: 4103
    {
        "AccidentArea": "Rural",
        "AddressChange_Claim": "no change",
        "Age": 29,
        "AgeOfPolicyHolder": "26 to 30",
        "AgeOfVehicle": "7 years",
        "AgentType": "External",
        "BasePolicy": "Collision",
        "DayOfWeek": "Tuesday",
        "DayOfWeekClaimed": "Wednesday",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "Deductible": 400,
        "DriverRating": 3,
        "Fault": "Third Party",
        "Make": "Toyota",
        "MaritalStatus": "Married",
        "Month": "Jan",
        "MonthClaimed": "Jan",
        "NumberOfCars": "1 vehicle",
        "NumberOfSuppliments": "none",
        "PolicyNumber": 4104,
        "PolicyType": "Sedan - Collision",
        "RepNumber": 4,
        "Sex": "Male",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "WeekOfMonth": 2,
        "WeekOfMonthClaimed": 3,
        "WitnessPresent": "No",
        "PoliceReportFiled": "No",
        "Year": 1994,
        "claim_id": "ROW-4103",
    },

    # Non-fraud row (FraudFound_P=0) — source row index: 10038
    {
        "AccidentArea": "Urban",
        "AddressChange_Claim": "no change",
        "Age": 54,
        "AgeOfPolicyHolder": "51 to 65",
        "AgeOfVehicle": "more than 7",
        "AgentType": "Internal",
        "BasePolicy": "All Perils",
        "DayOfWeek": "Saturday",
        "DayOfWeekClaimed": "Monday",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "Deductible": 400,
        "DriverRating": 1,
        "Fault": "Policy Holder",
        "Make": "Ford",
        "MaritalStatus": "Widow",
        "Month": "Oct",
        "MonthClaimed": "Oct",
        "NumberOfCars": "1 vehicle",
        "NumberOfSuppliments": "none",
        "PolicyNumber": 10039,
        "PolicyType": "Sedan - All Perils",
        "RepNumber": 9,
        "Sex": "Female",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "WeekOfMonth": 4,
        "WeekOfMonthClaimed": 4,
        "WitnessPresent": "No",
        "PoliceReportFiled": "No",
        "Year": 1996,
        "claim_id": "ROW-10038",
    },

    # Non-fraud row (FraudFound_P=0) — source row index: 11187
    {
        "AccidentArea": "Urban",
        "AddressChange_Claim": "no change",
        "Age": 22,
        "AgeOfPolicyHolder": "21 to 25",
        "AgeOfVehicle": "2 years",
        "AgentType": "External",
        "BasePolicy": "Liability",
        "DayOfWeek": "Wednesday",
        "DayOfWeekClaimed": "Wednesday",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "Deductible": 400,
        "DriverRating": 2,
        "Fault": "Third Party",
        "Make": "Nissan",
        "MaritalStatus": "Single",
        "Month": "May",
        "MonthClaimed": "May",
        "NumberOfCars": "1 vehicle",
        "NumberOfSuppliments": "none",
        "PolicyNumber": 11188,
        "PolicyType": "Sedan - Liability",
        "RepNumber": 11,
        "Sex": "Male",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "WeekOfMonth": 1,
        "WeekOfMonthClaimed": 1,
        "WitnessPresent": "No",
        "PoliceReportFiled": "No",
        "Year": 1996,
        "claim_id": "ROW-11187",
    },

    # Missing-fields variant (derived from row 2141, but omits some fields)
    {
        "Month": "Mar",
        "WeekOfMonth": 3,
        "DayOfWeek": "Friday",
        "Make": "Chevrolet",
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Friday",
        "MonthClaimed": "Apr",
        "WeekOfMonthClaimed": 1,
        "Sex": "Male",
        "Age": 35,
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - All Perils",
        "VehicleCategory": "Sedan",
        "PolicyNumber": 2142,
        "RepNumber": 4,
        "Deductible": 400,
        "DriverRating": 2,
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "AgeOfVehicle": "6 years",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "1 vehicle",
        "Year": 1994,
        "BasePolicy": "All Perils",
        "claim_id": "MISSING-FIELDS",
    },
]


@pytest.mark.parametrize("payload", ORTHOGONAL_PAYLOADS)
def test_predict_with_orthogonal_dataset_payloads(payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    assert data.get("claim_id") == payload.get("claim_id")
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "action" in data

    assert isinstance(data["fraud_probability"], float)
    assert 0.0 <= data["fraud_probability"] <= 1.0

    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert data["action"] in {
        "AUTO_APPROVE_PAYMENT_QUEUE",
        "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER",
        "BLOCK_AND_INVESTIGATE",
    }