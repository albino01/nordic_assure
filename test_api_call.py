# tests/test_api.py
# Run with: pytest -q
#
# Notes:
# - This uses FastAPI's TestClient (no network, no deployed URL required).
# - It assumes your FastAPI app is defined in api.py as: app = FastAPI(...)
# - It assumes you have artifacts/ (fraud_model.pkl + model_meta.json) present.

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


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