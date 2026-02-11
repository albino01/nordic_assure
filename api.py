import json
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException

MODEL_PATH = "artifacts/fraud_model.pkl"
META_PATH = "artifacts/model_meta.json"

app = FastAPI(title="NordicAssure Fraud API", version="1.0")

# Load model (pickle)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load metadata (JSON)
with open(META_PATH, "r") as f:
    meta = json.load(f)

LOW = float(meta.get("low_threshold", 0.30))
HIGH = float(meta.get("high_threshold", 0.70))
FEATURE_COLS = meta.get("feature_columns")  # list of training columns


def route(prob: float):
    if prob < LOW:
        return ("LOW", "AUTO_APPROVE_PAYMENT_QUEUE")
    elif prob <= HIGH:
        return ("MEDIUM", "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER")
    else:
        return ("HIGH", "BLOCK_AND_INVESTIGATE")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": meta.get("model_version", "unknown")}

@app.post("/predict")
def predict(payload: dict):
    try:
        claim_id = payload.get("claim_id", None)

        features = dict(payload)
        features.pop("claim_id", None)

        if FEATURE_COLS:
            row = {col: features.get(col, None) for col in FEATURE_COLS}
        else:
            row = features

        X = pd.DataFrame([row])   # <-- use row, not features
        prob = float(model.predict_proba(X)[:, 1][0])
        risk_level, action = route(prob)

        resp = {
            "fraud_probability": prob,
            "risk_level": risk_level,
            "action": action,
        }
        if claim_id is not None:
            resp["claim_id"] = claim_id

        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request or model error: {str(e)}")
    



# Add these endpoints to your existing FastAPI app (no /metrics)

@app.get("/meta")
def meta_info():
    """
    Returns model versioning + routing thresholds + expected feature columns
    so clients can build correct requests.
    """
    return {
        "model_version": meta.get("model_version", "unknown"),
        "thresholds": {
            "low": LOW,
            "high": HIGH,
        },
        "feature_columns": FEATURE_COLS,  # may be null if not provided in model_meta.json
        "routing": [
            {"risk_level": "LOW", "condition": f"p < {LOW}", "action": "AUTO_APPROVE_PAYMENT_QUEUE"},
            {"risk_level": "MEDIUM", "condition": f"{LOW} <= p <= {HIGH}", "action": "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER"},
            {"risk_level": "HIGH", "condition": f"p > {HIGH}", "action": "BLOCK_AND_INVESTIGATE"},
        ],
        "notes": [
            "If feature_columns is present, extra request fields are ignored and missing fields become null.",
            "If feature_columns is null, all fields (except claim_id) are used as features.",
        ],
    }


@app.get("/schema")
def schema():
    """
    Simple schema-like output to help clients.
    (If you later store dtypes in meta, you can enrich this.)
    """
    return {
        "request": {
            "content_type": "application/json",
            "body": {
                "type": "object",
                "optional": ["claim_id"],
                "properties": (
                    {c: {"type": ["string", "number", "boolean", "null"]} for c in FEATURE_COLS}
                    if FEATURE_COLS else
                    {"<feature_name>": {"type": ["string", "number", "boolean", "null"]}}
                ),
                "notes": [
                    "Send one JSON object per claim.",
                    "Use null for unknown/missing values.",
                    "If feature_columns is present, include those keys; unknown keys are ignored.",
                ],
            },
        },
        "response": {
            "type": "object",
            "properties": {
                "claim_id": {"type": ["string", "number"], "nullable": True},
                "fraud_probability": {"type": "number", "range": [0.0, 1.0]},
                "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                "action": {
                    "type": "string",
                    "enum": [
                        "AUTO_APPROVE_PAYMENT_QUEUE",
                        "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER",
                        "BLOCK_AND_INVESTIGATE",
                    ],
                },
            },
        },
    }