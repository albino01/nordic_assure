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

        # Remove non-feature fields if present
        features = dict(payload)
        features.pop("claim_id", None)

        X = pd.DataFrame([features])
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