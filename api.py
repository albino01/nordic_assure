import json
import os
import time
import pickle
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/fraud_model.pkl")
META_PATH  = os.getenv("META_PATH",  "artifacts/model_meta.json")

APP_NAME = "NordicAssure Fraud API"
APP_VERSION = "1.0"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("nordic_assure")

app = FastAPI(title=APP_NAME, version=APP_VERSION)

model = None
meta = {}
LOW = 0.30
HIGH = 0.70
FEATURE_COLS = None


def route(prob: float):
    if prob < LOW:
        return ("LOW", "AUTO_APPROVE_PAYMENT_QUEUE")
    elif prob <= HIGH:
        return ("MEDIUM", "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER")
    else:
        return ("HIGH", "BLOCK_AND_INVESTIGATE")


@app.on_event("startup")
def load_artifacts():
    global model, meta, LOW, HIGH, FEATURE_COLS
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        LOW = float(meta.get("low_threshold", 0.30))
        HIGH = float(meta.get("high_threshold", 0.70))
        FEATURE_COLS = meta.get("feature_columns")

        logger.info(f"Loaded model_version={meta.get('model_version','unknown')} LOW={LOW} HIGH={HIGH}")
    except Exception as e:
        logger.exception(f"Failed to load artifacts: {e}")
        model = None
        meta = {}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    ms = int((time.time() - start) * 1000)
    logger.info(f"{request.method} {request.url.path} {resp.status_code} {ms}ms")
    return resp


@app.get("/health")
def health():
    # liveness: process is running
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "model_version": meta.get("model_version", "unknown") if meta else "unloaded",
    }


@app.get("/ready")
def ready():
    # readiness: model loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": meta.get("model_version", "unknown")}


@app.post("/predict")
def predict(payload: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        claim_id = payload.get("claim_id", None)

        features = dict(payload)
        features.pop("claim_id", None)

        if FEATURE_COLS:
            row = {col: features.get(col, None) for col in FEATURE_COLS}
        else:
            if not features:
                raise HTTPException(status_code=400, detail="No feature fields provided")
            row = features

        X = pd.DataFrame([row])
        prob = float(model.predict_proba(X)[:, 1][0])
        risk_level, action = route(prob)

        resp = {"fraud_probability": prob, "risk_level": risk_level, "action": action}
        if claim_id is not None:
            resp["claim_id"] = claim_id
        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request or model error: {str(e)}")


@app.get("/meta")
def meta_info():
    if not meta:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version": meta.get("model_version", "unknown"),
        "thresholds": {"low": LOW, "high": HIGH},
        "feature_columns": FEATURE_COLS,
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
    props = (
        {c: {"type": ["string", "number", "boolean", "null"]} for c in FEATURE_COLS}
        if FEATURE_COLS else
        {"<feature_name>": {"type": ["string", "number", "boolean", "null"]}}
    )
    return {
        "request": {
            "content_type": "application/json",
            "body": {
                "type": "object",
                "optional": ["claim_id"],
                "properties": props,
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