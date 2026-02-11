# nordic_assure

**nordic_assure** is a machine-learning project that builds a **binary classifier** to predict whether a vehicle insurance claim is **fraudulent** or **legitimate**.  

---

## Quickstart

### 1) Clone
```bash
git clone https://github.com/albino01/nordic_assure.git
cd nordic_assure
```

---

## The training script

The training script builds and evaluates an **XGBoost-based vehicle fraud classifier** using `fraud_oracle.csv`, then exports the trained model and lightweight metadata for use in an API.

### 1) Load and split data
- Loads `fraud_oracle.csv` with pandas
- Uses `FraudFound_P` as the binary target `0/1`
- Splits into **train (80%)** and **test (20%)** using a **stratified split** so the fraud rate stays consistent

### 2) Preprocess numeric and categorical features
Creates a scikit-learn preprocessing pipeline:
- **Numeric columns:** missing values → **median** imputation
- **Categorical columns:** missing values → **most frequent** imputation, then **One-Hot Encoding**
  - `handle_unknown="ignore"` prevents inference from breaking on unseen categories

Preprocessing is wrapped in a `ColumnTransformer` and combined with the model in a single `Pipeline`, ensuring the same transforms are applied during training and inference.

### 3) Train an imbalanced fraud model (XGBoost)
Fraud is typically class-imbalanced. The script:
- Computes `scale_pos_weight = (# non-fraud) / (# fraud)` on the training set
- Passes it to `XGBClassifier` to increase sensitivity to the fraud class
- Optimizes using **PR-AUC** (`eval_metric="aucpr"`), which is often more informative than ROC-AUC for rare-event detection

### 4) Cross-validation and test evaluation (PR-AUC)
- Runs **5-fold Stratified Cross-Validation** on the training split and prints mean ± std **PR-AUC**
- Fits the final model on the training set and reports **Test PR-AUC** on the holdout set

### 5) Threshold selection for a precision-first fraud screen
Instead of the default 0.5 cutoff, the script:
- Computes the **precision–recall curve** on the test set
- Selects a threshold targeting ~**70% precision** (if achievable)
- Prints a `classification_report` at that threshold

This supports workflows where **false positives are costly** and cases are sent to a manual review queue.

### 6) Risk bucketing for triage
Converts predicted fraud probability `p` into 3 risk tiers:
- **Low:** `p < 0.30`
- **Medium:** `0.30 ≤ p < 0.70`
- **High:** `p ≥ 0.70`

Prints a summary table with:
- number of claims per tier
- number of frauds per tier
- fraud rate per tier

Also reports **fraud recall in Medium + High**, i.e., how many fraud cases would be flagged for investigation.

### 7) Export artifacts for deployment
Saves:
- `artifacts/fraud_model.pkl` — full preprocessing + model pipeline (ready for inference)
- `artifacts/model_meta.json` — model version and thresholds for API routing/versioning

---

## Overview (API)

Endpoints:
- `GET /health` — service health and model version
- `POST /predict` — score a single claim and return fraud probability + routing decision
- `GET /meta` — model thresholds, version, routing, expected features
- `GET /schema` — lightweight schema-like output for clients

The service loads:
- Model: `artifacts/fraud_model.pkl`
- Metadata: `artifacts/model_meta.json`

Metadata controls:
- `model_version`
- `low_threshold` (default 0.30)
- `high_threshold` (default 0.70)
- `feature_columns` (optional list of expected input columns)

---

## Base URL

Examples below assume:
- Local development: `http://localhost:8000`
- Google Cloud Platform: `https://nordic-assure-api-883165044435.europe-west1.run.app`

---

## Authentication

No authentication is implemented in the provided code.  
If your deployment adds auth (API keys, OAuth, IAM, etc.), follow your platform’s rules.

---

## Endpoint: `GET /health`

### Purpose
Verify the service is running and retrieve the model version.

### Request
No request body.

### Response (200 OK)
```json
{
  "status": "ok",
  "model_version": "1.0"
}
```

---

## Endpoint: `POST /predict`

### Purpose
Score a single claim and return fraud probability + routing decision.

### Request body
A single JSON object containing:
- Optional: `claim_id` (string/number)
- Feature key/value pairs (all other fields are treated as features)

If `feature_columns` is present in metadata:
- extra request fields are ignored
- missing fields become `null`

### Example request
```json
{
  "claim_id": "CLM-12345",
  "VehiclePrice": 21000,
  "Make": "Toyota",
  "AccidentArea": "Urban"
}
```

### Example response (200 OK)
```json
{
  "claim_id": "CLM-12345",
  "fraud_probability": 0.8421,
  "risk_level": "HIGH",
  "action": "BLOCK_AND_INVESTIGATE"
}
```

### Error response (400)
```json
{
  "detail": "Bad request or model error: <details>"
}
```

---

## Model artifacts and versioning (manual “active” model)

This service loads the **active production model** from fixed paths:

- `artifacts/fraud_model.pkl` — trained Pipeline (preprocess + XGBoost)
- `artifacts/model_meta.json` — thresholds + model version + (optional) feature columns

### How versioning works
- Each training run can be saved into a versioned folder for archive, e.g.:
  - `artifacts/models/NORDIC-ASSURE_0001/`
  - `artifacts/models/NORDIC-ASSURE_0001/`
- **Production is always the root artifacts**:
  - `artifacts/fraud_model.pkl`
  - `artifacts/model_meta.json`
- To “promote” a model version, you **manually copy** the desired versioned artifacts into the root.

Example promotion:
```bash
cp artifacts/v0007/fraud_model.pkl artifacts/fraud_model.pkl
cp artifacts/v0007/model_meta.json artifacts/model_meta.json
```

---

## Deployment (GCP Cloud Run)

> Note: These instructions require IAM Admin permissions and a tag `environment=development` added to the GCP project.

Install the CLI:
```bash
sudo snap install google-cloud-cli --classic
```

### Example tag binding
```bash

gcloud resource-manager tags bindings create \
  --parent=//cloudresourcemanager.googleapis.com/projects/883165044435 \
  --tag-value=tagValues/281481363704394

gcloud resource-manager tags bindings list \
  --parent=//cloudresourcemanager.googleapis.com/projects/883165044435

```

---

## Deploy script

```bash

#!/bin/bash
# deploy.sh

gcloud config set project nordic-assure
gcloud billing projects link nordic-assure --billing-account XXXXXXXXXXXXXXXXXXXX
gcloud config set run/region europe-west1
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy nordic-assure-api \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 2

```

---

## Test endpoint

```bash
#!/usr/bin/env bash

curl -X POST "https://nordic-assure-api-883165044435.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-12345",
    "policy_type": "Sport - Collision",
    "vehicle_category": "Sport",
    "incident_severity": "Major Damage",
    "authorities_contacted": "None",
    "incident_state": "NY",
    "claim_amount": 52080,
    "customer_age": 31
  }'
```

---

## Test suite

Run the following scripts to test the deployment:

```bash
./test_gcloud_api.sh              # runs a few curl commands
pytest -vvv test_gcloud_api.py    # verifies request/response for the predict endpoint
```

---

# NordicAssure Fraud API — Client Guide (v1.0)

This guide explains how to call the NordicAssure Fraud API and interpret responses.

---

## Overview

The API scores **one claim per request** and returns:

- `fraud_probability` — likelihood of fraud (0.0 to 1.0)
- `risk_level` — `LOW`, `MEDIUM`, or `HIGH`
- `action` — recommended routing action for operational handling

### Endpoints
- `GET /health` — service health + active model version
- `POST /predict` — score a claim
- `GET /meta` — model thresholds + routing rules + expected feature keys
- `GET /schema` — lightweight schema-like description for clients

---

## Base URL (Google Cloud Run)

Use the following base URL:

- `https://nordic-assure-api-883165044435.europe-west1.run.app`

---

## Authentication

Authentication depends on your deployment configuration.

If your environment uses:
- **API keys**: include the required header (NordicAssure will provide the header name/value format)
- **OAuth / IAM**: follow your organization’s access policy

If you are unsure, contact NordicAssure support to confirm the correct auth method.

---

## Content Type

All request bodies must be JSON:

- `Content-Type: application/json`

---

## Routing logic

The API applies two thresholds (`low_threshold`, `high_threshold`) to map probability `p` into risk levels:

- **LOW** if `p < low_threshold` → `AUTO_APPROVE_PAYMENT_QUEUE`
- **MEDIUM** if `low_threshold ≤ p ≤ high_threshold` → `FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER`
- **HIGH** if `p > high_threshold` → `BLOCK_AND_INVESTIGATE`

The active thresholds are returned by `GET /meta`.

---

## Endpoint: `GET /health`

### Purpose
Confirm the service is running and see the active model version.

### Request
No body.

### Response (200)
```json
{
  "status": "ok",
  "model_version": "1.0"
}
```

---

## Endpoint: `POST /predict`

### Purpose
Score a single claim and return fraud probability and a routing decision.

### Request body

A single JSON object containing:

- Optional: `claim_id` — your identifier for the claim (string or number)
- Feature fields — key/value pairs representing claim attributes

### Feature handling
- If `feature_columns` are defined in the active model metadata:
  - **extra fields are ignored**
  - **missing fields are treated as `null`**
- If `feature_columns` is not defined:
  - all fields (except `claim_id`) are used as model features

To discover the expected feature keys, call `GET /meta`.

### Example request
```json
{
  "claim_id": "CLM-12345",
  "policy_type": "Sport - Collision",
  "vehicle_category": "Sport",
  "incident_severity": "Major Damage",
  "authorities_contacted": "None",
  "incident_state": "NY",
  "claim_amount": 52080,
  "customer_age": 31
}
```

### Response (200)
```json
{
  "claim_id": "CLM-12345",
  "fraud_probability": 0.8421,
  "risk_level": "HIGH",
  "action": "BLOCK_AND_INVESTIGATE"
}
```

### Error response (400)
Returned for invalid payloads or inference errors.

```json
{
  "detail": "Bad request or model error: <details>"
}
```

---

## Endpoint: `GET /meta`

### Purpose
Return active model information to help clients build correct requests and interpret routing.

### Key field: `feature_columns`
`feature_columns` is the **input contract** for the model.

- If present, it lists the **exact JSON keys** the model expects.
- The API will:
  - **use only** these keys (ignore extra keys)
  - set any missing keys to **`null`** (missing values are handled by the model pipeline)

If `feature_columns` is `null`, the API will use all request fields (except `claim_id`) as model features.

### Response (200)
```json
{
  "model_version": "1.0",
  "thresholds": {
    "low": 0.3,
    "high": 0.7
  },
  "feature_columns": ["<feature_1>", "<feature_2>"],
  "routing": [
    {"risk_level": "LOW", "condition": "p < 0.3", "action": "AUTO_APPROVE_PAYMENT_QUEUE"},
    {"risk_level": "MEDIUM", "condition": "0.3 <= p <= 0.7", "action": "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER"},
    {"risk_level": "HIGH", "condition": "p > 0.7", "action": "BLOCK_AND_INVESTIGATE"}
  ],
  "notes": [
    "If feature_columns is present, extra request fields are ignored and missing fields become null.",
    "If feature_columns is null, all fields (except claim_id) are used as features."
  ]
}
```

---
## Endpoint: `GET /schema`

### Purpose
Return a lightweight schema-like description

The schema is **derived from** `feature_columns` in `artifacts/model_meta.json`

### Response (200)

```json
{
  "request": {
    "content_type": "application/json",
    "body": {
      "type": "object",
      "optional": ["claim_id"],
      "properties": {
        "<feature_1>": {"type": ["string", "number", "boolean", "null"]},
        "<feature_2>": {"type": ["string", "number", "boolean", "null"]}
      },
      "notes": [
        "Send one JSON object per claim.",
        "Use null for unknown/missing values.",
        "If feature_columns is present, include those keys; unknown keys are ignored."
      ]
    }
  },
  "response": {
    "type": "object",
    "properties": {
      "claim_id": {"type": ["string", "number"], "nullable": true},
      "fraud_probability": {"type": "number", "range": [0.0, 1.0]},
      "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
      "action": {
        "type": "string",
        "enum": [
          "AUTO_APPROVE_PAYMENT_QUEUE",
          "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER",
          "BLOCK_AND_INVESTIGATE"
        ]
      }
    }
  }
}

---

## cURL examples

### Health
```bash
curl -s https://nordic-assure-api-883165044435.europe-west1.run.app/health
```

### Predict
```bash
curl -s -X POST "https://nordic-assure-api-883165044435.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-12345",
    "policy_type": "Sport - Collision",
    "vehicle_category": "Sport",
    "incident_severity": "Major Damage",
    "authorities_contacted": "None",
    "incident_state": "NY",
    "claim_amount": 52080,
    "customer_age": 31
  }'
```

### Meta
```bash
curl -s https://nordic-assure-api-883165044435.europe-west1.run.app/meta
```

### Schema
```bash
curl -s https://nordic-assure-api-883165044435.europe-west1.run.app/schema
```

---

## Operational notes

- Send `null` for unknown values; the service will apply the same missing-value handling used by the model pipeline.
- One request scores one claim. If you need batch scoring, contact NordicAssure to discuss a batch endpoint.
- Record `model_version` from `/health` alongside decisions for auditability and monitoring.

---