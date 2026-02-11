#!/bin/bash

# test_gcloud_api.sh

curl https://nordic-assure-api-883165044435.europe-west1.run.app/health

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