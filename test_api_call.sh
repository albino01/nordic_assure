#!/bash/bin
curl -X POST "http://127.0.0.1:8000/predict" \
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

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"CLM-HIGH","claim_amount":999999999}'


curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"CLM-MED","claim_amount":50000}'

# expected response
#  {
#  "claim_id": "CLM-12345",
#  "fraud_probability": 0.73,
#  "risk_level": "HIGH",
#  "action": "BLOCK_AND_INVESTIGATE"
#}
