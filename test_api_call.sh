#!/bash/bin
curl -X POST https://YOUR_CLOUD_RUN_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-12345",
    "Month": "Aug",
    "WeekOfMonth": 2,
    "Make": "Honda",
    "AccidentArea": "Urban",
    "Age": 31
  }'


# expected response
#  {
#  "claim_id": "CLM-12345",
#  "fraud_probability": 0.73,
#  "risk_level": "HIGH",
#  "action": "BLOCK_AND_INVESTIGATE"
#}
