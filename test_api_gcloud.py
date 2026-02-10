# tests/test_api_gcloud.py
# Run with:
#   pytest -q
# or:
#   BASE_URL="https://...run.app" pytest -q
#
# This file tests the *deployed* Cloud Run service over HTTP.

import os
import requests

DEFAULT_BASE_URL = "https://nordic-assure-api-883165044435.europe-west1.run.app"
BASE_URL = os.getenv("BASE_URL", DEFAULT_BASE_URL).rstrip("/")

TIMEOUT = 30


def _post(path: str, payload: dict):
    return requests.post(f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT)


def _get(path: str):
    return requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)


def _assert_predict_response(data: dict, claim_id: str | None = None):
    if claim_id is not None:
        assert data.get("claim_id") == claim_id

    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "action" in data

    assert isinstance(data["fraud_probability"], (float, int))
    assert 0.0 <= float(data["fraud_probability"]) <= 1.0

    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert data["action"] in {
        "AUTO_APPROVE_PAYMENT_QUEUE",
        "FLAG_MANUAL_REVIEW_NOTIFY_ADJUSTER",
        "BLOCK_AND_INVESTIGATE",
    }


def test_health():
    r = _get("/health")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_minimal_payload_returns_required_fields():
    payload = {
        "claim_id": "CLM-12345",
        "Month": "Aug",
        "WeekOfMonth": 2,
        "Make": "Honda",
        "AccidentArea": "Urban",
        "Age": 31,
    }

    r = _post("/predict", payload)
    assert r.status_code == 200, r.text
    data = r.json()
    _assert_predict_response(data, claim_id="CLM-12345")


def test_predict_fraud_urban_allperils():
    payload = {
        "claim_id": "ROW-2141",
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
    }

    r = _post("/predict", payload)
    assert r.status_code == 200, r.text
    _assert_predict_response(r.json(), claim_id="ROW-2141")


def test_predict_fraud_rural_collision():
    payload = {
        "claim_id": "ROW-5355",
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
    }

    r = _post("/predict", payload)
    assert r.status_code == 200, r.text
    _assert_predict_response(r.json(), claim_id="ROW-5355")


def test_predict_fraud_address_change_liability():
    payload = {
        "claim_id": "ROW-13447",
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
    }

    r = _post("/predict", payload)
    assert r.status_code == 200, r.text
    _assert_predict_response(r.json(), claim_id="ROW-13447")


def test_predict_nonfraud_rural_collision():
    payload = {
        "claim_id": "ROW-4103",
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
    }

    r = _post("/predict", payload)
    assert r.status_code == 200, r.text
    _assert_predict_response(r.json(), claim_id="ROW-4103")


def test_predict_nonfraud_older_policyholder():
    payload = {
        "claim_id": "ROW-10038",
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
    }

    r = _post("/predict", payload)
    assert r.status_code == 200, r.text
    _assert_predict_response(r.json(), claim_id="ROW-10038")


def test_predict_nonfraud_young_driver_liability():
    payload = {
        "claim_id": "ROW-11187",
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
    }

    r = _post("/predict", payload)
    assert r.status_code == 200, r.text
    _assert_predict_response(r.json(), claim_id="ROW-11187")


def test_predict_missing_fields_should_still_work():
    payload = {
        "claim_id": "MISSING-FIELDS",
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
    }

    r = _post("/predict", payload)
    assert r.status_code == 200, r.text
    _assert_predict_response(r.json(), claim_id="MISSING-FIELDS")