import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_ready"] is True
    assert "metrics_summary" in data

def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_metrics" in data
    assert "MAE" in data["test_metrics"]
    assert "RMSE" in data["test_metrics"]
    assert "R2" in data["test_metrics"]

def test_valid_prediction():
    payload = {
        "age": 21,
        "gender": "Male",
        "country": "India",
        "academic_level": "Undergraduate",
        "most_used_platform": "Instagram",
        "purpose_of_use": "Entertainment",
        "avg_daily_usage_hours": 5.5,
        "daily_unlocks": 60,
        "study_hours": 4.0,
        "physical_activity_hours": 1.0,
        "sleep_hours_per_night": 7.0,
        "stress_level": "Medium"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert 0.0 <= data["score"] <= 10.0
    assert 0.0 <= data["predicted_mental_health_score"] <= 10.0
    assert data["band"] in ["Signal: strained", "Signal: balanced", "Signal: strong"]
    assert isinstance(data["context"], str)

def test_unlisted_country_prediction():
    payload = {
        "age": 25,
        "gender": "Female",
        "country": "Switzerland",  # Not in top 10 -> will map to 'Other'
        "academic_level": "Graduate",
        "most_used_platform": "LinkedIn",
        "purpose_of_use": "Networking",
        "avg_daily_usage_hours": 3.0,
        "daily_unlocks": 40,
        "study_hours": 6.0,
        "physical_activity_hours": 2.0,
        "sleep_hours_per_night": 8.0,
        "stress_level": "Low"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert 0.0 <= data["score"] <= 10.0

def test_validation_missing_field():
    payload = {
        "age": 21,
        "gender": "Male"
        # missing other required fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert "error" in data

def test_validation_out_of_range():
    payload = {
        "age": 150,  # Invalid: max is 100
        "gender": "Male",
        "country": "USA",
        "academic_level": "Undergraduate",
        "most_used_platform": "Instagram",
        "purpose_of_use": "Entertainment",
        "avg_daily_usage_hours": 28.0,  # Invalid: max is 24
        "daily_unlocks": -5,  # Invalid: min is 0
        "study_hours": 4.0,
        "physical_activity_hours": 1.0,
        "sleep_hours_per_night": 7.0,
        "stress_level": "InvalidStress"  # Invalid
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
