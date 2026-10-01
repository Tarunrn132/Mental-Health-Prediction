import os
import json
import pytest
from fastapi.testclient import TestClient
from backend.app import app, TOP_COUNTRIES, FRONTEND_DIR

client = TestClient(app)

def test_static_html_served_on_html_accept():
    response = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text
    assert "Mental Health Signal" in response.text
    assert 'href="style.css"' in response.text
    assert 'src="script.js"' in response.text

def test_static_css_served():
    response = client.get("/style.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert "--pine:" in response.text
    assert ".gauge-fill" in response.text

def test_static_js_served():
    response = client.get("/script.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "GAUGE_ARC_LENGTH" in response.text
    assert "predict-form" in response.text

def test_metrics_api_and_data_integrity():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_metrics" in data
    assert data["test_metrics"]["R2"] > 0.85
    assert data["test_metrics"]["MAE"] < 0.50
    assert data["test_metrics"]["RMSE"] < 0.60
    assert data["selected_model"] == "Random Forest (Tuned)"

def test_all_12_platforms():
    platforms = [
        'Facebook', 'Instagram', 'Snapchat', 'Twitter', 'YouTube',
        'TikTok', 'LinkedIn', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
    ]
    for platform in platforms:
        payload = {
            "age": 22,
            "gender": "Female",
            "country": "UK",
            "academic_level": "Undergraduate",
            "most_used_platform": platform,
            "purpose_of_use": "Education",
            "avg_daily_usage_hours": 4.0,
            "daily_unlocks": 50,
            "study_hours": 5.0,
            "physical_activity_hours": 1.5,
            "sleep_hours_per_night": 7.5,
            "stress_level": "Medium"
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200, f"Failed for platform {platform}: {res.text}"
        res_data = res.json()
        assert res_data["success"] is True
        assert 0.0 <= res_data["score"] <= 10.0

def test_all_stress_levels_and_bands():
    levels = ["Low", "Medium", "High", "Very High"]
    for stress in levels:
        payload = {
            "age": 20,
            "gender": "Male",
            "country": "Canada",
            "academic_level": "Undergraduate",
            "most_used_platform": "YouTube",
            "purpose_of_use": "Entertainment",
            "avg_daily_usage_hours": 3.0,
            "daily_unlocks": 30,
            "study_hours": 6.0,
            "physical_activity_hours": 2.0,
            "sleep_hours_per_night": 8.0,
            "stress_level": stress
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        res_data = res.json()
        assert res_data["band"] in ["Signal: strained", "Signal: balanced", "Signal: strong"]
        assert len(res_data["context"]) > 10

def test_edge_case_numeric_limits():
    # Test boundary limits: min edge
    min_payload = {
        "age": 10,
        "gender": "Male",
        "country": "Other",
        "academic_level": "High School",
        "most_used_platform": "YouTube",
        "purpose_of_use": "Education",
        "avg_daily_usage_hours": 0.0,
        "daily_unlocks": 0,
        "study_hours": 0.0,
        "physical_activity_hours": 0.0,
        "sleep_hours_per_night": 0.0,
        "stress_level": "Low"
    }
    res_min = client.post("/predict", json=min_payload)
    assert res_min.status_code == 200
    assert 0.0 <= res_min.json()["score"] <= 10.0

    # Test boundary limits: max edge
    max_payload = {
        "age": 100,
        "gender": "Female",
        "country": "USA",
        "academic_level": "Graduate",
        "most_used_platform": "LinkedIn",
        "purpose_of_use": "Networking",
        "avg_daily_usage_hours": 24.0,
        "daily_unlocks": 500,
        "study_hours": 24.0,
        "physical_activity_hours": 24.0,
        "sleep_hours_per_night": 24.0,
        "stress_level": "Very High"
    }
    res_max = client.post("/predict", json=max_payload)
    assert res_max.status_code == 200
    assert 0.0 <= res_max.json()["score"] <= 10.0

def test_invalid_categorical_rejected():
    bad_payload = {
        "age": 21,
        "gender": "NonBinaryNotSupportedByModel",
        "country": "India",
        "academic_level": "PhD",  # Dataset only has High School, Undergraduate, Graduate
        "most_used_platform": "MySpace",
        "purpose_of_use": "Gaming",
        "avg_daily_usage_hours": 5.0,
        "daily_unlocks": 40,
        "study_hours": 4.0,
        "physical_activity_hours": 1.0,
        "sleep_hours_per_night": 7.0,
        "stress_level": "Extreme"
    }
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422
    data = res.json()
    assert data["success"] is False
    assert "field_errors" in data
    fields_with_errors = [e["field"] for e in data["field_errors"]]
    assert "gender" in fields_with_errors
    assert "academic_level" in fields_with_errors
    assert "most_used_platform" in fields_with_errors
    assert "purpose_of_use" in fields_with_errors
    assert "stress_level" in fields_with_errors
