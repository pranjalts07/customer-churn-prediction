"""API endpoint tests"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.main import app


client = TestClient(app)


def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    """Test root endpoint returns HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_insights_global():
    """Test global insights endpoint"""
    response = client.get("/insights/global")
    assert response.status_code == 200
    data = response.json()
    assert "total_customers" in data
    assert "at_risk_count" in data
    assert "revenue_at_risk" in data
    assert data["model_auc_roc"] == 0.79


def test_predict():
    """Test prediction endpoint"""
    response = client.post(
        "/predict",
        json={
            "tenure": 12.0,
            "MonthlyCharges": 65.0,
            "TotalCharges": 780.0,
            "SeniorCitizen": 0,
            "Contract": 0,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_tier" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_tier"] in ["CRITICAL", "AT-RISK", "SAFE"]
