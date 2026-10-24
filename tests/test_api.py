"""Tests for the churn API. Run `python -m src.train.train` first so the model exists."""
import joblib
from fastapi.testclient import TestClient

from src.app.main import MODEL_PATH, app

client = TestClient(app)

VALID = {
    "tenure_months": 5,
    "monthly_charges": 89.9,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


def test_health_contract():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_contract():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    p = r.json()["churn_probability"]
    assert 0.0 <= p <= 1.0


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba")


def test_input_schema_rejects_bad_input():
    bad = {**VALID, "contract": "weekly"}
    assert client.post("/predict", json=bad).status_code == 422
    missing = {k: v for k, v in VALID.items() if k != "tenure_months"}
    assert client.post("/predict", json=missing).status_code == 422


def test_edge_case_brand_new_customer():
    new = {**VALID, "tenure_months": 0, "support_calls": 0}
    r = client.post("/predict", json=new)
    assert r.status_code == 200


def test_prediction_is_deterministic():
    a = client.post("/predict", json=VALID).json()
    b = client.post("/predict", json=VALID).json()
    assert a == b
