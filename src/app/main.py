"""Churn prediction API."""
import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
FEATURES = ["tenure_months", "monthly_charges", "contract", "internet_service", "support_calls"]


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: str = Field(min_length=1)
    support_calls: int = Field(ge=0)


app = FastAPI(title="AI381 churn API")
model = joblib.load(MODEL_PATH)
# which model is running: commit, MLflow run ID, data hash (written by scripts/promote.py)
META_PATH = Path(MODEL_PATH).parent / "metadata.json"
meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {"model_version": "local"}


@app.get("/health")
def health():
    return {"status": "ok", **meta}


@app.post("/predict")
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])[FEATURES]
    p = float(model.predict_proba(X)[0, 1])
    return {"churn_probability": round(p, 3), "model_version": meta.get("model_version")}
