# AI381 Churn Prediction Service

![CI](https://github.com/RBT173/ai381-churn-service/actions/workflows/ci.yml/badge.svg?branch=main)

A churn prediction model served as a tested FastAPI service in a Docker container,
built, scanned with Trivy and pushed to GHCR by GitHub Actions on every merge to main.

## Run it (one command)

```bash
docker run -p 8000:8000 ghcr.io/rbt173/ai381-churn-service:4bb7c840d3f13df24b5a90727121826a3f5d9a5b
```

Then:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure_months": 5, "monthly_charges": 89.9, "contract": "month-to-month", "internet_service": "fiber", "support_calls": 3}'
```

## Develop locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train.train
python -m pytest -v
```

## Image digest

`ghcr.io/rbt173/ai381-churn-service@sha256:786baf665c51a4b5638f2408082e12568779656eecbb397f30f465fd14556368`
