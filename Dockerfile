# ---------- Stage 1: builder (install packages + train the model) ----------
FROM python:3.14.4-slim@sha256:2ca02f32b4d9d893863367ce07ec1972819f476dd38d8612f2a9cb6a41cbb727 AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt
COPY src/ src/
COPY churn_v2.csv .
RUN /opt/venv/bin/python -m src.train.train --out /build/models/model.joblib

# ---------- Stage 2: runtime (small, no build leftovers, non-root) ----------
FROM python:3.14.4-slim@sha256:2ca02f32b4d9d893863367ce07ec1972819f476dd38d8612f2a9cb6a41cbb727 AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/models ./models
COPY src/ src/
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 MODEL_PATH=/app/models/model.joblib
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
