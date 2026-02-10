FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + artifacts
COPY api.py .
COPY artifacts/ artifacts/

# Cloud Run provides $PORT
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}
