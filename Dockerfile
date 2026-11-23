FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLIGHT_DELAY_MODEL_PATH=/app/models/flight_delay_bundle.joblib

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY models/ ./models/
RUN python -m pip install --no-cache-dir .

USER app
EXPOSE 8000
CMD ["uvicorn", "flight_delay.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

