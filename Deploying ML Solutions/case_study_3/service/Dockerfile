FROM python:3.12-slim

RUN useradd --create-home appuser
WORKDIR /app

COPY service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY pipeline /app/pipeline
COPY service /app/service
COPY artifacts /app/artifacts

ENV PYTHONPATH=/app
ENV ARTIFACT_DIR=/app/artifacts
ENV PORT=8010
EXPOSE 8010

RUN test -f /app/artifacts/model.joblib \
    && test -f /app/artifacts/online_model_init.joblib \
    && python -m pytest -q /app/service/tests \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app/service
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
