# One image that runs anywhere containers run: your laptop, Hugging Face
# Spaces (Docker SDK), Render, Railway, Fly.io, Google Cloud Run, Azure
# Container Apps, AWS App Runner, or a university server.
FROM python:3.12-slim

# Run as a non-root user (also required by Hugging Face Spaces).
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser . .
USER appuser

# Many hosts tell the app which port to use through $PORT; 7860 is the
# Hugging Face Spaces default.
ENV PORT=7860
EXPOSE 7860

# Fail the build if the model is missing or the tests do not pass.
RUN test -f model/house_price_model.joblib && python -m pytest -q

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
