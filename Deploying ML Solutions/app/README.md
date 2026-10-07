# Practice ML service: house-price model + help-desk agent

A small, deployable service used in the deployment and agentic AI lessons. It
serves a scikit-learn house-price pipeline over a REST API, a Gradio UI, and a
help-desk agent that calls the model and an FAQ search as tools.

**For learning only.** See [MODEL_CARD.md](MODEL_CARD.md) for intended use and
limitations.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Trains the pipeline; writes `model/house_price_model.joblib` and `model/metadata.json` |
| `predictor.py` | Loads the model once; input contract (`HouseFeatures`), range warnings, `predict()` |
| `api.py` | FastAPI app: `/health`, `/model-info`, `/predict`, `/predict/batch`, `/agent/chat`, UI at `/ui` |
| `ui.py` | Gradio interface: price form and agent chat |
| `agent.py` | Agent loop, tools, handoffs, guardrails, model adapters, feedback logging |
| `tests/` | API, behavioural, and agent evaluation tests |
| `Dockerfile` | Container image; runs the tests during the build |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
python -m pytest -q
uvicorn api:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs (API) or http://127.0.0.1:8000/ui (UI).

## Run with Docker

```bash
docker build -t house-price-service .
docker run -p 7860:7860 house-price-service
```

To use a real LLM for the agent, pass the settings as environment variables:

```bash
docker run -p 7860:7860 -e AGENT_MODEL=groq/llama-3.3-70b-versatile -e GROQ_API_KEY=... house-price-service
```

## Configuring the agent's language model

Leave `AGENT_MODEL` unset to use the free, offline `ScriptedModel`. Otherwise
set it to any [LiteLLM model string](https://docs.litellm.ai/docs/providers)
and provide that provider's key as an environment variable or host secret:

| Provider | `AGENT_MODEL` | Key variable |
| --- | --- | --- |
| Ollama (local, free) | `ollama_chat/qwen2.5:7b` | none; run `ollama pull qwen2.5:7b` first |
| Groq | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| OpenRouter | `openrouter/meta-llama/llama-3.3-70b-instruct:free` | `OPENROUTER_API_KEY` |

Never commit keys. Tests always run offline. To run the agent evaluation
against your configured LLM, use `EVAL_WITH_LLM=1 python -m pytest tests/test_agent.py`.

## Deploying

The image runs on any container host. For a Hugging Face Space, choose the
Docker SDK, copy this folder into the Space repository, and add to the top of
its README:

```yaml
---
title: House Price Service
sdk: docker
app_port: 7860
---
```

For Render, Railway, Fly.io, Google Cloud Run, or Azure Container Apps, point
the service at this folder's `Dockerfile`. These hosts set `$PORT`, which the
start command respects.

## Retraining

1. Change data or features in `train.py`; bump `MODEL_VERSION`.
2. `python train.py && python -m pytest -q`
3. Update `MODEL_CARD.md` with the new evaluation numbers.
4. Commit the new `model/` files and redeploy.
