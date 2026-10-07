"""REST API for the house-price model and the help-desk agent.

Run locally from this folder:

    uvicorn api:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive documentation, or
http://127.0.0.1:8000/ui for the Gradio interface.
"""

from contextlib import asynccontextmanager
from typing import Annotated

import gradio as gr
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

import agent
from predictor import HouseFeatures, Prediction, load_metadata, load_model, predict
from ui import build_ui


@asynccontextmanager
async def lifespan(app):
    # Load the model at start-up so the first request is not slow and a
    # missing model file fails the deployment immediately.
    load_model()
    yield


app = FastAPI(
    title="KamiLimu practice ML service",
    description="Classroom house-price model and help-desk agent. Not for real decisions.",
    version=load_metadata()["model_version"],
    lifespan=lifespan,
)


@app.get("/health")
def health():
    meta = load_metadata()
    return {"status": "ok", "model_version": meta["model_version"], "trained_at": meta["trained_at"]}


@app.get("/model-info")
def model_info():
    return load_metadata()


@app.post("/predict", response_model=Prediction)
def predict_price(features: HouseFeatures):
    return predict(features)


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(houses: Annotated[list[HouseFeatures], Body(max_length=100)]):
    return [predict(h) for h in houses]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    trace_id: str
    agent_path: list[str]
    escalated: bool


@app.post("/agent/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent.run(request.message)
    return ChatResponse(
        answer=result.answer,
        trace_id=result.trace_id,
        agent_path=result.agent_path,
        escalated=result.escalated,
    )


app = gr.mount_gradio_app(app, build_ui(), path="/ui")
