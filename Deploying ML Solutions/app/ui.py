"""Gradio interface: a form for the model and a chat window for the agent.

Run on its own (handy in Colab, where ``share=True`` gives a temporary
public link):

    python ui.py

In deployment, ``api.py`` mounts this same interface at ``/ui``.
"""

import gradio as gr

import agent
from predictor import HouseFeatures, predict


def estimate(qual, area, garage, basement, year, baths):
    try:
        features = HouseFeatures(
            OverallQual=qual, GrLivArea=area, GarageCars=garage, TotalBsmtSF=basement,
            YearBuilt=year, FullBath=baths,
        )
    except ValueError as exc:
        return f"Input problem: {exc}"
    result = predict(features)
    lines = [f"## ${result.predicted_price:,.0f}", f"Model version {result.model_version}"]
    lines += [f"⚠️ {w}" for w in result.warnings]
    lines.append("_Classroom estimate from historical Ames, Iowa data. Not a valuation._")
    return "\n\n".join(lines)


def chat(message, history):
    result = agent.run(message)
    route = " → ".join(result.agent_path)
    return f"{result.answer}\n\n<sub>route: {route} · trace {result.trace_id}</sub>"


def build_ui():
    with gr.Blocks(title="KamiLimu practice ML service") as demo:
        gr.Markdown("# KamiLimu practice ML service\nFor learning only.")
        with gr.Tab("Price estimator"):
            with gr.Row():
                with gr.Column():
                    inputs = [
                        gr.Slider(1, 10, value=7, step=1, label="Overall quality (1-10)"),
                        gr.Number(value=1710, label="Living area (sq ft)"),
                        gr.Slider(0, 4, value=2, step=1, label="Garage capacity (cars)"),
                        gr.Number(value=856, label="Basement area (sq ft)"),
                        gr.Number(value=2003, precision=0, label="Year built"),
                        gr.Slider(0, 4, value=2, step=1, label="Full bathrooms"),
                    ]
                    button = gr.Button("Estimate", variant="primary")
                output = gr.Markdown()
            button.click(estimate, inputs, output)
        with gr.Tab("Help-desk agent"):
            gr.ChatInterface(
                chat,
                examples=[
                    "What should I include in my project submission?",
                    "Estimate a house price: OverallQual=7 GrLivArea=1710 GarageCars=2 "
                    "TotalBsmtSF=856 YearBuilt=2003 FullBath=2",
                    "What medication should I take for a headache?",
                ],
            )
    return demo


if __name__ == "__main__":
    import sys

    build_ui().launch(share="google.colab" in sys.modules)
