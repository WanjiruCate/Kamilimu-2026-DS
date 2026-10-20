import gradio as gr

from predictor import predict_segment

ACTIONS = {
    "Loyal big spenders": "Early access and VIP rewards.",
    "Bargain hunters": "Flash-sale and discount-code SMS.",
    "Occasional shoppers": "Nudge with a reminder and a small incentive.",
    "Lapsed": "Win-back offer: 'We miss you'.",
}


def app_fn(age, orders, spend, days_since, returns, channel):
    try:
        r = predict_segment(age, orders, spend, days_since, returns, channel)
    except ValueError as e:
        return f"Input problem: {e}"
    msg = f"Segment: {r['segment']}\nSuggested action: {ACTIONS[r['segment']]}"
    return msg + ("\nNote: unusual customer, treat with care." if r["unusual"] else "")


demo = gr.Interface(
    fn=app_fn,
    inputs=[gr.Slider(18, 70, value=30, step=1, label="Age"),
            gr.Number(value=10, label="Orders in last 12 months"),
            gr.Number(value=20000, label="Total spend (KES)"),
            gr.Slider(1, 365, value=30, step=1, label="Days since last order"),
            gr.Number(value=1, label="Returns"),
            gr.Dropdown(["M-Pesa", "Card", "Cash"], value="M-Pesa", label="Payment channel")],
    outputs=gr.Textbox(label="Result"),
    title="Customer Segmenter",
    **({"flagging_mode": "never"} if int(gr.__version__.split(".")[0]) >= 5 else {"allow_flagging": "never"}),
)

if __name__ == "__main__":
    demo.launch()
