from pathlib import Path

import joblib
import pandas as pd

# The saved model must sit in the same folder as this file.
ARTIFACT = joblib.load(Path(__file__).with_name("customer_segmenter.joblib"))


def predict_segment(age, orders_12m, total_spend_kes, days_since_last_order, returns, channel):
    if orders_12m < 1:
        raise ValueError("orders_12m must be at least 1")
    if channel not in ("M-Pesa", "Card", "Cash"):
        raise ValueError("channel must be M-Pesa, Card or Cash")

    row = pd.DataFrame([{"age": age, "orders_12m": orders_12m, "total_spend_kes": total_spend_kes,
                         "days_since_last_order": days_since_last_order, "returns": returns,
                         "channel": channel}])
    pipe = ARTIFACT["pipeline"]
    dist = float(pipe.transform(row).min())
    return {"segment": ARTIFACT["segment_names"][int(pipe.predict(row)[0])],
            "unusual": dist > ARTIFACT["unusual_distance"],
            "model_version": ARTIFACT["version"]}
