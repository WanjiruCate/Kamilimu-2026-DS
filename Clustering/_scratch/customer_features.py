import numpy as np

NUMERIC = ["age", "orders_12m", "days_since_last_order",
           "return_rate", "log_avg_order_value"]
CATEGORICAL = ["channel"]


def add_features(df):
    out = df.copy()
    out["avg_order_value"] = out["total_spend_kes"] / out["orders_12m"]   # spend per order
    out["log_avg_order_value"] = np.log1p(out["avg_order_value"])         # tame the skew
    out["return_rate"] = out["returns"] / out["orders_12m"]               # share of orders returned
    return out