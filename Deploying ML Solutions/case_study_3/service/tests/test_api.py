def test_health_reports_the_full_extract(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["rows_in_feature_store"] >= 5_000_000


def test_example_window_is_classified(client):
    examples = client.get("/examples")
    assert examples.status_code == 200
    example = examples.json()[0]
    payload = {
        "site_id": example["site_id"],
        "room": example["room"],
        "available_power_sources": example["available_power_sources"],
        "history": example["history"],
        "site_median_frequency": example["site_median_frequency"],
        "n_sensors_in_bucket": example["n_sensors_in_bucket"],
        "model": "best",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["active_power_source"] in {"generator", "grid", "solar"}
    assert abs(sum(body["probabilities"].values()) - 1) < 0.02


def test_rules_lstm_and_rnn_endpoints(client):
    examples = client.get("/examples").json()
    example = examples[0]
    payload = {
        "site_id": example["site_id"],
        "room": example["room"],
        "available_power_sources": example["available_power_sources"],
        "history": example["history"],
        "site_median_frequency": example["site_median_frequency"],
        "n_sensors_in_bucket": example["n_sensors_in_bucket"],
    }
    for model in ("rules", "lstm", "rnn", "hgb"):
        payload["model"] = model
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["active_power_source"] in {"generator", "grid", "solar"}


def test_online_update_changes_weights_and_reset_restores_them(client):
    import numpy as np

    import predictor

    before = np.array(predictor.load_online()["sgd"].coef_, copy=True)
    examples = client.get("/examples").json()
    example = examples[0]
    feedback = {
        "site_id": example["site_id"],
        "room": example["room"],
        "available_power_sources": example["available_power_sources"],
        "history": example["history"],
        "site_median_frequency": example["site_median_frequency"],
        "n_sensors_in_bucket": example["n_sensors_in_bucket"],
        "label": "generator",
    }
    updated = client.post("/online/update", json=feedback)
    assert updated.status_code == 200
    after = np.array(predictor.load_online()["sgd"].coef_)
    try:
        assert float(np.max(np.abs(before - after))) > 0
    finally:
        reset = client.post("/online/reset")
        assert reset.status_code == 200
    restored = np.array(predictor.load_online()["sgd"].coef_)
    assert np.allclose(before, restored)


def test_browser_client_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Power source" in response.text
    assert "Why the boosting model" in response.text


def test_model_info_includes_the_shap_ranking(client):
    body = client.get("/model-info").json()
    ranking = body["shap"]["mean_abs_shap"]["generator"]
    assert next(iter(ranking)) == "abs_freq_dev_50"
    assert "has_grid" in ranking
