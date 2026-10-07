"""API tests. Run from the app folder with: python -m pytest"""

import pytest
from fastapi.testclient import TestClient

from api import app

GOOD_HOUSE = {
    "OverallQual": 7,
    "GrLivArea": 1710,
    "GarageCars": 2,
    "TotalBsmtSF": 856,
    "YearBuilt": 2003,
    "FullBath": 2,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_plausible_price(client):
    response = client.post("/predict", json=GOOD_HOUSE)
    assert response.status_code == 200
    body = response.json()
    assert 50_000 < body["predicted_price"] < 500_000
    assert body["warnings"] == []


def test_better_quality_is_not_cheaper(client):
    # A behavioural test: a sanity rule the model should respect.
    low = client.post("/predict", json={**GOOD_HOUSE, "OverallQual": 4}).json()
    high = client.post("/predict", json={**GOOD_HOUSE, "OverallQual": 9}).json()
    assert high["predicted_price"] >= low["predicted_price"]


@pytest.mark.parametrize(
    "field, bad_value",
    [("OverallQual", 11), ("GrLivArea", -5), ("GarageCars", 2.5), ("YearBuilt", "old")],
)
def test_invalid_inputs_are_rejected(client, field, bad_value):
    response = client.post("/predict", json={**GOOD_HOUSE, field: bad_value})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in GOOD_HOUSE.items() if k != "GarageCars"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_out_of_range_input_warns(client):
    body = client.post("/predict", json={**GOOD_HOUSE, "GrLivArea": 8000}).json()
    assert any("GrLivArea" in w for w in body["warnings"])


def test_batch(client):
    response = client.post("/predict/batch", json=[GOOD_HOUSE, GOOD_HOUSE])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_agent_endpoint(client):
    response = client.post("/agent/chat", json={"message": "What goes in a project submission?"})
    assert response.status_code == 200
    assert response.json()["agent_path"] == ["triage", "faq"]
