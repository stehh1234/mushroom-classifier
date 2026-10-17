"""Every endpoint: predictions, invalid input (422) and a missing model (503)."""

from fastapi.testclient import TestClient

from app import ml
from app.main import app
from app.schemas import EXAMPLE_MUSHROOM

# The second mushroom in the dataset: edible (smells of almond, brown spore print).
EDIBLE = {
    **EXAMPLE_MUSHROOM,
    "cap_color": "yellow",
    "odor": "almond",
    "gill_size": "broad",
    "stalk_root": "club",
    "spore_print_color": "brown",
    "population": "numerous",
    "habitat": "grasses",
}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": {"mushroom": True}}


def test_models(client):
    (info,) = client.get("/models").json()
    assert info["name"] == "mushroom"
    assert info["loaded"] is True
    assert info["metrics"]["roc_auc"] > 0.99


def test_poisonous_mushroom(client):
    response = client.post("/mushroom/predict", json=EXAMPLE_MUSHROOM)
    assert response.status_code == 200
    assert response.json()["verdict"] == "poisonous"


def test_edible_mushroom(client):
    body = client.post("/mushroom/predict", json=EDIBLE).json()
    assert body["verdict"] == "edible"
    assert body["poisonous_probability"] < body["threshold"]


def test_batch(client):
    response = client.post("/mushroom/predict/batch", json=[EXAMPLE_MUSHROOM, EDIBLE])
    assert [p["verdict"] for p in response.json()] == ["poisonous", "edible"]


def test_empty_batch(client):
    assert client.post("/mushroom/predict/batch", json=[]).status_code == 422


def test_unknown_word(client):
    response = client.post("/mushroom/predict", json={**EXAMPLE_MUSHROOM, "odor": "lemon"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "odor"]


def test_missing_and_extra_fields(client):
    incomplete = {k: v for k, v in EXAMPLE_MUSHROOM.items() if k != "odor"}
    assert client.post("/mushroom/predict", json=incomplete).status_code == 422
    extra = {**EXAMPLE_MUSHROOM, "veil_type": "partial"}
    assert client.post("/mushroom/predict", json=extra).status_code == 422


def test_model_not_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "MODELS_DIR", tmp_path)  # an empty folder: no model
    with TestClient(app) as client:
        response = client.post("/mushroom/predict", json=EXAMPLE_MUSHROOM)
    assert response.status_code == 503
    assert "train_mushroom.py" in response.json()["detail"]
