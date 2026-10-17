"""The training script produces a model that is clearly better than a baseline, and safe."""

import joblib

import train_mushroom
from app.codes import FEATURES


def test_every_code_is_translated():
    df = train_mushroom.load_data()
    assert len(df) == 8124
    assert df[FEATURES].notna().all().all()


def test_mushroom_model(models_dir):
    bundle = joblib.load(models_dir / "mushroom.joblib")
    metrics = bundle["metrics"]
    assert metrics["roc_auc"] > 0.99
    assert metrics["accuracy"] > 0.97
    assert metrics["recall"] >= 0.99  # (almost) no poisonous mushroom called edible
    assert 0 < bundle["threshold"] < 1
    assert bundle["features"] == FEATURES
