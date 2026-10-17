from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from app.codes import CODES, DROPPED, FEATURES

HERE = Path(__file__).parent
DATA = HERE / "data" / "mushrooms.csv"
MODELS = HERE / "models"

TARGET = "poisonous"  # 1 = poisonous, 0 = edible


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def load_data(data_path: Path = DATA) -> pd.DataFrame:
    """The CSV with readable column names and words instead of one-letter codes."""
    raw = pd.read_csv(data_path)
    raw.columns = [column.replace("-", "_") for column in raw.columns]
    df = pd.DataFrame({TARGET: (raw["class"] == "p").astype(int)})
    for feature, words in CODES.items():
        df[feature] = raw[feature].map(words)
    unknown = df[FEATURES].isna().sum()
    if unknown.any():
        raise ValueError(f"letters missing from app/codes.py: {unknown[unknown > 0].to_dict()}")
    return df


def make_encoder() -> OneHotEncoder:
    # One column per (feature, word). A word never seen in training becomes all zeros.
    return OneHotEncoder(handle_unknown="ignore")


def train(data_path: Path = DATA, models_dir: Path = MODELS, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *args, **kwargs: None)
    show = step if verbose else (lambda title: None)

    show("1. Load the data and translate the codes")
    df = load_data(data_path)
    say(f"{len(df)} mushrooms, {len(FEATURES)} features (dropped {', '.join(DROPPED)}: one value)")
    say(f"poisonous: {df[TARGET].mean():.1%}  <- the two classes are almost balanced")
    say(f"stalk_root is 'missing' for {(df['stalk_root'] == 'missing').sum()} mushrooms")

    show("2. Split: train on 80%, keep 20% aside for the final test")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    say(f"train: {len(X_train)} rows, test: {len(X_test)} rows (same share of poisonous)")

    show("3. A baseline: always say 'edible'")
    baseline = np.zeros(len(y_test), dtype=int)
    say(f"accuracy {accuracy_score(y_test, baseline):.1%}, poisonous caught: 0 of {y_test.sum()}")
    say("-> half right, and it sends every poisonous mushroom to the kitchen")

    show("4. One feature at a time: how far does each go on its own?")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    one_feature = make_pipeline(make_encoder(), LogisticRegression(max_iter=1000))
    alone = {
        feature: cross_val_score(one_feature, X_train[[feature]], y_train, cv=folds).mean()
        for feature in FEATURES
    }
    for feature, accuracy in sorted(alone.items(), key=lambda x: -x[1])[:5]:
        say(f"{feature:<26} {accuracy:.1%} accuracy alone")
    say("-> one sniff (odor) does most of the work")

    show("5. Compare two models with 5-fold cross-validation (ROC AUC)")
    candidates = {
        "logistic regression": make_pipeline(make_encoder(), LogisticRegression(max_iter=1000)),
        "random forest": make_pipeline(
            make_encoder(),
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        ),
    }
    scores = {}
    for name, model in candidates.items():
        auc = cross_val_score(model, X_train, y_train, cv=folds, scoring="roc_auc")
        scores[name] = auc.mean()
        say(f"{name:<20} ROC AUC {auc.mean():.4f} (+/- {auc.std():.4f})")
    best_name = max(scores, key=scores.get)
    model = candidates[best_name]
    say(f"-> best: {best_name}")

    show("6. Choose the threshold: never call a poisonous mushroom edible")
    # Calling a poisonous mushroom edible is far worse than a false alarm. Among the thresholds
    # that flag every poisonous mushroom in cross-validation, keep the one with the fewest
    # false alarms, and on a tie the lowest (the most cautious).
    proba_cv = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    safe = [t for t in thresholds if recall_score(y_train, proba_cv >= t) == 1.0]
    alarms = {t: int(((proba_cv >= t) & (y_train == 0)).sum()) for t in safe}
    threshold = float(min(safe, key=lambda t: (alarms[t], t))) if safe else 0.01
    false_alarms = int(((proba_cv >= threshold) & (y_train == 0)).sum())
    say(f"threshold {threshold}: all poisonous flagged, {false_alarms} edible flagged too")

    show("7. Train on all training data, evaluate once on the test set")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "precision": round(precision_score(y_test, predicted), 4),
        "recall": round(recall_score(y_test, predicted), 4),
        "f1": round(f1_score(y_test, predicted), 4),
        "poisonous_missed": int(fn),
        "false_alarms": int(fp),
    }
    for name, value in metrics.items():
        say(f"{name:<17} {value}")
    say(f"poisonous caught: {tp} of {tp + fn}; edible flagged by mistake: {fp} of {fp + tn}")

    show("8. Without the nose: train again with no odor (cross-validation)")
    # Permutation importance says almost nothing here: hide one feature and the others make up
    # for it. Dropping the strongest feature outright shows that redundancy directly.
    no_odor = [feature for feature in FEATURES if feature != "odor"]
    accuracy = cross_val_score(clone(model), X_train[no_odor], y_train, cv=folds).mean()
    say(f"without odor: {accuracy:.2%} accuracy  <- spore print, gills and stalk take over")

    show("9. Save the model")
    bundle = {
        "model": model,
        "algorithm": best_name,
        "features": FEATURES,
        "threshold": threshold,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "mushroom.joblib"
    joblib.dump(bundle, path)
    say(f"saved to {path}")
    return bundle


if __name__ == "__main__":
    train()
