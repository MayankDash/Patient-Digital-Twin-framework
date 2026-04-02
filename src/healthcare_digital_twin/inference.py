from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .paths import ARTIFACTS_DIR
from .schemas import PredictionRequest, PredictionResponse


DEFAULT_ARTIFACT_NAME = "logreg_pipeline"


def load_estimator(artifact_name: str = DEFAULT_ARTIFACT_NAME):
    model_path = ARTIFACTS_DIR / f"{artifact_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Train first: scripts/train.py"
        )
    return joblib.load(model_path)


def load_metadata(artifact_name: str = DEFAULT_ARTIFACT_NAME) -> dict[str, Any]:
    meta_path = ARTIFACTS_DIR / f"{artifact_name}.metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def predict(
    request: PredictionRequest,
    *,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
    threshold: float | None = None,
) -> PredictionResponse:
    estimator = load_estimator(artifact_name)
    metadata = load_metadata(artifact_name)

    effective_threshold = threshold
    if effective_threshold is None:
        effective_threshold = metadata.get("threshold")
    if effective_threshold is None:
        effective_threshold = 0.5

    X = request.to_frame()

    if not hasattr(estimator, "predict_proba"):
        # fall back to hard label prediction
        label = int(estimator.predict(X)[0])
        prob = float(label)
    else:
        prob = float(estimator.predict_proba(X)[:, 1][0])
        label = int(prob >= float(effective_threshold))

    return PredictionResponse(
        risk_label=label,
        risk_probability=prob,
        threshold=float(effective_threshold),
    )
