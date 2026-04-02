from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from .constants import FEATURES, TARGET
from .data import load_processed_patient_state
from .inference import load_estimator, load_metadata
from .paths import FIGURES_DIR


ModelId = Literal["l2", "l1"]


MODEL_ALIASES: dict[ModelId, str] = {
    "l2": "logreg_pipeline",
    "l1": "logreg_pipeline_l1",
}


def resolve_artifact_name(*, model: ModelId | None, artifact_name: str | None) -> str:
    if artifact_name:
        return artifact_name
    if model:
        return MODEL_ALIASES.get(model, MODEL_ALIASES["l2"])
    return MODEL_ALIASES["l2"]


def list_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model_id, artifact_name in MODEL_ALIASES.items():
        model_path = Path(".") / "artifacts" / f"{artifact_name}.joblib"
        meta = load_metadata(artifact_name)
        fig_name = f"{artifact_name}.confusion_matrix.png"
        fig_path = FIGURES_DIR / fig_name
        models.append(
            {
                "id": model_id,
                "artifact_name": artifact_name,
                "exists": model_path.exists(),
                "features": meta.get("features", FEATURES),
                "threshold": meta.get("threshold"),
                "tuning": meta.get("tuning"),
                "metrics": meta.get("metrics"),
                "created_at": meta.get("created_at"),
                "confusion_matrix_figure": fig_name if fig_path.exists() else None,
            }
        )
    return models


def dataset_overview() -> dict[str, Any]:
    df = load_processed_patient_state()

    overview: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_features": int(len(FEATURES)),
    }

    if TARGET in df.columns:
        overview["positive_rate"] = float(pd.to_numeric(df[TARGET], errors="coerce").mean())
        overview["positive_count"] = int(pd.to_numeric(df[TARGET], errors="coerce").sum())

    if "Gender" in df.columns:
        # Gender is encoded Male=0, Female=1
        overview["female_rate"] = float(pd.to_numeric(df["Gender"], errors="coerce").mean())

    # Feature summary (for dashboard cards / ranges)
    desc = df[FEATURES].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    feature_stats: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        feature_stats[feature] = {
            "min": float(desc[feature]["min"]),
            "p25": float(desc[feature]["25%"]),
            "median": float(desc[feature]["50%"]),
            "mean": float(desc[feature]["mean"]),
            "p75": float(desc[feature]["75%"]),
            "max": float(desc[feature]["max"]),
        }

    overview["feature_stats"] = feature_stats
    return overview


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _explain_logreg_pipeline(
    estimator: Any,
    X: pd.DataFrame,
) -> dict[str, Any] | None:
    """Return per-feature contributions for LogisticRegression inside our Pipeline.

    Contributions are computed in the standardized (post-scaler) space:
    contribution_i = z_i * coef_i

    This is meant for *explainability and what-if*, not clinical advice.
    """

    if not isinstance(estimator, Pipeline):
        return None

    try:
        imputer = estimator.named_steps["imputer"]
        scaler = estimator.named_steps["scaler"]
        model = estimator.named_steps["model"]
    except Exception:
        return None

    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        return None

    # Transform exactly like inference
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    coefs = np.asarray(model.coef_).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])

    # Single row expected
    z = np.asarray(X_scaled)[0]
    contrib = z * coefs
    logit = float(intercept + float(contrib.sum()))
    prob_from_logit = float(_sigmoid(logit))

    by_feature = {
        FEATURES[i]: float(contrib[i]) for i in range(min(len(FEATURES), contrib.shape[0]))
    }

    top = sorted(by_feature.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    top_drivers = [
        {
            "feature": f,
            "contribution": v,
            "direction": "increases" if v > 0 else "decreases",
        }
        for f, v in top
    ]

    return {
        "logit": logit,
        "probability_from_logit": prob_from_logit,
        "top_drivers": top_drivers,
        "contributions": by_feature,
    }


def analyze_what_if(
    *,
    artifact_name: str,
    threshold: float,
    base: dict[str, float],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run baseline + what-if scenarios, returning probabilities, deltas, and drivers."""

    estimator = load_estimator(artifact_name)

    rows: list[dict[str, Any]] = []
    rows.append({"name": "baseline", "inputs": base})

    for s in scenarios:
        name = str(s.get("name", "scenario"))
        overrides = dict(s.get("overrides") or {})
        inputs = {**base, **overrides}
        rows.append({"name": name, "inputs": inputs, "overrides": overrides})

    X = pd.DataFrame([r["inputs"] for r in rows], columns=FEATURES)

    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)[:, 1]
    else:
        # Fallback: hard predictions only
        probs = estimator.predict(X).astype(float)

    probs = np.asarray(probs, dtype=float)
    labels = (probs >= float(threshold)).astype(int)

    baseline_prob = float(probs[0])

    # Explain baseline + each scenario (if possible)
    explanations: list[dict[str, Any] | None] = []
    for i in range(X.shape[0]):
        explanations.append(_explain_logreg_pipeline(estimator, X.iloc[[i]]))

    out_scenarios: list[dict[str, Any]] = []
    for i, r in enumerate(rows[1:], start=1):
        delta = float(probs[i] - baseline_prob)
        exp = explanations[i]
        out_scenarios.append(
            {
                "name": r.get("name"),
                "overrides": r.get("overrides") or {},
                "risk_probability": float(probs[i]),
                "risk_label": int(labels[i]),
                "delta_probability": delta,
                "top_drivers": (exp or {}).get("top_drivers", []) if exp else [],
            }
        )

    base_exp = explanations[0]

    return {
        "baseline": {
            "risk_probability": baseline_prob,
            "risk_label": int(labels[0]),
            "threshold": float(threshold),
            "top_drivers": (base_exp or {}).get("top_drivers", []) if base_exp else [],
        },
        "scenarios": out_scenarios,
    }
