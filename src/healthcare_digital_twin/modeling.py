from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .constants import FEATURES, TARGET
from .paths import ARTIFACTS_DIR, FIGURES_DIR, ensure_dirs


@dataclass(frozen=True)
class TrainResult:
    estimator: BaseEstimator
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    threshold: float | None
    y_true: list[int]
    y_prob: list[float] | None


def _validate_dataset(df: pd.DataFrame) -> None:
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_cols)
        )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    _validate_dataset(df)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def make_logreg_pipeline(*, max_iter: int = 5000) -> Pipeline:
    # Matches notebook logic: median impute + standardize + logistic regression
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=max_iter, solver="liblinear")),
        ]
    )


def train_logistic_regression(
    df: pd.DataFrame,
    *,
    tune: bool = True,
    random_state: int = 42,
    test_size: float = 0.2,
    threshold: float | None = None,
    penalty: Literal["l1", "l2"] | None = None,
) -> tuple[TrainResult, dict[str, Any]]:
    """Train Logistic Regression; optionally tune C and penalty using 5-fold CV (F1).

    The tuned grid matches `TunningParameter.ipynb`.
    """

    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = make_logreg_pipeline()
    metadata: dict[str, Any] = {
        "features": FEATURES,
        "target": TARGET,
        "random_state": random_state,
        "test_size": test_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if tune:
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__penalty": [penalty] if penalty is not None else ["l1", "l2"],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            # scikit-learn 1.8+ deprecates `penalty` but notebooks (and paper) tune it.
            warnings.filterwarnings(
                "ignore",
                message=r".*'penalty' was deprecated.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*penalty is deprecated.*",
                category=UserWarning,
            )
            search.fit(X_train, y_train)
        estimator: BaseEstimator = search.best_estimator_
        metadata["tuning"] = {
            "enabled": True,
            "best_params": search.best_params_,
            "best_cv_f1": float(search.best_score_),
        }
    else:
        if penalty is not None:
            pipeline.set_params(model__penalty=penalty)
        pipeline.fit(X_train, y_train)
        estimator = pipeline
        metadata["tuning"] = {"enabled": False}

    y_prob = None
    if hasattr(estimator, "predict_proba"):
        y_prob = estimator.predict_proba(X_test)[:, 1]

    if threshold is not None:
        if y_prob is None:
            raise ValueError("threshold was provided but estimator has no predict_proba")
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = estimator.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    if y_prob is not None:
        # Add threshold-independent ranking metrics.
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred)

    result = TrainResult(
        estimator=estimator,
        metrics=metrics,
        confusion_matrix=cm.tolist(),
        threshold=threshold,
        y_true=[int(v) for v in np.asarray(y_test).reshape(-1).tolist()],
        y_prob=None if y_prob is None else [float(v) for v in np.asarray(y_prob).reshape(-1).tolist()],
    )

    metadata["metrics"] = metrics
    metadata["threshold"] = threshold

    return result, metadata


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def save_artifacts(
    *,
    estimator: BaseEstimator,
    metadata: dict[str, Any],
    artifact_name: str = "logreg_pipeline",
) -> tuple[Path, Path]:
    ensure_dirs()

    model_path = ARTIFACTS_DIR / f"{artifact_name}.joblib"
    meta_path = ARTIFACTS_DIR / f"{artifact_name}.metadata.json"

    joblib.dump(estimator, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_path, meta_path


def save_confusion_matrix_plot(
    cm: list[list[int]] | np.ndarray,
    *,
    title: str,
    out_name: str,
) -> Path:
    """Lightweight plot helper to avoid hard dependency on seaborn in core code."""

    import matplotlib.pyplot as plt

    cm_arr = np.asarray(cm)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm_arr, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])

    thresh = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(
                j,
                i,
                format(int(cm_arr[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    ensure_dirs()
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def save_roc_curve_plot(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    *,
    title: str,
    out_name: str,
) -> Path:
    """Save an ROC curve PNG into reports/figures."""

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    fig = plt.figure(figsize=(4.5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    ensure_dirs()
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def save_precision_recall_curve_plot(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    *,
    title: str,
    out_name: str,
) -> Path:
    """Save a Precision–Recall curve PNG into reports/figures."""

    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    precision, recall, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    ap = float(average_precision_score(y_true_arr, y_prob_arr))

    fig = plt.figure(figsize=(4.5, 4))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()

    ensure_dirs()
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 50)

    rows: list[dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred)),
            }
        )

    return pd.DataFrame(rows)
