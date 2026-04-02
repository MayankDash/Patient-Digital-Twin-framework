from __future__ import annotations

import argparse

from healthcare_digital_twin.data import load_processed_patient_state
from healthcare_digital_twin.modeling import (
    save_artifacts,
    save_confusion_matrix_plot,
    train_logistic_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export models")
    parser.add_argument(
        "--model",
        choices=["logreg"],
        default="logreg",
        help="Model to train",
    )
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV tuning")
    parser.add_argument(
        "--penalty",
        choices=["any", "l1", "l2"],
        default="any",
        help="If set, restrict Logistic Regression penalty during training/tuning.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (uses predict_proba). If omitted, uses model.predict().",
    )
    parser.add_argument(
        "--artifact-name",
        default="logreg_pipeline",
        help="Base name for output artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_processed_patient_state()

    if args.model == "logreg":
        penalty = None if args.penalty == "any" else args.penalty
        result, metadata = train_logistic_regression(
            df,
            tune=args.tune,
            threshold=args.threshold,
            penalty=penalty,
        )

        model_path, meta_path = save_artifacts(
            estimator=result.estimator,
            metadata=metadata,
            artifact_name=args.artifact_name,
        )
        fig_path = save_confusion_matrix_plot(
            result.confusion_matrix,
            title="Confusion Matrix - Logistic Regression",
            out_name=f"{args.artifact_name}.confusion_matrix.png",
        )

        print("Training complete")
        print("Metrics:", result.metrics)
        print("Model:", model_path)
        print("Metadata:", meta_path)
        print("Figure:", fig_path)


if __name__ == "__main__":
    main()
