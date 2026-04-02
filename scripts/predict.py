from __future__ import annotations

import argparse
import json

from healthcare_digital_twin.schemas import PredictionRequest
from healthcare_digital_twin.inference import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single prediction")
    parser.add_argument(
        "--json",
        required=True,
        help="JSON payload with feature values",
    )
    parser.add_argument(
        "--artifact-name",
        default="logreg_pipeline",
        help="Artifact base name in artifacts/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.json)
    req = PredictionRequest(**payload)
    resp = predict(req, artifact_name=args.artifact_name, threshold=args.threshold)
    print(resp.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
