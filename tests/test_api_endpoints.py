from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from healthcare_digital_twin.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info_endpoint() -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()

    assert data["default_model"] in ("l2", "l1")
    assert "dataset" in data
    assert data["dataset"]["n_rows"] > 0
    assert "models" in data
    assert isinstance(data["models"], list)


def test_analyze_endpoint_smoke() -> None:
    # Skip if the default artifact does not exist.
    if not (Path("artifacts") / "logreg_pipeline.joblib").exists():
        pytest.skip("Model artifact missing; run scripts/train.py first")

    payload = {
        "base": {
            "Age": 50,
            "Gender": 1,
            "BMI": 31.5,
            "Systolic_BP": 135,
            "Diastolic_BP": 85,
            "Glucose": 115,
            "Insulin": 14,
            "Total_Cholesterol": 210,
            "HDL_Cholesterol": 45,
        },
        "scenarios": [
            {"name": "glucose+20", "overrides": {"Glucose": 135}},
        ],
    }

    r = client.post("/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert "baseline" in data
    assert 0.0 <= data["baseline"]["risk_probability"] <= 1.0

    assert "scenarios" in data
    assert len(data["scenarios"]) == 1

    s0 = data["scenarios"][0]
    assert s0["name"] == "glucose+20"
    assert "delta_probability" in s0


def test_analyze_multiple_scenarios_length_matches_request() -> None:
    # Skip if the default artifact does not exist.
    if not (Path("artifacts") / "logreg_pipeline.joblib").exists():
        pytest.skip("Model artifact missing; run scripts/train.py first")

    payload = {
        "base": {
            "Age": 45,
            "Gender": 1,
            "BMI": 28.2,
            "Systolic_BP": 130,
            "Diastolic_BP": 82,
            "Glucose": 110,
            "Insulin": 12,
            "Total_Cholesterol": 190,
            "HDL_Cholesterol": 48,
        },
        "scenarios": [
            {"name": "s1", "overrides": {"Glucose": 120}},
            {"name": "s2", "overrides": {"BMI": 33}},
            {"name": "s3", "overrides": {"Systolic_BP": 150}},
        ],
    }

    r = client.post("/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["scenarios"]) == len(payload["scenarios"])


def test_predict_model_switching_l1_if_available() -> None:
    if not (Path("artifacts") / "logreg_pipeline.joblib").exists():
        pytest.skip("Model artifact missing; run scripts/train.py first")
    if not (Path("artifacts") / "logreg_pipeline_l1.joblib").exists():
        pytest.skip("L1 artifact missing; run scripts/train.py --penalty l1 first")

    payload = {
        "Age": 45,
        "Gender": 1,
        "BMI": 28.2,
        "Systolic_BP": 130,
        "Diastolic_BP": 82,
        "Glucose": 110,
        "Insulin": 12,
        "Total_Cholesterol": 190,
        "HDL_Cholesterol": 48,
    }

    r_l2 = client.post("/predict?model=l2", json=payload)
    assert r_l2.status_code == 200
    d_l2 = r_l2.json()
    assert 0.0 <= d_l2["risk_probability"] <= 1.0

    r_l1 = client.post("/predict?model=l1", json=payload)
    assert r_l1.status_code == 200
    d_l1 = r_l1.json()
    assert 0.0 <= d_l1["risk_probability"] <= 1.0


def test_figures_rejects_path_traversal_and_non_png() -> None:
    bad_names = ["../x.png", "a/b.png", "a\\b.png", "x.txt", "x.jpg"]
    for name in bad_names:
        r = client.get(f"/figures/{name}")
        assert r.status_code in (400, 404)


def test_cors_preflight_allows_local_vite_origin() -> None:
    # This is what the browser does before a cross-origin fetch.
    r = client.options(
        "/model-info",
        headers={
            "Origin": "http://localhost:5174",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5174"


def test_figures_endpoint_smoke() -> None:
    # If a known confusion matrix exists, ensure it is served.
    fig = Path("reports") / "figures" / "logreg_pipeline.confusion_matrix.png"
    if not fig.exists():
        pytest.skip("No confusion matrix figure found")

    r = client.get(f"/figures/{fig.name}")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("image/")
