from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .inference import predict
from .inference import load_metadata
from .insights import analyze_what_if, dataset_overview, list_models, resolve_artifact_name
from .paths import FIGURES_DIR
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(title="Metabolic Risk Digital Twin API", version="0.1.0")

# Local dev CORS (React dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):(517\d+|3000)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        default_model="l2",
        models=list_models(),
        dataset=dataset_overview(),
    )


@app.get("/figures/{name}")
def figures(name: str):
    # Only serve files from reports/figures to support the UI dashboard.
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid figure name")
    if not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only .png figures are supported")

    path = FIGURES_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(path)


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(
    payload: PredictionRequest,
    model: str | None = Query(default=None, description="Model id: l2 (default) or l1"),
    artifact_name: str | None = Query(default=None, description="Override artifact name"),
    threshold: float | None = Query(default=None, description="Override decision threshold"),
) -> PredictionResponse:
    resolved_artifact = resolve_artifact_name(model=model, artifact_name=artifact_name)  # type: ignore[arg-type]
    try:
        return predict(payload, artifact_name=resolved_artifact, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(
    payload: AnalyzeRequest,
    model: str | None = Query(default=None, description="Model id: l2 (default) or l1"),
    artifact_name: str | None = Query(default=None, description="Override artifact name"),
    threshold: float | None = Query(default=None, description="Override decision threshold"),
) -> AnalyzeResponse:
    resolved_artifact = resolve_artifact_name(model=model, artifact_name=artifact_name)  # type: ignore[arg-type]
    meta = load_metadata(resolved_artifact)
    effective_threshold = threshold
    if effective_threshold is None:
        effective_threshold = meta.get("threshold")
    if effective_threshold is None:
        effective_threshold = 0.5

    base = payload.base.model_dump()
    scenarios = [s.model_dump() for s in payload.scenarios]

    try:
        result = analyze_what_if(
            artifact_name=resolved_artifact,
            threshold=float(effective_threshold),
            base=base,
            scenarios=scenarios,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return AnalyzeResponse(
        model=model or "l2",
        artifact_name=resolved_artifact,
        baseline=result["baseline"],
        scenarios=result["scenarios"],
    )
