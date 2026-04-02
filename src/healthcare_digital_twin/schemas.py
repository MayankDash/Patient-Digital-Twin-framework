from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .constants import FEATURES


class PredictionRequest(BaseModel):
    Age: float = Field(..., ge=0, le=120)
    Gender: int = Field(..., ge=0, le=1, description="0=Male, 1=Female")
    BMI: float = Field(..., ge=0, le=100)
    Systolic_BP: float = Field(..., ge=0, le=300)
    Diastolic_BP: float = Field(..., ge=0, le=200)
    Glucose: float = Field(..., ge=0, le=500)
    Insulin: float = Field(..., ge=0, le=500)
    Total_Cholesterol: float = Field(..., ge=0, le=500)
    HDL_Cholesterol: float = Field(..., ge=0, le=200)

    @field_validator("Gender")
    @classmethod
    def _gender_is_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("Gender must be 0 (male) or 1 (female)")
        return v

    def to_frame(self) -> pd.DataFrame:
        data: dict[str, Any] = self.model_dump()
        return pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)


class PredictionResponse(BaseModel):
    risk_label: int
    risk_probability: float
    threshold: float


class ModelSummary(BaseModel):
    id: str
    artifact_name: str
    exists: bool
    threshold: float | None = None
    created_at: str | None = None
    features: list[str]
    tuning: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    confusion_matrix_figure: str | None = None


class DatasetOverview(BaseModel):
    n_rows: int
    n_columns: int
    n_features: int
    positive_rate: float | None = None
    positive_count: int | None = None
    female_rate: float | None = None
    feature_stats: dict[str, dict[str, float]]


class ModelInfoResponse(BaseModel):
    default_model: str
    models: list[ModelSummary]
    dataset: DatasetOverview


class WhatIfScenario(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    overrides: dict[str, float] = Field(default_factory=dict)

    @field_validator("overrides")
    @classmethod
    def _override_keys_are_features(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = [k for k in v.keys() if k not in FEATURES]
        if unknown:
            raise ValueError("Unknown feature(s) in overrides: " + ", ".join(unknown))
        return v


class AnalyzeRequest(BaseModel):
    base: PredictionRequest
    scenarios: list[WhatIfScenario] = Field(default_factory=list)


class Driver(BaseModel):
    feature: str
    contribution: float
    direction: str


class BaselineWithDrivers(PredictionResponse):
    top_drivers: list[Driver] = Field(default_factory=list)


class WhatIfScenarioResult(BaseModel):
    name: str
    overrides: dict[str, float]
    risk_probability: float
    risk_label: int
    delta_probability: float
    top_drivers: list[Driver] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    model: str
    artifact_name: str
    baseline: BaselineWithDrivers
    scenarios: list[WhatIfScenarioResult]
