from __future__ import annotations

import pandas as pd

from healthcare_digital_twin.modeling import train_logistic_regression
from healthcare_digital_twin.schemas import PredictionRequest


def test_train_and_predict_smoke() -> None:
    # Minimal synthetic dataset with required columns
    rows = [
        # Class 0
        {
            "Age": 40,
            "Gender": 0,
            "BMI": 25.0,
            "Systolic_BP": 120,
            "Diastolic_BP": 80,
            "Glucose": 90,
            "Insulin": 10,
            "Total_Cholesterol": 180,
            "HDL_Cholesterol": 55,
            "Metabolic_Risk": 0,
        },
        {
            "Age": 38,
            "Gender": 1,
            "BMI": 23.0,
            "Systolic_BP": 115,
            "Diastolic_BP": 76,
            "Glucose": 88,
            "Insulin": 9,
            "Total_Cholesterol": 170,
            "HDL_Cholesterol": 58,
            "Metabolic_Risk": 0,
        },
        {
            "Age": 35,
            "Gender": 1,
            "BMI": 22.0,
            "Systolic_BP": 110,
            "Diastolic_BP": 72,
            "Glucose": 85,
            "Insulin": 8,
            "Total_Cholesterol": 165,
            "HDL_Cholesterol": 60,
            "Metabolic_Risk": 0,
        },
        {
            "Age": 44,
            "Gender": 0,
            "BMI": 26.0,
            "Systolic_BP": 122,
            "Diastolic_BP": 81,
            "Glucose": 92,
            "Insulin": 11,
            "Total_Cholesterol": 185,
            "HDL_Cholesterol": 52,
            "Metabolic_Risk": 0,
        },
        {
            "Age": 29,
            "Gender": 0,
            "BMI": 21.5,
            "Systolic_BP": 108,
            "Diastolic_BP": 70,
            "Glucose": 82,
            "Insulin": 7,
            "Total_Cholesterol": 160,
            "HDL_Cholesterol": 62,
            "Metabolic_Risk": 0,
        },
        # Class 1
        {
            "Age": 62,
            "Gender": 1,
            "BMI": 33.0,
            "Systolic_BP": 145,
            "Diastolic_BP": 92,
            "Glucose": 140,
            "Insulin": 22,
            "Total_Cholesterol": 220,
            "HDL_Cholesterol": 40,
            "Metabolic_Risk": 1,
        },
        {
            "Age": 55,
            "Gender": 0,
            "BMI": 29.0,
            "Systolic_BP": 135,
            "Diastolic_BP": 88,
            "Glucose": 125,
            "Insulin": 16,
            "Total_Cholesterol": 205,
            "HDL_Cholesterol": 45,
            "Metabolic_Risk": 1,
        },
        {
            "Age": 58,
            "Gender": 1,
            "BMI": 31.0,
            "Systolic_BP": 142,
            "Diastolic_BP": 90,
            "Glucose": 132,
            "Insulin": 18,
            "Total_Cholesterol": 210,
            "HDL_Cholesterol": 42,
            "Metabolic_Risk": 1,
        },
        {
            "Age": 50,
            "Gender": 0,
            "BMI": 30.0,
            "Systolic_BP": 138,
            "Diastolic_BP": 86,
            "Glucose": 120,
            "Insulin": 15,
            "Total_Cholesterol": 200,
            "HDL_Cholesterol": 44,
            "Metabolic_Risk": 1,
        },
        {
            "Age": 66,
            "Gender": 1,
            "BMI": 34.0,
            "Systolic_BP": 150,
            "Diastolic_BP": 95,
            "Glucose": 150,
            "Insulin": 24,
            "Total_Cholesterol": 230,
            "HDL_Cholesterol": 38,
            "Metabolic_Risk": 1,
        },
    ]

    df = pd.DataFrame(rows)

    result, _ = train_logistic_regression(df, tune=False)
    assert "f1" in result.metrics

    # schema validation should accept valid payload
    _ = PredictionRequest(
        Age=45,
        Gender=1,
        BMI=28.2,
        Systolic_BP=130,
        Diastolic_BP=82,
        Glucose=110,
        Insulin=12,
        Total_Cholesterol=190,
        HDL_Cholesterol=48,
    )
