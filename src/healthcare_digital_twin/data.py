from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.impute import SimpleImputer

from .constants import FEATURES, GENDER_MAPPING, RAW_TARGET, TARGET
from .paths import PROCESSED_DIR, RAW_DIR


DEFAULT_PROCESSED_CSV: Final[Path] = PROCESSED_DIR / "patient_state_clean.csv"


def load_processed_patient_state(csv_path: Path = DEFAULT_PROCESSED_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {csv_path}. "
            "Run scripts/build_dataset.py to build it from raw NHANES XPT files."
        )
    return pd.read_csv(csv_path)


def build_patient_state_from_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Rebuild the processed patient-state table, matching `01_data_loading.ipynb`.

    Produces columns:
    - FEATURES
    - HbA1c
    - Metabolic_Risk (HbA1c >= 5.7)
    """

    required = {
        "DEMO_J.xpt",
        "BMX_J.xpt",
        "BPX_J.xpt",
        "GLU_J.xpt",
        "GHB_J.xpt",
        "INS_J.xpt",
        "TCHOL_J.xpt",
        "HDL_J.xpt",
    }
    missing = [name for name in sorted(required) if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing NHANES raw files in "
            f"{raw_dir}: {', '.join(missing)}"
        )

    demo = pd.read_sas(raw_dir / "DEMO_J.xpt")
    bmx = pd.read_sas(raw_dir / "BMX_J.xpt")
    bpx = pd.read_sas(raw_dir / "BPX_J.xpt")
    glu = pd.read_sas(raw_dir / "GLU_J.xpt")
    ghb = pd.read_sas(raw_dir / "GHB_J.xpt")
    ins = pd.read_sas(raw_dir / "INS_J.xpt")
    tchol = pd.read_sas(raw_dir / "TCHOL_J.xpt")
    hdl = pd.read_sas(raw_dir / "HDL_J.xpt")

    demo_sel = (
        demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]
        .copy()
        .rename(columns={"RIDAGEYR": "Age", "RIAGENDR": "Gender"})
    )
    bmx_sel = bmx[["SEQN", "BMXBMI"]].copy().rename(columns={"BMXBMI": "BMI"})

    bpx_temp = bpx[[
        "SEQN",
        "BPXSY1", "BPXSY2", "BPXSY3",
        "BPXDI1", "BPXDI2", "BPXDI3",
    ]].copy()
    bpx_temp["Systolic_BP"] = bpx_temp[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
    bpx_temp["Diastolic_BP"] = bpx_temp[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)
    bpx_sel = bpx_temp[["SEQN", "Systolic_BP", "Diastolic_BP"]]

    glu_sel = glu[["SEQN", "LBXGLU"]].copy().rename(columns={"LBXGLU": "Glucose"})
    ghb_sel = ghb[["SEQN", "LBXGH"]].copy().rename(columns={"LBXGH": RAW_TARGET})
    ins_sel = ins[["SEQN", "LBXIN"]].copy().rename(columns={"LBXIN": "Insulin"})

    tchol_sel = tchol[["SEQN", "LBXTC"]].copy().rename(columns={"LBXTC": "Total_Cholesterol"})
    hdl_sel = hdl[["SEQN", "LBDHDD"]].copy().rename(columns={"LBDHDD": "HDL_Cholesterol"})

    dfs = [demo_sel, bmx_sel, bpx_sel, glu_sel, ghb_sel, ins_sel, tchol_sel, hdl_sel]
    patient_state = reduce(lambda left, right: pd.merge(left, right, on="SEQN", how="inner"), dfs)

    # Encode Gender (match notebook)
    patient_state["Gender"] = patient_state["Gender"].map(GENDER_MAPPING)

    # Median imputation on FEATURES only (match notebook)
    imputer = SimpleImputer(strategy="median")
    patient_state[FEATURES] = imputer.fit_transform(patient_state[FEATURES])

    # Drop missing HbA1c before labeling (match notebook)
    patient_state = patient_state.dropna(subset=[RAW_TARGET]).copy()
    patient_state[TARGET] = (patient_state[RAW_TARGET] >= 5.7).astype(int)

    return patient_state


def save_processed_patient_state(df: pd.DataFrame, csv_path: Path = DEFAULT_PROCESSED_CSV) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path
