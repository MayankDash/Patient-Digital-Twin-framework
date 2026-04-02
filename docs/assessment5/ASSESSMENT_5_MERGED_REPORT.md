# Assessment 5 — Documentation + Test Report + Validation Report (Merged)

**Project title:** Metabolic Risk Digital Twin (NHANES 2017–2018)

**Submission items (rubric):**

1. Complete demo
2. Documentation along with test report / validation report
3. Poster (A1 size) and demo video

This merged report is written to be **professional**, **easy to verify**, and **aligned with the actual saved artifacts** in `artifacts/`.

---

## Table of contents

1. Executive summary
2. Repository overview

- 2.1 System architecture (diagram)

3. Dataset and label definition
4. Features (Digital Twin state)
5. Modeling approach
6. Hyperparameter tuning and evaluation
7. Testing report (rigorous verification)
8. Validation & verification report
9. Demo steps + evidence checklist

- 9.3 Evidence appendix (paste-ready)

10. Limitations

---

## 1) Executive summary

This project builds a **Patient Digital Twin** for metabolic risk screening using **NHANES 2017–2018**. Each patient is represented as a clinical feature vector (age, BMI, BP, glucose, insulin, cholesterol, HDL, gender). A supervised ML model outputs:

- **risk probability** in [0, 1]
- **risk label** derived from a decision threshold

The project is designed to be **reproducible and demo-ready**: notebook research is packaged into scripts, saved artifacts, an API, and a React UI.

**Constraint respected:** existing research notebooks were not modified.

---

## 2) Repository overview (where to find things)

- `data/raw/` — NHANES XPT files (source)
- `data/processed/` — clean patient dataset used for modeling
- `src/healthcare_digital_twin/` — reusable Python package (modeling + API)
- `scripts/` — CLI entry points (train/predict)
- `artifacts/` — exported model pipeline(s) + metadata
- `reports/figures/` — confusion matrix images
- `frontend/` — React UI (Vite)

### 2.1 System architecture (diagram)

```mermaid
flowchart LR
  A[Raw NHANES XPT files\n(data/raw)] --> B[Build / Merge / Clean\n(patient_state_clean.csv)]
  B --> C[Training + Tuning\n(scripts/train.py)]
  C --> D[Exported Artifacts\n(artifacts/*.joblib + *.metadata.json)]
  D --> E[FastAPI Inference Service\nGET /health\nPOST /predict]
  E --> F[React UI (Vite)\n(frontend)]
  D --> G[Offline CLI Inference\n(load joblib + predict_proba)]
  C --> H[Reports\nconfusion matrix\n(reports/figures)]
```

---

## 3) Dataset and label definition

**Dataset:** NHANES 2017–2018 cycle.

**Integration approach:** multiple NHANES tables are merged using participant identifier `SEQN`, creating one patient-level record.

**Final dataset for modeling:** `data/processed/patient_state_clean.csv`.

**Target (label):** `Metabolic_Risk` is derived from HbA1c thresholding used in the research notebooks.

- HbA1c is used only to create the label
- HbA1c is excluded from model inputs to prevent leakage

---

## 4) Features (Digital Twin state)

The deployed Digital Twin state is the 9-feature vector below (confirmed in artifact metadata):

- Age
- Gender (encoded Male=0, Female=1)
- BMI
- Systolic_BP
- Diastolic_BP
- Glucose
- Insulin
- Total_Cholesterol
- HDL_Cholesterol

---

## 5) Modeling approach

### 5.1 Models explored in research notebooks

The exploratory notebooks compare multiple models:

- Logistic Regression
- Random Forest
- Support Vector Machine (RBF)
- Gradient Boosting

### 5.2 Model used for deployment/demo

The production pipeline uses **Logistic Regression** because it is:

- interpretable (review-friendly)
- stable and fast
- produces calibrated-style probabilities usable with explicit thresholds

### 5.3 Preprocessing pipeline

A single scikit-learn Pipeline is used for both training and inference:

1. Median imputation
2. Standard scaling
3. Logistic Regression classifier

This prevents train/serve mismatch and reduces leakage risk.

---

## 6) Hyperparameter tuning and evaluation (artifact-based ground truth)

All results below are taken directly from:

- `artifacts/logreg_pipeline.metadata.json`
- `artifacts/logreg_pipeline_default.metadata.json`

### 6.1 Train/test strategy

- Stratified train/test split
- `test_size = 0.2`
- `random_state = 42`

### 6.2 Cross-validation (tuning)

- Stratified 5-fold cross-validation
- Optimized metric: F1

### 6.3 Hyperparameter search

Grid search over:

- `C ∈ {0.01, 0.1, 1, 10, 100}`
- `penalty ∈ {l1, l2}`

Best configuration (both runs):

- `model__C = 0.1`
- `model__penalty = l2`
- Best CV F1: **0.7369406318**

### 6.4 Test-set metrics (two decision policies)

| Decision policy       | Threshold | Accuracy | Precision |  Recall |      F1 |
| --------------------- | --------: | -------: | --------: | ------: | ------: |
| Default (`predict()`) |      null |  0.75517 |   0.71296 | 0.65812 | 0.68444 |
| Screening-friendly    |       0.4 |  0.75517 |   0.68110 | 0.73932 | 0.70902 |

Interpretation: threshold=0.4 increases recall and improves F1 in this test run, which is appropriate for screening use-cases where missing an at-risk patient is costly.

---

## 7) Testing report (rigorous verification)

This section is written to be **easy for a reviewer to reproduce**.

### 7.1 Automated tests (unit/smoke)

**Command executed (verified):**

```bash
.venv/bin/python -m pytest -q
```

**Outcome:** `1 passed` (local run).

What it covers:

- ML pipeline smoke behavior
- basic data split reliability

### 7.2 Artifact load + inference test (no API required)

This verifies that the exported model can be reloaded and used for inference (key for reproducibility).

**Command executed (verified):**

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import joblib
import pandas as pd

root = Path('.')
model_path = root / 'artifacts' / 'logreg_pipeline.joblib'
meta_path = root / 'artifacts' / 'logreg_pipeline.metadata.json'

pipe = joblib.load(model_path)
meta = json.loads(meta_path.read_text())

sample = {
    'Age': 50,
    'Gender': 1,
    'BMI': 31.5,
    'Systolic_BP': 135,
    'Diastolic_BP': 85,
    'Glucose': 115,
    'Insulin': 14,
    'Total_Cholesterol': 210,
    'HDL_Cholesterol': 45,
}

X = pd.DataFrame([sample], columns=meta['features'])
proba = float(pipe.predict_proba(X)[:, 1][0])
print('Loaded:', model_path.name)
print('Features:', len(meta['features']))
print('Threshold:', meta['threshold'])
print('Sample risk_probability:', round(proba, 6))
print('Sample risk_label:', int(proba >= (meta['threshold'] or 0.5)))
PY
```

**Outcome (verified):**

- Loaded: `logreg_pipeline.joblib`
- Features: `9`
- Threshold: `0.4`
- Sample risk_probability: `0.573235`
- Sample risk_label: `1`

### 7.3 API integration checks (recommended demo verification)

Start backend:

```bash
.venv/bin/python -m uvicorn healthcare_digital_twin.api:app --reload --port 8000
```

Health:

```bash
curl -s http://127.0.0.1:8000/health
```

Predict:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 50,
    "Gender": 1,
    "BMI": 31.5,
    "Systolic_BP": 135,
    "Diastolic_BP": 85,
    "Glucose": 115,
    "Insulin": 14,
    "Total_Cholesterol": 210,
    "HDL_Cholesterol": 45
  }'
```

CORS preflight (frontend compatibility):

```bash
curl -i -X OPTIONS http://127.0.0.1:8000/predict \
  -H "Origin: http://127.0.0.1:5173" \
  -H "Access-Control-Request-Method: POST"
```

### 7.4 Frontend user-journey checks (manual acceptance)

```bash
cd frontend
npm install
npm run dev
```

Acceptance criteria:

- UI loads without console errors
- form submit produces a prediction result
- “backend offline” scenario is handled clearly

---

## 8) Validation & verification report

### 8.1 Functional validation

Verified/expected system behaviors:

- data can be processed into a patient-level dataset
- model trains end-to-end via `scripts/train.py`
- artifacts export and can be reloaded for inference
- API returns probability + label
- UI can call the API with CORS

### 8.2 Scientific/ML validation

- stratified holdout split separates train/test
- 5-fold stratified CV used for tuning
- test metrics reported on held-out data
- explicit threshold policy evaluated (default vs 0.4)

### 8.3 Leakage prevention

- HbA1c is used only to create the label
- HbA1c excluded from input features
- preprocessing inside Pipeline prevents test-to-train leakage

---

## 9) Demo steps + evidence checklist

### 9.1 Demo steps

1. Start API and show `/health`
2. Open UI and submit one patient record
3. Explain probability vs label and threshold policy
4. Show artifacts metadata + confusion matrix image(s)

### 9.2 Evidence to attach for submission

- screenshot: `pytest` passing
- screenshot: UI prediction result
- screenshot: `artifacts/` showing metadata JSON
- confusion matrix image(s) under `reports/figures/`

### 9.3 Evidence appendix (paste-ready)

Use this section to paste screenshots in your final PDF submission.

**A. Automated tests (pytest)**

- Paste screenshot: terminal output showing `1 passed`
- Command shown in screenshot: `.venv/bin/python -m pytest -q`

**B. Artifact reproducibility (model reload + prediction)**

- Paste screenshot: terminal output showing:
  - `Loaded: logreg_pipeline.joblib`
  - `Features: 9`
  - `Threshold: 0.4`
  - `Sample risk_probability: ...`

**C. API readiness (`/health`)**

- Paste screenshot: `curl http://127.0.0.1:8000/health` output

**D. API inference (`/predict`)**

- Paste screenshot: `curl http://127.0.0.1:8000/predict` output with probability + label

**E. UI prediction**

- Paste screenshot: UI form filled + prediction result visible

**F. Confusion matrix figure(s)**

- Paste image(s): from `reports/figures/`

---

## 10) Limitations (honest academic scope)

- NHANES is cross-sectional: output is risk estimation, not diagnosis
- generalization requires external validation on local clinical populations
- threshold should be selected by domain requirements (false negative vs false positive costs)
