# HealthCareDigitalTwinn

Professionalized project version of your research notebooks for **Metabolic Risk Prediction** using **NHANES 2017–2018**.

Important: Existing notebooks in `notebooks/` are **not modified** (they back your paper). New reproducible notebooks are added alongside them.

## Quickstart

### 1) Install dependencies

```bash
".venv/bin/python" -m pip install -r requirements.txt
".venv/bin/python" -m pip install -e .
```

### 2) (Optional) Rebuild processed dataset from raw XPT

```bash
".venv/bin/python" scripts/build_dataset.py
```

### 3) Train + export the Logistic Regression pipeline

```bash
".venv/bin/python" scripts/train.py --model logreg --tune --threshold 0.4
```

To export an explicit **L1** regularized artifact alongside the default **L2** one:

```bash
".venv/bin/python" scripts/train.py --model logreg --tune --penalty l1 --threshold 0.4 --artifact-name logreg_pipeline_l1
```

Artifacts are written to `artifacts/`.

### 4) Run inference (CLI)

```bash
".venv/bin/python" scripts/predict.py --json '{"Age":45,"Gender":1,"BMI":28.2,"Systolic_BP":130,"Diastolic_BP":82,"Glucose":110,"Insulin":12,"Total_Cholesterol":190,"HDL_Cholesterol":48}'
```

### 5) Run API

```bash
".venv/bin/python" -m uvicorn healthcare_digital_twin.api:app --reload --port 8000
```

Then POST to `http://127.0.0.1:8000/predict`.

Dashboard/UI endpoints:

- `GET /model-info` (available models + dataset stats)
- `POST /analyze` (what-if scenarios + top drivers)
- `GET /figures/{name}` (serves confusion matrix PNGs)

### 6) Run the React frontend (UI)

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

Optional: to point the UI at a different backend URL, copy `frontend/.env.example` to
`frontend/.env.local` and set `VITE_API_BASE_URL`.

## Repo layout

- `data/` raw + processed datasets
- `src/healthcare_digital_twin/` reusable project code
- `scripts/` training/build/predict entrypoints
- `artifacts/` exported model pipelines + metadata
- `reports/figures/` saved plots
- `notebooks/` original + new reproducible notebooks
- `frontend/` React UI to call the API
