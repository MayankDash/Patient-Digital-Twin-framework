# Test Case Report — HealthCareDigitalTwinn

This report provides an industry-style test plan and test case matrix for the HealthCareDigitalTwinn project (ML pipeline + API + UI). It includes automated test results and manual acceptance checks.

## 1) Test scope and objectives

**In scope**

- ML pipeline behavior (training, artifacts, inference compatibility)
- API contract and key endpoints: `/health`, `/predict`, `/model-info`, `/analyze`, `/figures/{name}`
- Security hardening checks for file-serving endpoint (`/figures`)
- Frontend quality gates (lint/build) and end-to-end UI user journeys

**Out of scope (explicitly)**

- Clinical validation (this is an analytics demo, not a medical device)
- Real-world bias/fairness audits beyond basic dataset summary
- Load testing / performance benchmarks

## 2) Test environment (execution evidence)

- OS: macOS
- Python: 3.13.7 (project venv)
- FastAPI: 0.135.3
- scikit-learn: 1.8.0
- Node: v24.9.0
- npm: 11.6.0

## 3) Test strategy (levels)

- **Unit/Component tests:** pure-Python logic and schema validation (pytest)
- **API integration tests:** FastAPI `TestClient` exercises endpoints and contracts (pytest)
- **Frontend gates:** ESLint + production build (Vite)
- **Manual acceptance:** browser journeys covering the predictor and what-if simulator

### 3.1 Entry and exit criteria

**Entry criteria**

- Dependencies installed (Python venv + Node modules)
- For tests that require trained artifacts/figures: artifacts present under `artifacts/` and figures under `reports/figures/` (otherwise those specific tests SKIP)

**Exit criteria**

- Backend automated tests meet the run’s pass criteria (no unexpected failures)
- Frontend quality gates pass (`npm run lint`, `npm run build`)
- Manual acceptance tests for critical journeys (predict + what-if) executed and PASS for submission

### 3.2 Test data and prerequisites

- Automated tests use synthetic data for ML smoke validation (no external downloads needed)
- For full UI/API demonstration with real data, ensure the processed dataset exists under `data/processed/` and run training to generate artifacts

## 4) Automated test execution

### 4.1 Backend automated tests (pytest)

**Command**

```bash
cd HealthCareDigitalTwinn
.venv/bin/python -m pytest -q
```

**Result (local run evidence)**

```text
9 passed in 1.33s
```

### 4.2 Frontend quality gates (lint + build)

**Command**

```bash
cd frontend
npm run lint
npm run build
```

**Result (local run evidence)**

- ESLint: pass
- Vite build: pass

## 5) Test case matrix (industry-style)

Status definitions:

- **PASS:** executed and met expected results
- **SKIP:** intentionally skipped due to missing prerequisites (e.g., artifacts)
- **READY:** written and defined; to be executed manually in a browser

### 5.1 Backend / API test cases (automated)

| ID      | Area     | Test case                                    | Expected result                                            | Status         | Evidence                                                                              |
| ------- | -------- | -------------------------------------------- | ---------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------- |
| API-000 | API      | `GET /health`                                | HTTP 200; returns `{status: ok}`                           | PASS           | `tests/test_api_endpoints.py::test_health_endpoint`                                   |
| API-001 | API      | `GET /model-info`                            | HTTP 200; returns `default_model`, `dataset`, `models[]`   | PASS           | `tests/test_api_endpoints.py::test_model_info_endpoint`                               |
| API-002 | API      | `POST /analyze` smoke                        | HTTP 200; baseline prob in [0,1]; scenarios include deltas | PASS           | `tests/test_api_endpoints.py::test_analyze_endpoint_smoke`                            |
| API-003 | API      | `/analyze` multi-scenario length             | Response scenarios length equals request length            | PASS           | `tests/test_api_endpoints.py::test_analyze_multiple_scenarios_length_matches_request` |
| API-004 | API      | `/predict` model switching (L1 if available) | `model=l2` and `model=l1` both return prob in [0,1]        | PASS or SKIP\* | `tests/test_api_endpoints.py::test_predict_model_switching_l1_if_available`           |
| API-005 | Security | `/figures` rejects traversal                 | Names with `/`, `\\`, `..` or non-`.png` rejected          | PASS           | `tests/test_api_endpoints.py::test_figures_rejects_path_traversal_and_non_png`        |
| API-006 | Interop  | CORS preflight for Vite origin               | OPTIONS returns allow-origin for `http://localhost:5174`   | PASS           | `tests/test_api_endpoints.py::test_cors_preflight_allows_local_vite_origin`           |
| API-007 | API      | `/figures/{name}` smoke (known PNG)          | If confusion matrix exists, it is served as an image       | PASS or SKIP\* | `tests/test_api_endpoints.py::test_figures_endpoint_smoke`                            |

\*PASS if required artifacts/figures exist; otherwise marked SKIP by the test.

### 5.2 ML pipeline test cases (automated)

| ID     | Area     | Test case           | Expected result                                               | Status | Evidence                                            |
| ------ | -------- | ------------------- | ------------------------------------------------------------- | ------ | --------------------------------------------------- |
| ML-001 | Modeling | Train+predict smoke | Training returns metrics incl. `f1`; schema validates payload | PASS   | `tests/test_smoke.py::test_train_and_predict_smoke` |

### 5.3 Frontend test cases (quality gates + manual acceptance)

| ID     | Area     | Test case             | Expected result                                                             | Status | Evidence           |
| ------ | -------- | --------------------- | --------------------------------------------------------------------------- | ------ | ------------------ |
| UI-001 | Frontend | ESLint                | `npm run lint` succeeds                                                     | PASS   | `frontend` scripts |
| UI-002 | Frontend | Production build      | `npm run build` succeeds                                                    | PASS   | `frontend` scripts |
| UI-003 | Journey  | Predict risk          | Inputs → Predict → label + probability render                               | READY  | Browser run        |
| UI-004 | Journey  | What-if curve updates | Run prediction → choose feature + Δ → curve renders; selected point updates | READY  | Browser run        |
| UI-005 | Journey  | Model switch L2↔L1    | Toggle model; predict/what-if continue working                              | READY  | Browser run        |
| UI-006 | Negative | Missing fields        | UI blocks submit or shows a validation message                              | READY  | Browser run        |
| UI-007 | Negative | Backend offline       | UI shows API disconnected; no crashes                                       | READY  | Browser run        |

## 6) Execution notes / prerequisites

- Some tests intentionally **skip** if model artifacts are missing. To generate artifacts:

```bash
.venv/bin/python scripts/train.py --model logreg --tune --penalty l2 --threshold 0.4 --artifact-name logreg_pipeline
.venv/bin/python scripts/train.py --model logreg --tune --penalty l1 --threshold 0.4 --artifact-name logreg_pipeline_l1
```

## 7) Summary

- Backend automated suite: **PASS** (9/9)
- Frontend gates: **PASS** (lint + build)
- Manual UI acceptance tests: **READY** (defined and repeatable)
