# Validation Report — HealthCareDigitalTwinn

This document provides a validation narrative (intended use + acceptance criteria + traceability) for the HealthCareDigitalTwinn project. It references the executed automated tests and defines the remaining manual evidence needed for a complete “validation package”.

## 1) Intended use and non-intended use

**Intended use**

- Academic/demo application that estimates metabolic risk from a fixed set of patient features.
- Demonstrates: reproducible ML pipeline, saved artifacts, API inference, and a UI that performs “what-if” analysis.

**Not intended use**

- Not for clinical diagnosis, treatment decisions, or real-world screening without external clinical validation.

## 2) Validation approach

Validation is supported by:

- Automated verification via pytest (API contract + security + ML smoke)
- Frontend quality gates (lint/build)
- Manual acceptance tests for end-to-end UI behavior (defined in the test report)

**Primary evidence**

- Test case matrix and executed automated results in `docs/assessment5/TEST_REPORT.md`
- Saved model artifacts/metadata (when generated) under `artifacts/`

## 3) Acceptance criteria

The project is considered validated for intended academic/demo use when:

- Automated backend tests pass (no unexpected failures)
- Frontend lint/build passes
- UI acceptance journeys (predict + what-if) are executed and captured as evidence

## 4) Validation status (current)

- **Automated backend verification:** PASS (pytest suite executed successfully)
- **Frontend quality gates:** PASS
- **Manual UI validation evidence:** PENDING (test cases defined; execution evidence not embedded here)

## 5) Reproducibility and controls

- Training and inference use a scikit-learn `Pipeline` (imputation + scaling + classifier) to reduce leakage risk.
- Exported artifacts include metadata (feature list, parameters, metrics, threshold if used).
- Deterministic splitting and CV configuration are controlled in training code (where applicable).

## 6) Data leakage prevention

- Label-derived fields are excluded from model input features.
- Preprocessing is fit only on training data via Pipeline.

## 7) API validation summary

The API is validated by automated tests against the expected contract for:

- `/health`
- `/model-info`
- `/predict` (including model selection where artifacts exist)
- `/analyze`
- `/figures/{name}` (with traversal and type hardening)
- CORS preflight compatibility with the local Vite dev origin

## 8) Requirements → tests traceability matrix

This matrix maps validation requirements to the test cases defined/executed in `docs/assessment5/TEST_REPORT.md`.

| Requirement ID | Requirement statement                                        | Evidence (test case IDs) |
| -------------- | ------------------------------------------------------------ | ------------------------ |
| VAL-REQ-001    | API exposes a readiness endpoint                             | API-000                  |
| VAL-REQ-002    | API provides model/dataset overview for UI                   | API-001                  |
| VAL-REQ-003    | API supports baseline + scenario what-if analysis            | API-002, API-003         |
| VAL-REQ-004    | API returns prediction probability and decision consistently | API-004, ML-001          |
| VAL-REQ-005    | File-serving endpoint is hardened (no traversal; PNG only)   | API-005, API-007         |
| VAL-REQ-006    | Frontend can call backend cross-origin in local dev          | API-006                  |
| VAL-REQ-007    | Frontend quality gates succeed for submission                | UI-001, UI-002           |
| VAL-REQ-008    | Critical UI journeys work end-to-end                         | UI-003, UI-004, UI-005   |
| VAL-REQ-009    | Negative UI behaviors are safe (no crash; clear feedback)    | UI-006, UI-007           |

## 9) Risks and limitations

- Dataset is cross-sectional; the model is a risk estimator for demo purposes.
- External generalization requires validation on a local population.
- Threshold selection should be driven by domain cost trade-offs and is configurable at inference.

## 10) Deviations / open items

- UI acceptance tests are defined but require final execution evidence (screenshots/recording) to claim “full validation”.

## 11) Sign-off statement

Based on the executed automated verification (backend + frontend gates) and the defined manual acceptance tests, the system is **verified** and **conditionally validated** for the stated academic/demo intended use, with the condition that UI acceptance evidence is captured before final submission.
