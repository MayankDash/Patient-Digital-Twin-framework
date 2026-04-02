# Poster Outline (A1) — HealthCareDigitalTwinn

Use this as the content blueprint for an A1 poster. Keep visuals large and text minimal.

---

## 1. Title + authors

- Title: “Metabolic Risk Digital Twin using NHANES 2017–2018”
- Your name, course, professor, institution

---

## 2. Motivation / Problem

- Early identification of metabolic risk is important
- Screening should prioritize sensitivity/recall

---

## 3. Dataset

- NHANES 2017–2018
- Patient-level merge using SEQN
- Final dataset: `patient_state_clean.csv`

Include: a small table preview and number of rows/columns.

---

## 4. Digital Twin definition

A digital twin here = patient inputs + ML model + decision policy

- Inputs: Age, Gender, BMI, BP, Glucose, Insulin, Cholesterol, HDL
- Output: probability + label

---

## 5. Method

Pipeline:

1. Median imputation
2. StandardScaler
3. Logistic Regression

Validation:

- 80/20 stratified split
- 5-fold stratified CV
- Metric: F1

---

## 6. Results (from artifacts)

Include a small metrics table:

- Best CV F1: 0.7369 (C=0.1, penalty=l2)
- Test (default): Acc 0.7552, F1 0.6844
- Test (threshold 0.4): Acc 0.7552, F1 0.7090 (higher recall)

Add 1–2 confusion matrix plots.

---

## 7. System demo (architecture)

Show a block diagram:

- Data → Training Script → Artifacts → FastAPI `/predict` → React UI

---

## 8. Conclusion

- Reproducible ML + deployable demo
- Threshold policy enables screening vs confirmatory behavior

---

## 9. Limitations / future work

- External validation required for clinical deployment
- Consider calibration, additional features, fairness checks

---

## 10. QR code (optional)

- Link to repository or demo video
