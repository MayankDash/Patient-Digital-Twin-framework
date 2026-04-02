# Demo Video Script (2–3 minutes) — Assessment 5

## 0. Title card (5 seconds)

“HealthCareDigitalTwinn — Metabolic Risk Digital Twin using NHANES 2017–2018”

---

## 1. Problem statement (15–20 seconds)

“Metabolic disorders like diabetes are often under-detected. This project builds a patient digital twin that predicts metabolic risk from common clinical measurements. I trained and deployed a machine learning model using NHANES 2017–2018 data, and I made it reproducible and demo-ready with scripts, artifacts, an API, and a UI.”

---

## 2. What a Digital Twin means here (15–20 seconds)

“In this project, a digital twin is: patient inputs → a trained model → a risk probability and a decision label. The probability allows different thresholds depending on whether we want screening or confirmatory behavior.”

---

## 3. Show system architecture (20–30 seconds)

(Show repo tree or a simple diagram slide)

- “Raw NHANES XPT files are merged into a clean patient-level dataset.”
- “A scikit-learn pipeline does imputation, scaling, and logistic regression.”
- “Artifacts are exported for consistent inference.”
- “FastAPI serves `/predict`, and a React UI calls it.”

---

## 4. Show the model and results (30–40 seconds)

(Show metadata JSON briefly)

- “I tuned logistic regression with 5-fold stratified CV, optimizing F1.”
- “Best hyperparameters were C=0.1, penalty=l2.”
- “I evaluated on a held-out test set and stored all metrics in artifacts.”

Mention threshold comparison:

- “Default decision rule F1 is 0.6844.”
- “With a screening threshold of 0.4, recall improves and F1 becomes 0.7090.”

---

## 5. Live demo (40–60 seconds)

(Show terminal + browser)

1. “Backend is running.” (Show `/health`)
2. “Now I’ll use the UI to enter a patient profile.”
3. Click Predict.
4. “We get a probability and label. The threshold policy is shown, so it’s transparent.”

---

## 6. Testing and reproducibility (15–20 seconds)

“I also added tests with pytest, and the pipeline is reproducible because preprocessing is in a single pipeline and artifacts store the exact feature list, best params, and metrics.”

---

## 7. Closing (5–10 seconds)

“Overall, this project turns notebook research into a deployable digital twin with validated results and a clean demo experience.”
