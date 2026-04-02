# Demo Checklist — Assessment 5

## 1. Pre-demo setup (5–10 minutes before)

- [ ] Confirm Python venv exists and dependencies installed
- [ ] Confirm Node dependencies installed in `frontend/`
- [ ] Confirm `artifacts/` contains trained model + metadata
- [ ] Close other apps using ports 8000 and 5173

---

## 2. Start backend (FastAPI)

From project root:

```bash
".venv/bin/python" -m uvicorn healthcare_digital_twin.api:app --reload --port 8000
```

Quick check:

```bash
curl -s http://127.0.0.1:8000/health
```

---

## 3. Start frontend (React)

```bash
cd frontend
npm run dev
```

Open:

- `http://127.0.0.1:5173`

---

## 4. Live demo steps (what to show)

1. Show `/health` is OK (curl or browser)
2. In the UI, enter a sample patient record
3. Click Predict
4. Explain output:
   - `risk_probability` is the model’s estimate
   - `risk_label` is derived from the threshold policy
5. Mention that threshold can be chosen for screening (e.g., 0.4) to increase recall
6. Show evidence artifacts:
   - metadata JSON with metrics in `artifacts/`
   - confusion matrix in `reports/figures/`

---

## 5. Backup plan (if something fails)

- If frontend fails: demo with `curl /predict`
- If API fails: show artifacts and metrics from metadata JSON
- If training takes time: do not retrain live; use pre-generated artifacts

---

## 6. Evidence to capture

- [ ] Screenshot: UI prediction result
- [ ] Screenshot: terminal `pytest` passing
- [ ] Screenshot: `artifacts/` folder showing metadata JSON
- [ ] Screenshot: confusion matrix image in `reports/figures/`
