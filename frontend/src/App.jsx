import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

const FIELD_META = [
  {
    key: "Age",
    label: "Age (years)",
    type: "number",
    min: 0,
    max: 120,
    step: 1,
  },
  {
    key: "Gender",
    label: "Gender",
    type: "select",
    options: [
      { value: 0, label: "Male (0)" },
      { value: 1, label: "Female (1)" },
    ],
  },
  { key: "BMI", label: "BMI", type: "number", min: 0, max: 100, step: 0.1 },
  {
    key: "Systolic_BP",
    label: "Systolic BP (mmHg)",
    type: "number",
    min: 0,
    max: 300,
    step: 1,
  },
  {
    key: "Diastolic_BP",
    label: "Diastolic BP (mmHg)",
    type: "number",
    min: 0,
    max: 200,
    step: 1,
  },
  {
    key: "Glucose",
    label: "Glucose (mg/dL)",
    type: "number",
    min: 0,
    max: 500,
    step: 1,
  },
  {
    key: "Insulin",
    label: "Insulin (µU/mL)",
    type: "number",
    min: 0,
    max: 500,
    step: 1,
  },
  {
    key: "Total_Cholesterol",
    label: "Total Cholesterol (mg/dL)",
    type: "number",
    min: 0,
    max: 500,
    step: 1,
  },
  {
    key: "HDL_Cholesterol",
    label: "HDL Cholesterol (mg/dL)",
    type: "number",
    min: 0,
    max: 200,
    step: 1,
  },
];

const CURVE_POINTS = 21;

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function linspace(a, b, n) {
  if (n <= 1) return [a];
  const step = (b - a) / (n - 1);
  return Array.from({ length: n }, (_, i) => a + step * i);
}

function ProbabilityCurve({ points, highlightX }) {
  if (!points?.length) return null;

  const W = 360;
  const H = 140;
  const P = 14;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const xScale = (x) => P + ((x - minX) / (maxX - minX || 1)) * (W - 2 * P);
  const yScale = (y) => H - P - ((y - minY) / (maxY - minY || 1)) * (H - 2 * P);

  const poly = points
    .map((p) => `${xScale(p.x).toFixed(2)},${yScale(p.y).toFixed(2)}`)
    .join(" ");

  const hx = clamp(highlightX ?? points[0].x, minX, maxX);
  const nearest = points.reduce(
    (best, p) => {
      const d = Math.abs(p.x - hx);
      return d < best.d ? { d, p } : best;
    },
    { d: Infinity, p: points[0] },
  ).p;

  return (
    <div className="curve">
      <div className="curveTop">
        <div className="curveTitle">Risk probability vs Δ</div>
        <div className="hint">
          Δ range:{" "}
          <span className="mono">
            [{minX.toFixed(1)}, {maxX.toFixed(1)}]
          </span>
        </div>
      </div>
      <svg
        className="curveSvg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Probability curve"
      >
        <rect
          x="0"
          y="0"
          width={W}
          height={H}
          rx="10"
          ry="10"
          className="curveBg"
        />
        <line x1={P} y1={H - P} x2={W - P} y2={H - P} className="curveAxis" />
        <line x1={P} y1={P} x2={P} y2={H - P} className="curveAxis" />
        <polyline points={poly} fill="none" className="curveLine" />
        <circle
          cx={xScale(nearest.x)}
          cy={yScale(nearest.y)}
          r="4"
          className="curveDot"
        />
      </svg>
      <div className="curveMeta">
        Selected Δ: <span className="mono">{nearest.x.toFixed(2)}</span>
        {" · "}
        Prob: <span className="mono">{nearest.y.toFixed(4)}</span>
      </div>
    </div>
  );
}

function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState(null);
  const [modelInfoStatus, setModelInfoStatus] = useState("idle");
  const [modelInfoError, setModelInfoError] = useState("");
  const [selectedModel, setSelectedModel] = useState("l2");

  const hasModelInfoRef = useRef(false);
  const [form, setForm] = useState({
    Age: "",
    Gender: 0,
    BMI: "",
    Systolic_BP: "",
    Diastolic_BP: "",
    Glucose: "",
    Insulin: "",
    Total_Cholesterol: "",
    HDL_Cholesterol: "",
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [lastPayload, setLastPayload] = useState(null);

  const [whatIfFeature, setWhatIfFeature] = useState("Glucose");
  const [whatIfDelta, setWhatIfDelta] = useState(0);
  const [analysis, setAnalysis] = useState(null);

  useEffect(() => {
    let isMounted = true;
    let pollId = null;

    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
        if (!isMounted) return;
        setApiStatus(r.ok ? "connected" : "error");

        if (!r.ok) return;
        if (hasModelInfoRef.current) return;

        setModelInfoStatus("loading");
        const mi = await fetch(`${API_BASE_URL}/model-info`, {
          cache: "no-store",
        });
        if (!isMounted) return;
        if (!mi.ok) {
          const text = await mi.text().catch(() => "");
          setModelInfoError(text || `/model-info failed (${mi.status})`);
          setModelInfoStatus("error");
          return;
        }

        const data = await mi.json();
        if (!isMounted) return;
        setModelInfo(data);
        hasModelInfoRef.current = true;
        setModelInfoError("");
        setModelInfoStatus("loaded");
      } catch {
        if (!isMounted) return;
        setApiStatus("disconnected");
      }
    };

    // Poll briefly so the dashboard still loads if the backend starts after the UI.
    poll();
    pollId = window.setInterval(() => {
      poll();
    }, 1500);

    return () => {
      isMounted = false;
      if (pollId != null) window.clearInterval(pollId);
    };
  }, []);

  useEffect(() => {
    if (!modelInfo) return;
    hasModelInfoRef.current = true;
    if (modelInfoStatus !== "loaded") setModelInfoStatus("loaded");
  }, [modelInfo, modelInfoStatus]);

  const selectedModelSummary = useMemo(() => {
    if (!modelInfo?.models) return null;
    return modelInfo.models.find((m) => m.id === selectedModel) || null;
  }, [modelInfo, selectedModel]);

  const validationErrors = useMemo(() => {
    const errs = [];
    for (const field of FIELD_META) {
      const value = form[field.key];
      if (field.type === "select") continue;
      if (value === "" || value === null || value === undefined) {
        errs.push(`${field.label} is required.`);
        continue;
      }
      const num = Number(value);
      if (!Number.isFinite(num)) {
        errs.push(`${field.label} must be a number.`);
        continue;
      }
      if (typeof field.min === "number" && num < field.min)
        errs.push(`${field.label} must be ≥ ${field.min}.`);
      if (typeof field.max === "number" && num > field.max)
        errs.push(`${field.label} must be ≤ ${field.max}.`);
    }
    return errs;
  }, [form]);

  const onChange = (key) => (e) => {
    const val = e.target.value;
    setForm((prev) => ({ ...prev, [key]: val }));
  };

  const onGenderChange = (e) => {
    setForm((prev) => ({ ...prev, Gender: Number(e.target.value) }));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);
    setAnalysis(null);

    if (validationErrors.length) {
      setError(validationErrors[0]);
      return;
    }

    setIsSubmitting(true);
    try {
      const payload = {
        ...form,
        Age: Number(form.Age),
        BMI: Number(form.BMI),
        Systolic_BP: Number(form.Systolic_BP),
        Diastolic_BP: Number(form.Diastolic_BP),
        Glucose: Number(form.Glucose),
        Insulin: Number(form.Insulin),
        Total_Cholesterol: Number(form.Total_Cholesterol),
        HDL_Cholesterol: Number(form.HDL_Cholesterol),
      };

      setLastPayload(payload);

      const r = await fetch(
        `${API_BASE_URL}/predict?model=${encodeURIComponent(selectedModel)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );

      if (!r.ok) {
        const text = await r.text();
        throw new Error(text || `Request failed (${r.status})`);
      }

      const data = await r.json();
      setResult(data);
    } catch (err) {
      setError(err?.message || "Something went wrong.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const whatIfMeta = useMemo(() => {
    return FIELD_META.find((f) => f.key === whatIfFeature) || null;
  }, [whatIfFeature]);

  const whatIfBounds = useMemo(() => {
    if (!lastPayload || !whatIfMeta || whatIfMeta.type === "select")
      return null;
    const baseVal = Number(lastPayload[whatIfFeature]);
    const min = typeof whatIfMeta.min === "number" ? whatIfMeta.min : baseVal;
    const max = typeof whatIfMeta.max === "number" ? whatIfMeta.max : baseVal;
    return {
      baseVal,
      minDelta: Math.round((min - baseVal) * 10) / 10,
      maxDelta: Math.round((max - baseVal) * 10) / 10,
      step: whatIfMeta.step ?? 1,
    };
  }, [lastPayload, whatIfMeta, whatIfFeature]);

  const runWhatIf = async () => {
    setError("");
    setAnalysis(null);

    if (!lastPayload) {
      setError("Run a prediction first to enable What-If simulation.");
      return;
    }
    if (!whatIfBounds) {
      setError("Select a numeric feature for What-If simulation.");
      return;
    }

    const baseVal = whatIfBounds.baseVal;
    const delta = Number(whatIfDelta);
    const newVal = baseVal + delta;
    const scenarioName = `selected:${whatIfFeature}${delta >= 0 ? "+" : ""}${delta}`;

    // Build a small sweep so we can draw a visual curve.
    const minDelta = whatIfBounds.minDelta;
    const maxDelta = whatIfBounds.maxDelta;
    const sweep = linspace(minDelta, maxDelta, CURVE_POINTS)
      .map((d) => Math.round(d * 10) / 10)
      .filter((d) => Number.isFinite(d));

    setIsAnalyzing(true);
    try {
      const body = {
        base: lastPayload,
        scenarios: [
          { name: scenarioName, overrides: { [whatIfFeature]: newVal } },
          ...sweep.map((d) => ({
            name: `curve:${d}`,
            overrides: { [whatIfFeature]: baseVal + d },
          })),
        ],
      };

      const r = await fetch(
        `${API_BASE_URL}/analyze?model=${encodeURIComponent(selectedModel)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );

      if (!r.ok) {
        const text = await r.text();
        throw new Error(text || `Request failed (${r.status})`);
      }

      const data = await r.json();
      setAnalysis(data);
    } catch (err) {
      setError(err?.message || "Something went wrong.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const curvePoints = useMemo(() => {
    if (!analysis?.scenarios?.length || !whatIfBounds) return null;
    const pts = [];
    for (const s of analysis.scenarios) {
      if (typeof s?.name !== "string") continue;
      if (!s.name.startsWith("curve:")) continue;
      const x = Number(s.name.slice("curve:".length));
      const y = Number(s.risk_probability);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      pts.push({ x, y });
    }
    pts.sort((a, b) => a.x - b.x);
    return pts.length ? pts : null;
  }, [analysis, whatIfBounds]);

  const selectedScenario = useMemo(() => {
    if (!analysis?.scenarios?.length) return null;
    return (
      analysis.scenarios.find(
        (s) => typeof s?.name === "string" && s.name.startsWith("selected:"),
      ) || null
    );
  }, [analysis]);

  const probPct = result
    ? Math.round(Number(result.risk_probability) * 1000) / 10
    : null;
  const riskText = result
    ? result.risk_label === 1
      ? "High risk"
      : "Low risk"
    : "";

  return (
    <div className="page">
      <header className="header">
        <div className="titleRow">
          <div>
            <h1 className="title">Metabolic Risk Digital Twin</h1>
            <p className="subtitle">
              Prediction dashboard + what-if simulator (API + UI demo).
            </p>
          </div>
          <div className={`badge badge--${apiStatus}`}>API: {apiStatus}</div>
        </div>
      </header>

      <main className="grid">
        <section className="card" aria-label="Input form">
          <div className="cardHeader">
            <div>
              <h2 className="cardTitle">Patient Inputs</h2>
              <p className="help">
                Model expects the same features used in your paper notebooks.
              </p>
            </div>
            <div className="modelPicker">
              <label className="label" htmlFor="model">
                Model
              </label>
              <select
                id="model"
                className="input"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="l2">Best (L2) — default</option>
                <option value="l1">Sparse (L1) — explainable</option>
              </select>
            </div>
          </div>

          <form className="form" onSubmit={onSubmit}>
            <div className="fields">
              {FIELD_META.map((f) => (
                <div className="field" key={f.key}>
                  <label className="label" htmlFor={f.key}>
                    {f.label}
                  </label>
                  {f.type === "select" ? (
                    <select
                      id={f.key}
                      className="input"
                      value={form.Gender}
                      onChange={onGenderChange}
                    >
                      {f.options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={f.key}
                      className="input"
                      inputMode="decimal"
                      type="number"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={form[f.key]}
                      onChange={onChange(f.key)}
                      placeholder="Enter value"
                      required
                    />
                  )}
                </div>
              ))}
            </div>

            {error ? <div className="alert">{error}</div> : null}

            <div className="actions">
              <button className="button" type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Predicting…" : "Predict risk"}
              </button>
              <div className="hint">
                Backend: <span className="mono">{API_BASE_URL}</span>
              </div>
            </div>
          </form>
        </section>

        <div className="stack">
          <section className="card" aria-label="Prediction result">
            <h2 className="cardTitle">Prediction</h2>
            {!result ? (
              <p className="empty">Submit the form to see results.</p>
            ) : (
              <div className="result">
                <div className="resultTop">
                  <div>
                    <div className="resultLabel">{riskText}</div>
                    <div className="resultMeta">
                      Model: <span className="mono">{selectedModel}</span>
                      {" · "}
                      Threshold:{" "}
                      <span className="mono">{result.threshold}</span>
                    </div>
                  </div>
                  <div className="score">
                    <div className="scoreValue">{probPct}%</div>
                    <div className="scoreCaption">risk probability</div>
                  </div>
                </div>

                <div className="bar" aria-hidden="true">
                  <div
                    className="barFill"
                    style={{
                      width: `${Math.min(100, Math.max(0, probPct))}%`,
                    }}
                  />
                </div>

                <div className="resultDetails">
                  <div className="kv">
                    <span className="k">Risk label</span>
                    <span className="v mono">{result.risk_label}</span>
                  </div>
                  <div className="kv">
                    <span className="k">Probability</span>
                    <span className="v mono">{result.risk_probability}</span>
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className="card" aria-label="What-if simulator">
            <h2 className="cardTitle">What‑If Simulator</h2>
            <p className="help">
              Change one factor and see how the predicted risk shifts
              (model-driven).
            </p>

            {!lastPayload ? (
              <p className="empty">Run a prediction to enable simulation.</p>
            ) : (
              <div className="whatif">
                <div className="whatifRow">
                  <div className="field">
                    <label className="label" htmlFor="whatIfFeature">
                      Feature
                    </label>
                    <select
                      id="whatIfFeature"
                      className="input"
                      value={whatIfFeature}
                      onChange={(e) => {
                        setWhatIfFeature(e.target.value);
                        setWhatIfDelta(0);
                      }}
                    >
                      {FIELD_META.filter((f) => f.type !== "select").map(
                        (f) => (
                          <option key={f.key} value={f.key}>
                            {f.label}
                          </option>
                        ),
                      )}
                    </select>
                  </div>

                  <div className="field">
                    <label className="label" htmlFor="whatIfDelta">
                      Change (Δ)
                    </label>
                    <input
                      id="whatIfDelta"
                      className="input"
                      type="number"
                      step={whatIfBounds?.step ?? 1}
                      min={whatIfBounds?.minDelta ?? undefined}
                      max={whatIfBounds?.maxDelta ?? undefined}
                      value={whatIfDelta}
                      onChange={(e) => setWhatIfDelta(e.target.value)}
                    />
                  </div>
                </div>

                <div className="actions">
                  <button
                    className="button"
                    type="button"
                    onClick={runWhatIf}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? "Analyzing…" : "Run what‑if"}
                  </button>
                  {whatIfBounds ? (
                    <div className="hint">
                      Base: <span className="mono">{whatIfBounds.baseVal}</span>
                      {" · "}
                      Range Δ:{" "}
                      <span className="mono">
                        [{whatIfBounds.minDelta}, {whatIfBounds.maxDelta}]
                      </span>
                    </div>
                  ) : null}
                </div>

                {!analysis ? null : (
                  <div className="whatifOut">
                    <div className="kv">
                      <span className="k">Baseline probability</span>
                      <span className="v mono">
                        {analysis.baseline.risk_probability}
                      </span>
                    </div>
                    <div className="kv">
                      <span className="k">New probability</span>
                      <span className="v mono">
                        {selectedScenario?.risk_probability ??
                          analysis.scenarios?.[0]?.risk_probability}
                      </span>
                    </div>
                    <div className="kv">
                      <span className="k">Δ probability</span>
                      <span className="v mono">
                        {selectedScenario?.delta_probability ??
                          analysis.scenarios?.[0]?.delta_probability}
                      </span>
                    </div>

                    {curvePoints ? (
                      <ProbabilityCurve
                        points={curvePoints}
                        highlightX={Number(whatIfDelta)}
                      />
                    ) : null}

                    {selectedScenario?.top_drivers?.length ? (
                      <div className="drivers">
                        <div className="driversTitle">
                          Top drivers (scenario)
                        </div>
                        <ul className="driversList">
                          {selectedScenario.top_drivers.map((d) => (
                            <li key={d.feature} className="driversItem">
                              <span className="mono">{d.feature}</span>
                              <span className="driversHint">
                                {d.direction} risk (
                                {Number(d.contribution).toFixed(3)})
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="card" aria-label="Model dashboard">
            <h2 className="cardTitle">Model Dashboard</h2>
            {!modelInfo ? (
              <div>
                {modelInfoStatus === "error" ? (
                  <div>
                    <div className="alert">
                      Failed to load model stats from{" "}
                      <span className="mono">{API_BASE_URL}</span>.
                      {modelInfoError ? (
                        <div className="mono" style={{ marginTop: 6 }}>
                          {modelInfoError}
                        </div>
                      ) : null}
                    </div>
                    <div className="actions">
                      <button
                        className="button"
                        type="button"
                        onClick={() => {
                          setModelInfoError("");
                          setModelInfoStatus("idle");
                          // Trigger an immediate reload attempt.
                          fetch(`${API_BASE_URL}/model-info`, {
                            cache: "no-store",
                          })
                            .then(async (mi) => {
                              if (!mi.ok) {
                                const text = await mi.text().catch(() => "");
                                throw new Error(
                                  text || `/model-info failed (${mi.status})`,
                                );
                              }
                              return mi.json();
                            })
                            .then((data) => {
                              setModelInfo(data);
                              setModelInfoStatus("loaded");
                            })
                            .catch((err) => {
                              setModelInfoError(
                                err?.message || "Request failed",
                              );
                              setModelInfoStatus("error");
                            });
                        }}
                      >
                        Retry
                      </button>
                      <div className="hint">
                        Tip: ensure the backend is running and CORS allows this
                        origin.
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="empty">Loading model and dataset stats…</p>
                )}
              </div>
            ) : (
              <div className="dash">
                <div className="dashGrid">
                  <div className="kv">
                    <span className="k">Dataset rows</span>
                    <span className="v mono">{modelInfo.dataset.n_rows}</span>
                  </div>
                  <div className="kv">
                    <span className="k">Positive rate</span>
                    <span className="v mono">
                      {modelInfo.dataset.positive_rate != null
                        ? Math.round(modelInfo.dataset.positive_rate * 1000) /
                            10 +
                          "%"
                        : "n/a"}
                    </span>
                  </div>
                  <div className="kv">
                    <span className="k">Female rate</span>
                    <span className="v mono">
                      {modelInfo.dataset.female_rate != null
                        ? Math.round(modelInfo.dataset.female_rate * 1000) /
                            10 +
                          "%"
                        : "n/a"}
                    </span>
                  </div>
                  <div className="kv">
                    <span className="k">Selected model</span>
                    <span className="v mono">
                      {selectedModelSummary?.artifact_name || selectedModel}
                    </span>
                  </div>
                </div>

                {selectedModelSummary?.metrics ? (
                  <div className="dashGrid">
                    {Object.entries(selectedModelSummary.metrics).map(
                      ([k, v]) => (
                        <div className="kv" key={k}>
                          <span className="k">Test {k}</span>
                          <span className="v mono">{Number(v).toFixed(4)}</span>
                        </div>
                      ),
                    )}
                  </div>
                ) : null}

                {selectedModelSummary?.confusion_matrix_figure ? (
                  <div className="figure">
                    <div className="figureTitle">
                      Confusion matrix (test set)
                    </div>
                    <p className="help">
                      This is computed once during training on a held-out test
                      split, so it will not change when you edit a single
                      patient’s inputs. It may change when you switch models (L1
                      vs L2) or retrain.
                    </p>
                    <img
                      className="figureImg"
                      alt="Confusion matrix"
                      src={`${API_BASE_URL}/figures/${selectedModelSummary.confusion_matrix_figure}?v=${encodeURIComponent(
                        selectedModelSummary.created_at || selectedModel,
                      )}`}
                    />
                  </div>
                ) : null}
              </div>
            )}
          </section>
        </div>
      </main>

      <footer className="footer">
        <span>
          Tip: start the backend with{" "}
          <span className="mono">
            ".venv/bin/python" -m uvicorn healthcare_digital_twin.api:app --port
            8000
          </span>
        </span>
      </footer>
    </div>
  );
}

export default App;
