import { useEffect, useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-basic-dist-min'
import { getArchitecture, getBenchmarks, getModelCard, getReadiness, inspectEcg, predictEcg } from './api'
import type { Architecture, Benchmarks, ConfusionCell, Inspection, ModelCard, Prediction, Readiness } from './types'
import './App.css'

const LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
const Plot = createPlotlyComponent(Plotly)
const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`
const numberOrDash = (value: number | null, digits = 1) => value == null ? '—' : value.toFixed(digits)

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <article className="metric-card"><p>{label}</p><strong>{value}</strong><span>{detail}</span></article>
}

function ClinicianHero() {
  return (
    <section id="top" className="hero-section">
      <div className="hero-copy">
        <div className="hero-kicker"><span>Research prototype</span> ECG decision support</div>
        <h1>Review the tracing.<br /><em>See the evidence.</em></h1>
        <p>A focused workspace for MI-pattern screening from a ten-second, twelve-lead ECG. The result is shown with signal findings and the operating threshold used to reach it.</p>
        <div className="hero-actions"><a className="button primary" href="#review">Review an ECG</a><a className="button ghost" href="#evidence">View validation evidence</a></div>
        <div className="boundary"><strong>Decision support only</strong><span>Correlate with symptoms, serial ECGs, biomarkers and local clinical protocol.</span></div>
      </div>
      <div className="clinical-overview" aria-label="Clinical review workflow">
        <div className="overview-head"><span>CLINICIAN WORKFLOW</span><strong>One ECG, one review path</strong></div>
        <ol>
          <li><span>01</span><div><strong>Validate the signal</strong><small>Lead order, duration, sampling rate and quality</small></div></li>
          <li><span>02</span><div><strong>Inspect the tracing</strong><small>Twelve leads with rhythm and morphology context</small></div></li>
          <li><span>03</span><div><strong>Review the screen</strong><small>Probability, threshold and measured characteristics</small></div></li>
        </ol>
        <div className="overview-foot"><i /> The model abstains when the frozen quality policy fails.</div>
      </div>
    </section>
  )
}

function ThresholdScale({ score, threshold }: { score: number; threshold: number }) {
  const delta = (score - threshold) * 100
  return (
    <div className="threshold-panel">
      <div className="threshold-copy">
        <div><small>DECISION THRESHOLD</small><strong>{percent(threshold)}</strong></div>
        <p>{delta >= 0 ? `${delta.toFixed(1)} percentage points above threshold` : `${Math.abs(delta).toFixed(1)} percentage points below threshold`}</p>
      </div>
      <div className="threshold-scale" aria-label={`Score ${percent(score)}, threshold ${percent(threshold)}`}>
        <div className="threshold-zone low">Below threshold</div><div className="threshold-zone high">At or above threshold</div>
        <span className="threshold-marker" style={{ left: `${threshold * 100}%` }}><i /><b>Threshold</b></span>
        <span className="score-marker" style={{ left: `${Math.min(98, Math.max(2, score * 100))}%` }}><i /><b>Your ECG</b></span>
      </div>
      <p className="threshold-footnote">This cut-off is fixed for every ECG. It is not personalized or changed after seeing the result.</p>
    </div>
  )
}

function SignalCharacteristics({ prediction }: { prediction: Prediction }) {
  const c = prediction.signal_characteristics
  return (
    <div className="characteristics-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">Signal characteristics</span><h3>Measurements supporting review</h3></div>
        <span className={`quality-pill ${c.quality.state.toLowerCase()}`}>Quality {c.quality.state}</span>
      </div>
      <div className="finding-grid">
        <article><small>HEART RATE</small><strong>{numberOrDash(c.rhythm.heart_rate_bpm, 0)} <em>bpm</em></strong><span>{c.rhythm.heart_rate_context}</span></article>
        <article><small>RR VARIABILITY</small><strong>{numberOrDash(c.rhythm.rr_iqr_ms, 0)} <em>ms IQR</em></strong><span>Median RR {numberOrDash(c.rhythm.rr_median_ms, 0)} ms · CV {numberOrDash(c.rhythm.rr_cv, 2)}</span></article>
        <article><small>ST DISPLACEMENT ≥0.10 mV</small><strong>{numberOrDash(c.st_t.st_positive_leads, 0)} ↑ / {numberOrDash(c.st_t.st_negative_leads, 0)} ↓</strong><span>Number of leads above / below the reference level</span></article>
        <article><small>T-WAVE POLARITY</small><strong>{numberOrDash(c.st_t.t_inversion_leads, 0)} <em>leads</em></strong><span>Leads with extracted negative T-wave polarity</span></article>
        <article><small>FRONTAL AXIS PROXY</small><strong>{numberOrDash(c.spatial.frontal_axis_proxy_deg, 0)}<em>°</em></strong><span>Waveform-derived proxy; clinician confirmation required</span></article>
        <article><small>PRECORDIAL TRANSITION</small><strong>{c.spatial.precordial_transition ?? '—'}</strong><span>Estimated R/S transition lead</span></article>
      </div>
      <div className="territory-block">
        <div className="territory-head"><strong>Regional ST/T summary</strong><span>Values are measurements, not automated ECG statements.</span></div>
        <div className="territory-table">
          <div className="territory-row territory-labels"><span>Region</span><span>Mean ST</span><span>Max |ST|</span><span>Abnormal leads</span><span>T inversion</span></div>
          {c.st_t.territories.map((item) => (
            <div className="territory-row" key={item.name}>
              <span><strong>{item.name}</strong><small>{item.leads}</small></span>
              <span>{numberOrDash(item.st_mean_mv, 3)} mV</span><span>{numberOrDash(item.st_absmax_mv, 3)} mV</span>
              <span>{numberOrDash(item.st_abnormal_leads, 0)}</span><span>{item.t_inversion_fraction == null ? '—' : percent(item.t_inversion_fraction, 0)}</span>
            </div>
          ))}
        </div>
      </div>
      <details className="measurement-note"><summary>Measurement limits</summary><p>{c.availability.interval_reason}. {c.availability.intervals_omitted.join(', ')} are therefore omitted from this review rather than shown as unreliable values.</p></details>
    </div>
  )
}

function ClinicalResult({ prediction }: { prediction: Prediction }) {
  const positive = prediction.prediction === 'MI_PATTERN'
  return (
    <article className={`clinical-result ${positive ? 'positive' : 'negative'}`} aria-live="polite">
      <div className="result-topbar"><div><span className="eyebrow">Screening result</span><h2>{positive ? 'MI pattern screen positive' : 'MI pattern screen negative'}</h2></div><span className="result-status"><i />{positive ? 'Review priority' : 'Below review threshold'}</span></div>
      <div className="result-layout">
        <div className="decision-panel">
          <div className="probability-ring" style={{ '--score': `${prediction.mi_pattern_probability * 360}deg` } as CSSProperties}><div><strong>{percent(prediction.mi_pattern_probability)}</strong><span>MI-pattern probability</span></div></div>
          <h3>{positive ? 'Above the frozen operating threshold' : 'Below the frozen operating threshold'}</h3>
          <p>{positive ? 'Review the tracing promptly and correlate the screen with the complete clinical picture.' : 'A negative screen does not rule out MI when symptoms or other clinical evidence remain concerning.'}</p>
          <ThresholdScale score={prediction.mi_pattern_probability} threshold={prediction.decision_threshold} />
        </div>
        <SignalCharacteristics prediction={prediction} />
      </div>
      <div className="result-disclaimer"><strong>Clinical boundary</strong><span>This output identifies an MI-like pattern in the submitted ECG. It is not a diagnosis, infarct-timing estimate or future-event risk score.</span><code>{prediction.signal_sha256.slice(0, 12)}</code></div>
    </article>
  )
}

function UploadPanel({ readiness }: { readiness: Readiness | null }) {
  const [file, setFile] = useState<File | null>(null)
  const [inspection, setInspection] = useState<Inspection | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  async function inspect() { if (!file) return; setBusy(true); setMessage(''); setPrediction(null); try { setInspection(await inspectEcg(file)) } catch (error) { setMessage(error instanceof Error ? error.message : 'Inspection failed') } finally { setBusy(false) } }
  async function predict() { if (!file) return; setBusy(true); setMessage(''); try { setPrediction(await predictEcg(file)) } catch (error) { setMessage(error instanceof Error ? error.message : 'Prediction failed') } finally { setBusy(false) } }
  const traces = useMemo(() => {
    if (!inspection?.valid || !inspection.input) return []
    return LEAD_ORDER.map((lead, index) => ({ x: inspection.input!.preview[lead].map((_, sample) => sample / 20), y: inspection.input!.preview[lead].map((value) => value + (11 - index) * 3), type: 'scatter' as const, mode: 'lines' as const, name: lead, line: { color: index < 6 ? '#c93647' : '#3e5f8a', width: 1.25 }, hovertemplate: `${lead} · %{x:.2f}s · %{y:.3f}<extra></extra>` }))
  }, [inspection])
  return (
    <section id="review" className="section analysis-section">
      <div className="section-heading left-heading"><span className="eyebrow">ECG review</span><h2>Start with the signal.</h2><p>Upload a de-identified ten-second ECG. Structural validation and quality control run before the screening model.</p></div>
      <div className="analysis-grid">
        <div className="upload-card"><div className="step-label">STEP 1 · LOAD ECG</div><label className="drop-zone"><input type="file" accept=".csv,.json,.zip" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setInspection(null); setPrediction(null); setMessage('') }} /><span className="upload-icon">+</span><strong>{file ? file.name : 'Choose ECG file'}</strong><small>WFDB ZIP · 12-column CSV · JSON · max 5 MB</small></label><div className="button-row"><button className="button secondary" disabled={!file || busy} onClick={inspect}>{busy ? 'Working…' : 'Validate signal'}</button><button className="button primary" disabled={!inspection?.valid || busy || !readiness?.model_ready} onClick={predict}>{busy ? 'Working…' : 'Run screening'}</button></div>
          {!readiness?.model_ready && <div className="notice amber"><strong>Model unavailable</strong><span>The verified bundle must be mounted before screening.</span></div>}
          {inspection && <div className={`notice ${inspection.valid ? 'green' : 'red'}`}><strong>{inspection.valid ? 'Signal accepted' : 'Input rejected'}</strong><span>{inspection.valid ? '12 leads · 10 seconds · 100 Hz · ready for review' : inspection.errors.join(' · ')}</span></div>}
          {message && <div className="notice amber"><strong>Could not complete review</strong><span>{message}</span></div>}
          <div className="privacy-note"><strong>De-identify before upload</strong><span>The prototype processes the file in the local API and does not create a patient record.</span></div>
        </div>
        <div className="waveform-card"><div className="card-title"><div><span className="live-dot" /> Twelve-lead tracing</div><small>{inspection?.input?.signal_sha256.slice(0, 12) ?? 'awaiting ECG'}</small></div>{traces.length ? <Plot data={traces} layout={{ autosize: true, height: 430, margin: { l: 30, r: 10, t: 10, b: 32 }, paper_bgcolor: 'transparent', plot_bgcolor: '#fffdfd', showlegend: false, xaxis: { title: { text: 'Seconds' }, gridcolor: '#ecdfe0', zeroline: false }, yaxis: { showticklabels: false, gridcolor: '#f2e9ea', zeroline: false } }} config={{ displayModeBar: false, responsive: true }} useResizeHandler style={{ width: '100%' }} /> : <div className="empty-waveform"><div className="pulse-line" /><strong>No tracing loaded</strong><p>A validated twelve-lead preview will appear here.</p></div>}</div>
      </div>
      {prediction && <ClinicalResult prediction={prediction} />}
    </section>
  )
}

function ArchitecturePanel({ architecture }: { architecture: Architecture | null }) {
  const active = architecture?.routes
  return (
    <section id="how-it-works" className="section architecture-section">
      <div className="section-heading"><span className="eyebrow">How it works</span><h2>One validated ECG. Two fixed predictive routes.</h2><p>Technical implementation is kept here so the clinical review remains focused on the patient-facing output.</p></div>
      <div className="flow"><div className="flow-node common-node"><small>COMMON CLINICAL INPUT</small><strong>12-lead ECG · 10 seconds · 100 Hz</strong><span>Structural validation → quality control → minimally processed signal</span></div><div className="route-split" aria-hidden="true"><span /><span /></div><div className="route-grid">
        <article className="route-card quantum-card"><div className="route-label"><i /> Quantum route {active?.quantum.active ? 'active' : ''}</div><h3>Waveform representation</h3><ol><li>Compact Transformer creates h128</li><li>Fold-frozen PLS compresses h128 → q4</li><li>Angle encoding enters a 4-qubit VQC</li></ol><code>quantum score sQ</code></article>
        <article className="route-card classical-card"><div className="route-label"><i /> Classical route {active?.classical.active ? 'active' : ''}</div><h3>Clinical morphology</h3><ol><li>106 deployable ECG measurements</li><li>Frozen missing-value conditioning</li><li>Histogram gradient boosting</li></ol><code>classical score sC</code></article>
      </div><div className="fusion-node"><small>FIXED FUSION AND CALIBRATION</small><strong>σ(β₀ + βQ·sQ + βC·sC)</strong><span>Both routes run for every accepted ECG → calibrated MI-pattern probability</span></div></div>
      <div className="architecture-notes"><article><strong>No route gate</strong><span>No per-patient model selection or fallback route.</span></article><article><strong>One output</strong><span>Route scores are technical inputs, not separate diagnoses.</span></article><article><strong>Quality abstention</strong><span>No result is produced when the frozen QC policy fails.</span></article></div>
    </section>
  )
}

function ConfusionTile({ kind, label, cell, tone }: { kind: string; label: string; cell?: ConfusionCell; tone: string }) {
  if (!cell) return null
  return <article className={`confusion-tile ${tone}`}><div><span>{kind}</span><strong>{cell.count.toLocaleString()}</strong></div><b>{percent(cell.rate, 2)}</b><p>{label}<small>{cell.denominator}</small></p></article>
}

function EvidencePanel({ benchmarks, modelCard }: { benchmarks: Benchmarks | null; modelCard: ModelCard | null }) {
  const rows = benchmarks?.stabilized_two_seed ?? []
  const threshold = benchmarks?.threshold_selection
  const cm = benchmarks?.confusion_matrix
  return (
    <section id="evidence" className="section evidence-section">
      <div className="section-heading"><span className="eyebrow">Validation evidence</span><h2>Performance at the clinical operating point.</h2><p>Patient-separated PTB-XL development evidence. Counts and percentages describe research ECGs, not the prevalence or performance expected in a hospital population.</p></div>
      <div className="metrics-row"><Metric label="Development ECGs" value="17,348" detail="14,958 patients" /><Metric label="Hybrid AUPRC" value="0.8360" detail="ranking across thresholds" /><Metric label="Sensitivity" value="77.79%" detail="at ≈90% specificity" /><Metric label="Brier score" value="0.0896" detail="lower is better" /></div>
      <div className="threshold-evidence"><div className="threshold-number"><small>FROZEN DECISION THRESHOLD</small><strong>{threshold ? percent(threshold.frozen_threshold) : '43.5%'}</strong><span>Selected on held-out fold {threshold?.calibration_fold ?? 9}</span></div><div className="threshold-reason"><h3>Why this threshold?</h3><p>{threshold?.reasoning ?? 'Selected on a held-out calibration cohort to target approximately 90% specificity.'}</p><div><span><strong>{threshold ? percent(threshold.observed_specificity, 2) : '90.04%'}</strong> specificity on calibration fold</span><span><strong>{threshold ? percent(threshold.observed_sensitivity, 2) : '73.93%'}</strong> sensitivity on calibration fold</span></div></div><div className="threshold-caution"><strong>Why not 50%?</strong><p>Probability calibration and clinical action are separate decisions. The cut-off is chosen for the intended false-alert trade-off; 50% has no automatic clinical privilege.</p></div></div>
      <div className="confusion-section"><div className="subsection-heading"><div><span className="eyebrow">Hybrid fusion · development operating point</span><h3>What each outcome means</h3></div><p>{cm?.scope}</p></div><div className="confusion-grid"><ConfusionTile kind="TP" label="MI pattern correctly flagged" cell={cm?.true_positives} tone="good" /><ConfusionTile kind="FN" label="MI pattern missed by the screen" cell={cm?.false_negatives} tone="warning" /><ConfusionTile kind="FP" label="Non-MI ECG incorrectly flagged" cell={cm?.false_positives} tone="warning" /><ConfusionTile kind="TN" label="Non-MI ECG correctly cleared" cell={cm?.true_negatives} tone="good" /></div><div className="derived-metrics"><article><span>Positive predictive value</span><strong>{cm ? percent(cm.positive_predictive_value, 2) : '72.36%'}</strong><small>Of positive screens, the share that were MI-pattern ECGs in this dataset</small></article><article><span>Negative predictive value</span><strong>{cm ? percent(cm.negative_predictive_value, 2) : '92.33%'}</strong><small>Of negative screens, the share that were non-MI ECGs in this dataset</small></article><article><span>Accuracy</span><strong>{cm ? percent(cm.accuracy, 2) : '86.93%'}</strong><small>All correct screens divided by all evaluated ECGs</small></article></div></div>
      <div className="benchmark-card"><div className="benchmark-table"><div className="table-row table-head"><span>System</span><span>AUPRC</span><span>AUROC</span><span>Brier ↓</span><span>Sensitivity</span></div>{rows.map((row) => <div className={`table-row ${row.name === 'Hybrid fusion' ? 'selected' : ''}`} key={row.name}><span><i className={row.name === 'Hybrid fusion' ? 'hybrid-dot' : ''} />{row.name}</span><strong>{row.auprc.toFixed(4)}</strong><span>{row.auroc.toFixed(4)}</span><span>{row.brier.toFixed(4)}</span><span>{percent(row.sensitivity_at_90_specificity, 2)}</span></div>)}</div><div className="evidence-note"><strong>Evidence boundary</strong><p>{benchmarks?.interpretation ?? 'Loading comparison…'}</p><span>No quantum advantage is claimed</span></div></div>
      <div className="model-evidence"><article><small>TASK</small><p>{modelCard?.task ?? 'MI-pattern versus non-MI-pattern classification'}</p></article><article><small>DATA</small><p>PTB-XL v1.0.3 · PTB-XL+ reference measurements</p></article><article><small>CLAIM BOUNDARY</small><p>{modelCard?.claim_boundary ?? 'Research screening only.'}</p></article></div>
    </section>
  )
}

function App() {
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [architecture, setArchitecture] = useState<Architecture | null>(null)
  const [benchmarks, setBenchmarks] = useState<Benchmarks | null>(null)
  const [modelCard, setModelCard] = useState<ModelCard | null>(null)
  useEffect(() => { void Promise.all([getReadiness(), getArchitecture(), getBenchmarks(), getModelCard()]).then(([ready, arch, metrics, card]) => { setReadiness(ready); setArchitecture(arch); setBenchmarks(metrics); setModelCard(card) }) }, [])
  return <main><header className="site-header"><a className="brand" href="#top"><span className="brand-mark">A</span><span>AQUIRE<em>MED</em></span></a><nav><a href="#review">Review ECG</a><a href="#how-it-works">How it works</a><a href="#evidence">Evidence</a></nav><div className={`api-state ${readiness?.model_ready ? 'ready' : 'pending'}`}><i />{readiness?.model_ready ? 'Screening service ready' : 'Service unavailable'}</div></header><ClinicianHero /><UploadPanel readiness={readiness} /><ArchitecturePanel architecture={architecture} /><EvidencePanel benchmarks={benchmarks} modelCard={modelCard} /><footer><div className="brand"><span className="brand-mark">A</span><span>AQUIRE<em>MED</em></span></div><p>Research prototype · MI-pattern ECG decision support</p></footer></main>
}

export default App
