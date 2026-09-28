import { useEffect, useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-basic-dist-min'
import { getArchitecture, getBenchmarks, getModelCard, getReadiness, inspectEcg, predictEcg } from './api'
import type { Architecture, Benchmarks, Inspection, ModelCard, Prediction, Readiness } from './types'
import './App.css'

const LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
const Plot = createPlotlyComponent(Plotly)

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  )
}

function ArchitecturePanel({ architecture }: { architecture: Architecture | null }) {
  const active = architecture?.routes
  return (
    <section id="architecture" className="section architecture-section">
      <div className="section-heading">
        <span className="eyebrow">Fixed hybrid architecture</span>
        <h2>Both predictive routes run for every accepted ECG.</h2>
        <p>No model-selection gate, residual target, or patient-level route switching is used.</p>
      </div>
      <div className="flow">
        <div className="flow-node common-node">
          <small>COMMON INPUT</small>
          <strong>12-lead ECG</strong>
          <span>Validation · QC · minimal signal</span>
        </div>
        <div className="route-split" aria-hidden="true"><span /><span /></div>
        <div className="route-grid">
          <article className="route-card quantum-card">
            <div className="route-label"><i /> Quantum route {active?.quantum.active ? 'active' : ''}</div>
            <h3>Waveform representation</h3>
            <ol>
              <li>Compact Transformer → h128</li>
              <li>Robust scaling → PLS-q4</li>
              <li>Angle encoding → 4-qubit VQC</li>
            </ol>
            <code>quantum score sQ</code>
          </article>
          <article className="route-card classical-card">
            <div className="route-label"><i /> Classical route {active?.classical.active ? 'active' : ''}</div>
            <h3>Clinical morphology</h3>
            <ol>
              <li>106 deployable measurements</li>
              <li>Frozen feature conditioning</li>
              <li>HistGradientBoosting</li>
            </ol>
            <code>classical score sC</code>
          </article>
        </div>
        <div className="fusion-node">
          <small>ONE PATIENT-FACING OUTPUT</small>
          <strong>σ(β₀ + βQ·sQ + βC·sC)</strong>
          <span>Calibration → MI-pattern screening result</span>
        </div>
      </div>
    </section>
  )
}

function UploadPanel({ readiness }: { readiness: Readiness | null }) {
  const [file, setFile] = useState<File | null>(null)
  const [inspection, setInspection] = useState<Inspection | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')

  async function inspect() {
    if (!file) return
    setBusy(true)
    setMessage('')
    try {
      setInspection(await inspectEcg(file))
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Inspection failed')
    } finally {
      setBusy(false)
    }
  }

  async function predict() {
    if (!file) return
    setBusy(true)
    setMessage('')
    try {
      setPrediction(await predictEcg(file))
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Prediction failed')
    } finally {
      setBusy(false)
    }
  }

  const traces = useMemo(() => {
    if (!inspection?.valid || !inspection.input) return []
    return LEAD_ORDER.map((lead, index) => ({
      x: inspection.input!.preview[lead].map((_, sample) => sample / 20),
      y: inspection.input!.preview[lead].map((value) => value + (11 - index) * 3),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: lead,
      line: { color: index < 6 ? '#db3948' : '#6c48c7', width: 1.3 },
      hovertemplate: `${lead} · %{x:.2f}s · %{y:.3f}<extra></extra>`,
    }))
  }, [inspection])

  return (
    <section id="analyze" className="section analysis-section">
      <div className="section-heading narrow">
        <span className="eyebrow">Analyze one ECG</span>
        <h2>Inspect the signal before inference.</h2>
        <p>Upload a 10-second WFDB ZIP, canonical 12-column CSV, or JSON array. Files stay inside the API process.</p>
      </div>
      <div className="analysis-grid">
        <div className="upload-card">
          <label className="drop-zone">
            <input
              type="file"
              accept=".csv,.json,.zip"
              onChange={(event) => {
                setFile(event.target.files?.[0] ?? null)
                setInspection(null)
                setPrediction(null)
                setMessage('')
              }}
            />
            <span className="upload-icon">↥</span>
            <strong>{file ? file.name : 'Choose an ECG file'}</strong>
            <small>CSV · JSON · WFDB ZIP · maximum 5 MB</small>
          </label>
          <div className="button-row">
            <button className="button secondary" disabled={!file || busy} onClick={inspect}>
              {busy ? 'Checking…' : 'Inspect ECG'}
            </button>
            <button className="button primary" disabled={!inspection?.valid || busy || !readiness?.model_ready} onClick={predict}>
              Run hybrid model
            </button>
          </div>
          {!readiness?.model_ready && (
            <div className="notice amber">
              <strong>Model bundle pending</strong>
              <span>The API is correctly blocking predictions until the checksummed dual-route bundle is mounted.</span>
            </div>
          )}
          {inspection && (
            <div className={`notice ${inspection.valid ? 'green' : 'red'}`}>
              <strong>{inspection.valid ? 'Structure accepted' : 'Input rejected'}</strong>
              <span>{inspection.valid ? '12 leads · 1,000 samples · 100 Hz' : inspection.errors.join(' · ')}</span>
            </div>
          )}
          {message && <div className="notice amber"><span>{message}</span></div>}
        </div>
        <div className="waveform-card">
          <div className="card-title">
            <div><span className="live-dot" /> Signal preview</div>
            <small>{inspection?.input?.signal_sha256.slice(0, 12) ?? 'waiting for ECG'}</small>
          </div>
          {traces.length ? (
            <Plot
              data={traces}
              layout={{
                autosize: true,
                height: 430,
                margin: { l: 30, r: 10, t: 10, b: 32 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: '#fffafa',
                showlegend: false,
                xaxis: { title: { text: 'Seconds' }, gridcolor: '#f4d9dc', zeroline: false },
                yaxis: { showticklabels: false, gridcolor: '#f8e8ea', zeroline: false },
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: '100%' }}
            />
          ) : (
            <div className="empty-waveform">
              <div className="pulse-line" />
              <p>A validated ECG preview will appear here.</p>
            </div>
          )}
        </div>
      </div>
      {prediction && (
        <article className="result-card" aria-live="polite">
          <div className="result-primary">
            <span className="eyebrow">Fused screening output</span>
            <div
              className="probability-ring"
              style={{ '--score': `${prediction.mi_pattern_probability * 360}deg` } as CSSProperties}
            >
              <div>
                <strong>{(prediction.mi_pattern_probability * 100).toFixed(1)}%</strong>
                <span>MI pattern</span>
              </div>
            </div>
            <h3>{prediction.prediction === 'MI_PATTERN' ? 'MI pattern detected' : 'No MI pattern detected'}</h3>
            <p>Threshold {(prediction.decision_threshold * 100).toFixed(1)}% · clinician review required</p>
          </div>
          <div className="result-explanation">
            <small>ONE HYBRID RESULT</small>
            <h3>Both routes contributed before calibration.</h3>
            <p>{prediction.interpretation}</p>
            <div className="route-score-grid">
              <div><span>Quantum route · VQC ensemble</span><strong>{prediction.routes.quantum.score.toFixed(4)}</strong></div>
              <div><span>Classical route · morphology HGB</span><strong>{prediction.routes.classical.score.toFixed(4)}</strong></div>
              <div><span>Fusion logit</span><strong>{prediction.fusion.raw_logit.toFixed(4)}</strong></div>
              <div><span>Model</span><strong>{prediction.model_version}</strong></div>
            </div>
            <p className="technical-note">Route scores are technical evidence, not separate diagnoses. There is no gate or fallback route.</p>
          </div>
        </article>
      )}
    </section>
  )
}

function BenchmarkPanel({ benchmarks }: { benchmarks: Benchmarks | null }) {
  const rows = benchmarks?.stabilized_two_seed ?? []
  return (
    <section id="evidence" className="section evidence-section">
      <div className="section-heading">
        <span className="eyebrow">Development evidence</span>
        <h2>Competitive hybrid screening, with an honest comparator.</h2>
        <p>Patient-separated out-of-fold evidence from PTB-XL folds 1–8. These are research results, not clinical deployment claims.</p>
      </div>
      <div className="metrics-row">
        <Metric label="Development ECGs" value="17,348" detail="14,958 patients" />
        <Metric label="Hybrid AUPRC" value="0.8360" detail="two-seed stabilized screen" />
        <Metric label="Sensitivity" value="77.79%" detail="at ≈90% specificity" />
        <Metric label="Quantum width" value="4 qubits" detail="two re-uploading blocks" />
      </div>
      <div className="benchmark-card">
        <div className="benchmark-table">
          <div className="table-row table-head"><span>System</span><span>AUPRC</span><span>AUROC</span><span>Brier ↓</span><span>Sensitivity</span></div>
          {rows.map((row) => (
            <div className={`table-row ${row.name === 'Hybrid fusion' ? 'selected' : ''}`} key={row.name}>
              <span><i className={row.name === 'Hybrid fusion' ? 'hybrid-dot' : ''} />{row.name}</span>
              <strong>{row.auprc.toFixed(4)}</strong>
              <span>{row.auroc.toFixed(4)}</span>
              <span>{row.brier.toFixed(4)}</span>
              <span>{(row.sensitivity_at_90_specificity * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
        <div className="evidence-note">
          <strong>What the result supports</strong>
          <p>{benchmarks?.interpretation ?? 'Loading benchmark interpretation…'}</p>
          <span>No quantum advantage is claimed.</span>
        </div>
      </div>
    </section>
  )
}

function App() {
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [architecture, setArchitecture] = useState<Architecture | null>(null)
  const [benchmarks, setBenchmarks] = useState<Benchmarks | null>(null)
  const [modelCard, setModelCard] = useState<ModelCard | null>(null)

  useEffect(() => {
    void Promise.all([getReadiness(), getArchitecture(), getBenchmarks(), getModelCard()]).then(
      ([ready, arch, metrics, card]) => {
        setReadiness(ready)
        setArchitecture(arch)
        setBenchmarks(metrics)
        setModelCard(card)
      },
    )
  }, [])

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top"><span className="brand-mark">A</span><span>AQUIRE<em>MED</em></span></a>
        <nav><a href="#architecture">Architecture</a><a href="#analyze">Analyze ECG</a><a href="#evidence">Evidence</a></nav>
        <div className={`api-state ${readiness?.model_ready ? 'ready' : 'pending'}`}><i />{readiness?.model_ready ? 'Model ready' : 'Bundle pending'}</div>
      </header>

      <section id="top" className="hero-section">
        <div className="hero-copy">
          <div className="hero-kicker"><span>SIH 26139</span> Hybrid quantum machine learning</div>
          <h1>Two predictive routes.<br /><em>One accountable result.</em></h1>
          <p>A fixed quantum–classical system for MI-pattern screening from a ten-second, twelve-lead ECG.</p>
          <div className="hero-actions"><a className="button primary" href="#analyze">Analyze an ECG</a><a className="button ghost" href="#architecture">See how it works</a></div>
          <div className="boundary"><strong>Research prototype</strong><span>Not an acute-MI diagnosis or future-event prediction.</span></div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="orb quantum-orb"><small>QUANTUM</small><strong>q4 VQC</strong><span>Waveform patterns</span></div>
          <div className="orb classical-orb"><small>CLASSICAL</small><strong>106 features</strong><span>Clinical morphology</span></div>
          <div className="fusion-badge"><small>FUSED</small><strong>P(MI pattern)</strong></div>
          <svg viewBox="0 0 600 400"><path d="M140 150 C250 150, 235 205, 300 220" /><path d="M460 150 C350 150, 365 205, 300 220" /><path d="M300 250 L300 310" /></svg>
        </div>
      </section>

      <ArchitecturePanel architecture={architecture} />
      <UploadPanel readiness={readiness} />
      <BenchmarkPanel benchmarks={benchmarks} />

      <section className="section model-card-section">
        <div><span className="eyebrow">Model card</span><h2>Designed for scrutiny.</h2></div>
        <div className="model-card-grid">
          <article><small>TASK</small><p>{modelCard?.task ?? 'MI-pattern versus non-MI-pattern classification'}</p></article>
          <article><small>DATA</small><p>PTB-XL v1.0.3 · PTB-XL+ reference measurements</p></article>
          <article><small>BOUNDARY</small><p>{modelCard?.claim_boundary ?? 'No clinical or quantum-advantage claim.'}</p></article>
        </div>
      </section>

      <footer><div className="brand"><span className="brand-mark">A</span><span>AQUIRE<em>MED</em></span></div><p>Smart India Hackathon research prototype · fixed hybrid fusion</p></footer>
    </main>
  )
}

export default App
