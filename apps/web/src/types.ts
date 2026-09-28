export interface Readiness {
  status: 'ok' | 'not_ready'
  model_ready: boolean
  model_version?: string
  detail?: string
}

export interface Architecture {
  routing: string
  routes: {
    quantum: { active: boolean; steps: string[]; output: string }
    classical: { active: boolean; steps: string[]; output: string }
    fusion: { active: boolean; formula: string; output: string }
  }
}

export interface BenchmarkRow {
  name: string
  auprc: number
  auroc: number
  brier: number
  sensitivity_at_90_specificity: number
}

export interface Benchmarks {
  scope: string
  stabilized_two_seed: BenchmarkRow[]
  interpretation: string
}

export interface ModelCard {
  name: string
  task: string
  claim_boundary: string
}

export interface Inspection {
  valid: boolean
  input?: {
    format: 'csv' | 'json' | 'wfdb_zip'
    sampling_rate_hz: number
    shape: [number, number]
    lead_order: string[]
    signal_sha256: string
    preview: Record<string, number[]>
  }
  errors: string[]
  warnings: string[]
}
