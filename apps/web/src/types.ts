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

export interface ConfusionCell {
  count: number
  rate: number
  denominator: string
}

export interface Benchmarks {
  scope: string
  stabilized_two_seed: BenchmarkRow[]
  interpretation: string
  threshold_selection: {
    frozen_threshold: number
    calibration_fold: number
    target_specificity: number
    observed_specificity: number
    observed_sensitivity: number
    reasoning: string
  }
  confusion_matrix: {
    scope: string
    true_positives: ConfusionCell
    false_negatives: ConfusionCell
    false_positives: ConfusionCell
    true_negatives: ConfusionCell
    positive_predictive_value: number
    negative_predictive_value: number
    accuracy: number
  }
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

export interface TerritoryFinding {
  name: string
  leads: string
  st_mean_mv: number | null
  st_absmax_mv: number | null
  st_abnormal_leads: number | null
  t_inversion_fraction: number | null
}

export interface SignalCharacteristics {
  quality: { state: string; failed_leads: number; issues: string[]; finite_fraction: number }
  rhythm: {
    heart_rate_bpm: number | null
    heart_rate_context: string
    rr_median_ms: number | null
    rr_iqr_ms: number | null
    rr_cv: number | null
  }
  st_t: {
    global_st_rms_mv: number | null
    st_positive_leads: number | null
    st_negative_leads: number | null
    t_inversion_leads: number | null
    territories: TerritoryFinding[]
  }
  spatial: { frontal_axis_proxy_deg: number | null; precordial_transition: string | null }
  amplitude: { minimum_mv: number; maximum_mv: number; peak_to_peak_mv: number }
  availability: {
    measurements_available: number
    extractor_notes: string[]
    intervals_omitted: string[]
    interval_reason: string
  }
}

export interface Prediction {
  prediction: 'MI_PATTERN' | 'NON_MI_PATTERN'
  mi_pattern_probability: number
  decision_threshold: number
  routes: {
    quantum: { active: true; score: number }
    classical: { active: true; score: number }
  }
  fusion: { active: true; raw_logit: number; uncalibrated_probability: number }
  model_version: string
  signal_sha256: string
  signal_characteristics: SignalCharacteristics
  interpretation: string
}
