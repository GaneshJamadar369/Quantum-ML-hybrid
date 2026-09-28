import type { Architecture, Benchmarks, Inspection, ModelCard, Prediction, Readiness } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.detail?.message ?? payload.detail?.detail ?? `API request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export async function getReadiness(): Promise<Readiness> {
  try {
    return await getJson<Readiness>('/api/v1/health/ready')
  } catch (error) {
    return { status: 'not_ready', model_ready: false, detail: error instanceof Error ? error.message : 'API unavailable' }
  }
}

export const getArchitecture = () => getJson<Architecture>('/api/v1/architecture')
export const getBenchmarks = () => getJson<Benchmarks>('/api/v1/benchmarks')
export const getModelCard = () => getJson<ModelCard>('/api/v1/model-card')

async function postFile<T>(path: string, file: File): Promise<T> {
  const body = new FormData()
  body.append('file', file)
  const response = await fetch(`${API_BASE}${path}`, { method: 'POST', body })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload.detail?.message ?? payload.detail?.detail ?? `Request failed (${response.status})`)
  }
  return payload as T
}

export const inspectEcg = (file: File) => postFile<Inspection>('/api/v1/ecg/inspect', file)
export const predictEcg = (file: File) => postFile<Prediction>('/api/v1/predictions', file)
