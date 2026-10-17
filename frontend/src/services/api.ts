// Typed API Client for Transformer Explorer
// By Hammad, AI Engineer

import type {
  HealthResponse,
  ModelInfoResponse,
  TokenizeResponse,
  EmbeddingResponse,
  PredictNextTokenResponse,
  GenerateResponse,
  AttentionResponse,
  LayerInspectionResponse,
  TutorStatusResponse,
  TutorExplainResponse,
  TutorAskResponse,
  TutorQuizResponse,
  TutorCompareGenerationResponse,
  DebugLogsResponse,
  DebugInfoResponse,
  CompareModelsResponse,
} from '../types/api';

const API_BASE = '/api';

export class ApiError extends Error {
  status?: number;
  setupGuidance?: string;

  constructor(message: string, status?: number, setupGuidance?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.setupGuidance = setupGuidance;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });

    if (!response.ok) {
      let errorDetail = `Server returned HTTP ${response.status}`;
      let setupGuidance = '';
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          if (typeof errorJson.detail === 'string') {
            errorDetail = errorJson.detail;
          } else if (typeof errorJson.detail === 'object') {
            errorDetail = errorJson.detail.message || errorJson.detail.error || JSON.stringify(errorJson.detail);
            setupGuidance = errorJson.detail.setup_guidance || '';
          }
        }
      } catch {
        // Fallback to text if JSON parse fails
      }
      throw new ApiError(errorDetail, response.status, setupGuidance);
    }

    return (await response.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    // Network failure (server down)
    throw new ApiError(
      'Cannot connect to the FastAPI backend at http://127.0.0.1:8000. ' +
      'Please ensure the backend is running via PowerShell: ' +
      '.\\venv\\Scripts\\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000',
      0
    );
  }
}

export const api = {
  getHealth: () => request<HealthResponse>('/health'),
  getModelInfo: () => request<ModelInfoResponse>('/model-info'),
  tokenize: (text: string) =>
    request<TokenizeResponse>('/tokenize', {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),
  getEmbeddings: (text: string, slice_size: number = 16) =>
    request<EmbeddingResponse>('/embeddings', {
      method: 'POST',
      body: JSON.stringify({ text, slice_size }),
    }),
  predictNextToken: (text: string, temperature: number = 1.0, top_k: number = 10) =>
    request<PredictNextTokenResponse>('/predict-next-token', {
      method: 'POST',
      body: JSON.stringify({ text, temperature, top_k }),
    }),
  generate: (
    prompt: string,
    max_new_tokens: number = 20,
    temperature: number = 0.7,
    top_k: number = 50,
    seed?: number
  ) =>
    request<GenerateResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify({ prompt, max_new_tokens, temperature, top_k, seed }),
    }),
  getAttention: (text: string, layer: number = 0, head: number = 0) =>
    request<AttentionResponse>('/attention', {
      method: 'POST',
      body: JSON.stringify({ text, layer, head }),
    }),
  inspectLayer: (text: string, layer: number = 0, slice_size: number = 16, token_index: number = 0) =>
    request<LayerInspectionResponse>('/layer-inspection', {
      method: 'POST',
      body: JSON.stringify({ text, layer, slice_size, token_index }),
    }),
  getTutorStatus: () => request<TutorStatusResponse>('/tutor/status'),
  explainView: (view_name: string, context_summary: Record<string, unknown>) =>
    request<TutorExplainResponse>('/tutor/explain', {
      method: 'POST',
      body: JSON.stringify({ view_name, context_summary }),
    }),
  askTutor: (question: string, context_summary?: Record<string, unknown>) =>
    request<TutorAskResponse>('/tutor/ask', {
      method: 'POST',
      body: JSON.stringify({ question, context_summary }),
    }),
  getTutorQuiz: (topic: string = 'general') =>
    request<TutorQuizResponse>('/tutor/quiz', {
      method: 'POST',
      body: JSON.stringify({ topic }),
    }),
  compareGeneration: (prompt: string, local_generation?: string) =>
    request<TutorCompareGenerationResponse>('/tutor/compare-generation', {
      method: 'POST',
      body: JSON.stringify({ prompt, local_generation }),
    }),
  getDebugLogs: () => request<DebugLogsResponse>('/debug/logs'),
  clearDebugLogs: () =>
    request<{ status: string; total_captured: number }>('/debug/clear', {
      method: 'POST',
    }),
  getDebugInfo: () => request<DebugInfoResponse>('/debug/info'),
  compareModels: (text: string, model_a: string = 'distilgpt2', model_b: string = 'gpt2') =>
    request<CompareModelsResponse>('/compare-models', {
      method: 'POST',
      body: JSON.stringify({ text, model_a, model_b }),
    }),
};
