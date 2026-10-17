// TypeScript definitions for Transformer Explorer API
// By Hammad, AI Engineer

export type DataOriginType =
  | 'Real model output'
  | 'Educational explanation'
  | 'Illustrative diagram'
  | 'Gemini-written explanation';

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string;
  device: string;
  error_message?: string | null;
  setup_guidance?: string | null;
  data_origin: DataOriginType;
}

export interface ModelInfoResponse {
  model_name: string;
  architecture: string;
  num_layers: number;
  num_heads: number;
  hidden_size: number;
  vocab_size: number;
  max_position_embeddings: number;
  num_parameters: number;
  device: string;
  attn_implementation: string;
  trust_remote_code: boolean;
  data_origin: DataOriginType;
}

export interface TokenItem {
  index: number;
  token_id: number;
  token_str: string;
  token_bytes_display: string;
}

export interface TokenizeResponse {
  text: string;
  token_count: number;
  token_ids: number[];
  tokens: TokenItem[];
  input_tensor_shape: number[];
  data_origin: DataOriginType;
}

export interface VectorStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  norm: number;
}

export interface TokenEmbeddingItem {
  index: number;
  token_id: number;
  token_str: string;
  token_embedding_slice: number[];
  token_stats: VectorStats;
  positional_embedding_slice: number[];
  positional_stats: VectorStats;
  combined_embedding_slice: number[];
  combined_stats: VectorStats;
}

export interface EmbeddingResponse {
  text: string;
  token_count: number;
  hidden_dimension: number;
  slice_size: number;
  token_embedding_shape: number[];
  positional_embedding_shape: number[];
  combined_embedding_shape: number[];
  tokens: TokenEmbeddingItem[];
  explanation: string;
  explanation_origin?: DataOriginType;
  data_origin: DataOriginType;
}

export interface CandidateToken {
  rank: number;
  token_id: number;
  token_str: string;
  token_bytes_display: string;
  logit: number;
  probability: number;
  cumulative_probability: number;
}

export interface PredictNextTokenResponse {
  prompt: string;
  prompt_token_ids: number[];
  prompt_tokens: string[];
  vocab_size: number;
  temperature_used: number;
  top_k_used: number;
  temperature_explanation: string;
  top_candidates: CandidateToken[];
  top_1_token_id: number;
  top_1_token_str: string;
  total_probability_top_k: number;
  explanation?: string;
  explanation_origin?: DataOriginType;
  data_origin: DataOriginType;
}

export interface GeneratedTokenItem {
  step: number;
  token_id: number;
  token_str: string;
}

export interface GenerateResponse {
  prompt: string;
  prompt_token_count: number;
  generated_text: string;
  continuation: string;
  new_token_ids: number[];
  new_tokens: GeneratedTokenItem[];
  new_token_count: number;
  finish_reason: string;
  settings: Record<string, unknown>;
  data_origin: DataOriginType;
}

export interface AttentionWeightItem {
  source_index: number;
  source_token: string;
  target_index: number;
  target_token: string;
  weight: number;
}

export interface QKVExtractionSummary {
  layer: number;
  head: number;
  head_dim: number;
  fused_c_attn_width: number;
  q_shape: number[];
  k_shape: number[];
  v_shape: number[];
  q_slice: number[];
  k_slice: number[];
  v_slice: number[];
  max_recomputed_difference: number;
}

export interface AttentionResponse {
  text: string;
  layer: number;
  head: number;
  token_count: number;
  tokens: string[];
  matrix: number[][];
  top_weights: AttentionWeightItem[];
  is_causal_mask_applied: boolean;
  upper_triangle_zero: boolean;
  qkv: QKVExtractionSummary;
  explanation: string;
  explanation_origin?: DataOriginType;
  data_origin: DataOriginType;
}

export interface LayerActivationComponent {
  name: string;
  description: string;
  shape: number[];
  slice_values: number[];
  stats: VectorStats;
  is_captured: boolean;
}

export interface LayerInspectionResponse {
  text: string;
  layer: number;
  token_index: number;
  token_str: string;
  num_layers: number;
  num_heads: number;
  hidden_size: number;
  token_count: number;
  tokens: string[];
  components: LayerActivationComponent[];
  final_hidden_state_shape: number[];
  logits_shape: number[];
  explanation: string;
  explanation_origin?: DataOriginType;
  data_origin: DataOriginType;
}

export interface TutorStatusResponse {
  configured: boolean;
  model: string;
  status_message: string;
  data_origin: DataOriginType;
}

export interface TutorExplainResponse {
  explanation: string;
  model: string;
  data_origin: DataOriginType;
}

export interface TutorAskResponse {
  answer: string;
  model: string;
  data_origin: DataOriginType;
}

export interface QuizQuestionItem {
  id: number;
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
}

export interface TutorQuizResponse {
  topic: string;
  questions: QuizQuestionItem[];
  model: string;
  data_origin: DataOriginType;
}

export interface TutorCompareGenerationResponse {
  prompt: string;
  gemini_continuation: string;
  model_name: string;
  disclaimer: string;
  data_origin: DataOriginType;
}

export interface RequestLogEntry {
  id: number;
  timestamp: string;
  endpoint: string;
  method: string;
  status_code: number;
  duration_ms: number;
  model_name: string;
  device: string;
  input_token_ids: number[];
  tensor_shapes: Record<string, number[]>;
  tensor_dtypes: Record<string, string>;
  layer_selected?: number | null;
  head_selected?: number | null;
  generation_settings?: Record<string, unknown> | null;
  gemini_status?: string | null;
  warnings: string[];
  errors: string[];
}

export interface DebugLogsResponse {
  total_captured: number;
  capacity: number;
  logs: RequestLogEntry[];
  data_origin: DataOriginType;
}

export interface DebugInfoResponse {
  model_name: string;
  model_revision: string;
  device: string;
  dtype: string;
  attn_implementation: string;
  tokenizer_class: string;
  vocab_size: number;
  torch_version: string;
  transformers_version: string;
  data_origin: DataOriginType;
}

export interface ModelComparisonStats {
  model_name: string;
  param_count: number;
  param_count_display: string;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
  vocab_size: number;
  tokens: string[];
  token_ids: number[];
  top_candidates: CandidateToken[];
  top_1_token_id: number;
  top_1_token_str: string;
  top_1_probability: number;
  total_top_k_prob: number;
}

export interface CompareModelsResponse {
  text: string;
  model_a: ModelComparisonStats;
  model_b: ModelComparisonStats;
  differences: Record<string, unknown>;
  explanation: string;
  explanation_origin?: DataOriginType;
  data_origin: DataOriginType;
}

export interface LogitLensCandidate {
  token_id: number;
  token_str: string;
  probability: number;
  logit: number;
  rank: number;
}

export interface LogitLensLayerStep {
  layer: number;
  layer_name: string;
  top_candidates: LogitLensCandidate[];
  target_rank: number;
  target_probability: number;
  entropy: number;
  cosine_sim_to_previous: number | null;
  residual_norm: number;
}

export interface LogitLensResponse {
  prompt: string;
  token_count: number;
  selected_token_index: number;
  selected_token_str: string;
  final_top_token_id: number;
  final_top_token_str: string;
  layers: LogitLensLayerStep[];
  explanation: string;
  explanation_origin?: DataOriginType;
  data_origin: DataOriginType;
}

