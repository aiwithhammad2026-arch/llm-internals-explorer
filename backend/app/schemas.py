"""Pydantic schemas for request and response validation.
By Hammad, AI Engineer.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator

# Ground-truth origin labels as specified in Section 1
DataOriginType = Literal[
    "Real model output",
    "Educational explanation",
    "Illustrative diagram",
    "Gemini-written explanation"
]

# =====================================================================
# Phase 1 Schemas: Health, Model Info, Tokenization
# =====================================================================

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status: healthy, loading, or degraded")
    model_loaded: bool = Field(..., description="Whether the Transformer model is ready in memory")
    model_name: str = Field(..., description="Pretrained model identifier")
    device: str = Field(..., description="Device model is currently residing on (cpu/cuda)")
    error_message: Optional[str] = Field(None, description="Detailed error message if loading failed")
    setup_guidance: Optional[str] = Field(None, description="Actionable recovery steps for the user")
    data_origin: DataOriginType = "Real model output"


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    num_layers: int = Field(..., description="Number of Transformer blocks (e.g. 6 in distilgpt2)")
    num_heads: int = Field(..., description="Attention heads per layer (e.g. 12 in distilgpt2)")
    hidden_size: int = Field(..., description="Embedding and hidden dimension (e.g. 768 in distilgpt2)")
    vocab_size: int = Field(..., description="Vocabulary size (e.g. 50257 in GPT-2)")
    max_position_embeddings: int = Field(..., description="Context window size (e.g. 1024)")
    num_parameters: int = Field(..., description="Total trainable parameter count")
    device: str
    attn_implementation: str = Field(..., description="Must be 'eager' for real attention extraction")
    trust_remote_code: bool = Field(False, description="Strict security enforcement flag")
    data_origin: DataOriginType = "Real model output"


class TokenItem(BaseModel):
    index: int = Field(..., description="0-based token index in sequence")
    token_id: int = Field(..., description="Vocabulary token integer ID from BPE tokenizer")
    token_str: str = Field(..., description="Decoded token string representation")
    token_bytes_display: str = Field(..., description="Raw byte / unicode representation with GPT-2 space marker")


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Text to tokenize")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class TokenizeResponse(BaseModel):
    text: str
    token_count: int
    token_ids: List[int]
    tokens: List[TokenItem]
    input_tensor_shape: List[int] = Field(..., description="Shape of token tensor: [batch_size, seq_len]")
    data_origin: DataOriginType = "Real model output"


# =====================================================================
# Phase 2 Schemas: Embeddings
# =====================================================================

class VectorStats(BaseModel):
    min: float = Field(..., description="Minimum value across vector dimensions")
    max: float = Field(..., description="Maximum value across vector dimensions")
    mean: float = Field(..., description="Mean value across vector dimensions")
    std: float = Field(..., description="Standard deviation across vector dimensions")
    norm: float = Field(..., description="L2 Euclidean norm of the vector")


class TokenEmbeddingItem(BaseModel):
    index: int = Field(..., description="Token sequence position (0 to seq_len-1)")
    token_id: int = Field(..., description="BPE token ID")
    token_str: str = Field(..., description="Decoded token string")
    token_embedding_slice: List[float] = Field(..., description="First N dimensions of Word Token Embedding (wte)")
    token_stats: VectorStats = Field(..., description="Summary statistics across all 768 dimensions of wte")
    positional_embedding_slice: List[float] = Field(..., description="First N dimensions of Word Positional Embedding (wpe)")
    positional_stats: VectorStats = Field(..., description="Summary statistics across all 768 dimensions of wpe")
    combined_embedding_slice: List[float] = Field(..., description="First N dimensions of wte + wpe (input to Block 0)")
    combined_stats: VectorStats = Field(..., description="Summary statistics across all 768 dimensions of wte + wpe")


class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text to embed")
    slice_size: int = Field(16, ge=4, le=64, description="Number of vector dimensions to return in slice")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class EmbeddingResponse(BaseModel):
    text: str
    token_count: int
    hidden_dimension: int = Field(..., description="Full embedding dimension (768 for distilgpt2)")
    slice_size: int = Field(..., description="Dimension count returned in vector slices")
    token_embedding_shape: List[int] = Field(..., description="Shape of wte tensor: [1, seq_len, 768]")
    positional_embedding_shape: List[int] = Field(..., description="Shape of wpe tensor: [1, seq_len, 768]")
    combined_embedding_shape: List[int] = Field(..., description="Shape of combined tensor: [1, seq_len, 768]")
    tokens: List[TokenEmbeddingItem]
    explanation: str = Field(
        ...,
        description="Educational explanation of learned token vs positional embeddings and their addition"
    )
    explanation_origin: DataOriginType = "Educational explanation"
    data_origin: DataOriginType = "Real model output"


# =====================================================================
# Phase 2 Schemas: Next-Token Prediction
# =====================================================================

class CandidateToken(BaseModel):
    rank: int = Field(..., description="Candidate rank (1 to top_k)")
    token_id: int = Field(..., description="Vocabulary token ID")
    token_str: str = Field(..., description="Decoded string token")
    token_bytes_display: str = Field(..., description="Byte display with GPT-2 space representation")
    logit: float = Field(..., description="Raw unscaled logit from the language modeling head")
    probability: float = Field(..., description="Normalized softmax probability under selected temperature")
    cumulative_probability: float = Field(..., description="Running sum of candidate probabilities")


class PredictNextTokenRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Prompt text")
    temperature: float = Field(1.0, gt=0.0, le=5.0, description="Temperature scaling factor (0.1 to 5.0)")
    top_k: int = Field(10, ge=1, le=100, description="Number of top candidates to inspect")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class PredictNextTokenResponse(BaseModel):
    prompt: str
    prompt_token_ids: List[int]
    prompt_tokens: List[str]
    vocab_size: int
    temperature_used: float
    top_k_used: int
    temperature_explanation: str = Field(
        ...,
        description="Explains that temperature scales the logits before softmax and does NOT retrain the model"
    )
    explanation: Optional[str] = Field(None, description="Educational explanation of next-token prediction and temperature")
    explanation_origin: DataOriginType = "Educational explanation"
    top_candidates: List[CandidateToken]
    top_1_token_id: int
    top_1_token_str: str
    total_probability_top_k: float = Field(..., description="Sum of probabilities among returned top-k candidates")
    data_origin: DataOriginType = "Real model output"


# =====================================================================
# Phase 2 Schemas: Text Generation
# =====================================================================

class GeneratedTokenItem(BaseModel):
    step: int = Field(..., description="Generation step (1 to max_new_tokens)")
    token_id: int = Field(..., description="Sampled token ID")
    token_str: str = Field(..., description="Sampled token decoded string")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512, description="Prompt text")
    max_new_tokens: int = Field(20, ge=1, le=128, description="Maximum number of new tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=5.0, description="Sampling temperature (0.0 = greedy argmax)")
    top_k: int = Field(50, ge=0, le=100, description="Top-k filtering threshold (0 = disable top-k)")
    seed: Optional[int] = Field(None, ge=0, le=2147483647, description="Optional random seed for reproducibility")

    @field_validator("prompt")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or only whitespace")
        return v


class GenerateResponse(BaseModel):
    prompt: str
    prompt_token_count: int
    generated_text: str = Field(..., description="Full text including prompt and newly generated continuation")
    continuation: str = Field(..., description="Newly generated continuation text only")
    new_token_ids: List[int] = Field(..., description="List of generated token IDs")
    new_tokens: List[GeneratedTokenItem] = Field(..., description="Detailed token objects with step index")
    new_token_count: int = Field(..., description="Number of newly generated tokens")
    finish_reason: str = Field(..., description="'max_tokens' or 'eos_token'")
    settings: Dict[str, Any] = Field(..., description="Sampling parameters used during generation")
    data_origin: DataOriginType = "Real model output"


# =====================================================================
# Phase 3 Schemas: Attention, Q/K/V, and Block Inspection
# =====================================================================

class AttentionWeightItem(BaseModel):
    source_index: int = Field(..., description="Query token index (attending token)")
    source_token: str = Field(..., description="Query token string representation")
    target_index: int = Field(..., description="Key token index (attended-to token)")
    target_token: str = Field(..., description="Key token string representation")
    weight: float = Field(..., description="Softmax attention probability")


class QKVExtractionSummary(BaseModel):
    layer: int
    head: int
    head_dim: int = Field(64, description="Hidden dimension per attention head (768 / 12 = 64)")
    fused_c_attn_width: int = Field(2304, description="Width of fused c_attn output (3 * 768)")
    q_shape: List[int] = Field(..., description="Shape of Q for this head: [seq_len, 64]")
    k_shape: List[int] = Field(..., description="Shape of K for this head: [seq_len, 64]")
    v_shape: List[int] = Field(..., description="Shape of V for this head: [seq_len, 64]")
    q_slice: List[float] = Field(..., description="Sample slice of Q vector for final token position")
    k_slice: List[float] = Field(..., description="Sample slice of K vector for final token position")
    v_slice: List[float] = Field(..., description="Sample slice of V vector for final token position")
    max_recomputed_difference: float = Field(
        ...,
        description="Max absolute error between real attention and softmax((Q @ K^T)/sqrt(64) + causal_mask)"
    )


class AttentionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text to analyze")
    layer: int = Field(0, ge=0, description="Transformer layer index (0 to num_layers-1)")
    head: int = Field(0, ge=0, description="Attention head index (0 to num_heads-1)")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class AttentionResponse(BaseModel):
    text: str
    layer: int
    head: int
    token_count: int
    tokens: List[str] = Field(..., description="Token sequence strings")
    matrix: List[List[float]] = Field(..., description="Full [seq_len, seq_len] attention matrix for selected head")
    top_weights: List[AttentionWeightItem] = Field(..., description="Top 15 attention weights sorted descending")
    is_causal_mask_applied: bool = Field(True, description="True if upper triangle is zeroed out by causal mask")
    upper_triangle_zero: bool = Field(True, description="Verified that no token attends to future tokens")
    qkv: QKVExtractionSummary = Field(..., description="Real Q/K/V vectors and mathematical verification")
    explanation: str = Field(..., description="Educational explanation of causal self-attention mechanism and limits")
    explanation_origin: DataOriginType = "Educational explanation"
    data_origin: DataOriginType = "Real model output"


class LayerActivationComponent(BaseModel):
    name: str = Field(..., description="Component identifier (e.g. input_hidden_state, ln_1, attn_proj, etc.)")
    description: str = Field(..., description="Function of this component in the Transformer block")
    shape: List[int] = Field(..., description="Shape of captured activation tensor")
    slice_values: List[float] = Field(..., description="Bounded slice (first N dimensions of final sequence position)")
    stats: VectorStats = Field(..., description="Summary statistics across all dimensions")
    is_captured: bool = Field(True, description="True since this component was extracted via PyTorch hooks")


class LayerInspectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text")
    layer: int = Field(0, ge=0, description="Transformer layer index to inspect (0 to num_layers-1)")
    token_index: int = Field(0, ge=0, description="Token sequence index to inspect (default: 0)")
    slice_size: int = Field(16, ge=4, le=64, description="Number of vector dimensions in slice")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class LayerInspectionResponse(BaseModel):
    text: str
    layer: int
    token_index: int = Field(0, description="Token sequence index inspected")
    token_str: str = Field("", description="Decoded token string inspected")
    num_layers: int
    num_heads: int
    hidden_size: int
    token_count: int
    tokens: List[str]
    components: List[LayerActivationComponent] = Field(
        ...,
        description="Captured activation components in chronological execution order through the block"
    )
    final_hidden_state_shape: List[int] = Field(..., description="Shape of final hidden state after ln_f")
    logits_shape: List[int] = Field(..., description="Shape of final output logits: [batch, seq, vocab]")
    explanation: str = Field(..., description="Explanation of residual connections and block flow")
    explanation_origin: DataOriginType = "Educational explanation"
    data_origin: DataOriginType = "Real model output"


# =====================================================================
# Phase 6 Schemas: Gemini AI Tutor
# =====================================================================

class TutorStatusResponse(BaseModel):
    configured: bool = Field(..., description="True if GEMINI_API_KEY is configured in backend/.env or environment")
    model: str = Field(..., description="Active Gemini model name")
    status_message: str = Field(..., description="Guidance or operational status message")
    data_origin: DataOriginType = "Gemini-written explanation"


class TutorExplainRequest(BaseModel):
    view_name: str = Field(..., min_length=1, max_length=50, description="Identifier of the current inspector view")
    context_summary: Dict[str, Any] = Field(default_factory=dict, description="Bounded numerical & structural context")

    @field_validator("context_summary")
    @classmethod
    def validate_summary_size(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure context summary payload is bounded to protect latency & safety
        serialized = str(v)
        if len(serialized) > 4096:
            raise ValueError("Context summary payload exceeds maximum allowed size (4096 chars).")
        return v


class TutorExplainResponse(BaseModel):
    explanation: str = Field(..., description="Plain-English architectural explanation from Gemini")
    model: str = Field(..., description="Gemini model that generated the explanation")
    data_origin: DataOriginType = "Gemini-written explanation"


class TutorAskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, description="User's natural language question")
    context_summary: Optional[Dict[str, Any]] = Field(default=None, description="Optional bounded view context")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace only.")
        return stripped


class TutorAskResponse(BaseModel):
    answer: str = Field(..., description="AI Tutor pedagogical response")
    model: str = Field(..., description="Gemini model name")
    data_origin: DataOriginType = "Gemini-written explanation"


class QuizQuestion(BaseModel):
    id: int = Field(..., description="Question identifier (1, 2, 3)")
    question: str = Field(..., description="Question prompt")
    options: List[str] = Field(..., min_length=2, max_length=4, description="Multiple choice options")
    correct_index: int = Field(..., ge=0, le=3, description="Zero-based index of the correct option")
    explanation: str = Field(..., description="Explanation of why this answer is correct")


class TutorQuizRequest(BaseModel):
    topic: Optional[str] = Field("general", max_length=50, description="Topic to quiz on: e.g. tokenization, embeddings, attention, residuals")


class TutorQuizResponse(BaseModel):
    topic: str
    questions: List[QuizQuestion] = Field(..., description="3 interactive quiz questions")
    model: str
    data_origin: DataOriginType = "Gemini-written explanation"


class TutorCompareGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512, description="Prompt text to continue")
    local_generation: Optional[str] = Field(None, max_length=1024, description="Local distilgpt2 continuation for comparison")


class TutorCompareGenerationResponse(BaseModel):
    prompt: str
    gemini_continuation: str = Field(..., description="Gemini continuation text")
    model_name: str
    disclaimer: str = Field(..., description="Notice that Gemini is a different, closed foundation model")
    data_origin: DataOriginType = "Gemini-written explanation"


# =====================================================================
# Phase 7 Schemas: Debug Console & Model Comparison
# =====================================================================

class RequestLogEntry(BaseModel):
    id: int = Field(..., description="Sequential request index")
    timestamp: str = Field(..., description="ISO formatted timestamp")
    endpoint: str = Field(..., description="API path requested")
    method: str = Field(..., description="HTTP method (GET/POST)")
    status_code: int = Field(..., description="HTTP status code returned")
    duration_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="Model used for this request")
    device: str = Field(..., description="Device executed on (cpu/cuda)")
    input_token_ids: List[int] = Field(default_factory=list, description="Bounded first 16 token IDs")
    tensor_shapes: Dict[str, List[int]] = Field(default_factory=dict, description="Captured tensor dimensions")
    tensor_dtypes: Dict[str, str] = Field(default_factory=dict, description="Captured tensor data types")
    layer_selected: Optional[int] = Field(None, description="Layer index if applicable")
    head_selected: Optional[int] = Field(None, description="Head index if applicable")
    generation_settings: Optional[Dict[str, Any]] = Field(None, description="Sampling settings if generation")
    gemini_status: Optional[str] = Field(None, description="Gemini API invocation status (success/error type only)")
    warnings: List[str] = Field(default_factory=list, description="Sanitized warnings scrubbed of paths")
    errors: List[str] = Field(default_factory=list, description="Sanitized error descriptions scrubbed of paths")


class DebugLogsResponse(BaseModel):
    total_captured: int
    capacity: int = 50
    logs: List[RequestLogEntry]
    data_origin: DataOriginType = "Real model output"


class DebugInfoResponse(BaseModel):
    model_name: str
    model_revision: str
    device: str
    dtype: str
    attn_implementation: str
    tokenizer_class: str
    vocab_size: int
    torch_version: str
    transformers_version: str
    data_origin: DataOriginType = "Real model output"


class ModelComparisonStats(BaseModel):
    model_name: str
    param_count: int = Field(..., description="Total trainable parameter count")
    param_count_display: str = Field(..., description="Formatted parameter count (e.g. 82M or 124M)")
    hidden_size: int = Field(..., description="Hidden state dimension (768)")
    num_layers: int = Field(..., description="Number of Transformer blocks")
    num_heads: int = Field(..., description="Number of attention heads")
    vocab_size: int = Field(..., description="Vocabulary token count")
    tokens: List[str] = Field(..., description="Token strings for the input text")
    token_ids: List[int] = Field(..., description="Token IDs for the input text")
    top_candidates: List[CandidateToken] = Field(..., description="Top 10 predicted next tokens")
    top_1_token_id: int
    top_1_token_str: str
    top_1_probability: float
    total_top_k_prob: float


class CompareModelsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text to evaluate on both models")
    model_a: str = Field("distilgpt2", description="Primary model identifier")
    model_b: str = Field("gpt2", description="Secondary model identifier")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only.")
        return v


class CompareModelsResponse(BaseModel):
    text: str
    model_a: ModelComparisonStats
    model_b: ModelComparisonStats
    differences: Dict[str, Any] = Field(..., description="Explicit architectural differences (e.g. layers, parameter gap)")
    explanation: str = Field(..., description="Educational explanation comparing distilgpt2 distillation with gpt2")
    explanation_origin: DataOriginType = "Educational explanation"
    data_origin: DataOriginType = "Real model output"


class LogitLensRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text to analyze through the Logit Lens")
    token_index: Optional[int] = Field(None, description="Sequence token position to unembed (default: last token for next-token prediction)")
    top_k: int = Field(5, ge=1, le=20, description="Number of candidate tokens to extract per layer")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only.")
        return v


class LogitLensCandidate(BaseModel):
    token_id: int
    token_str: str
    probability: float
    logit: float
    rank: int


class LogitLensLayerStep(BaseModel):
    layer: int = Field(..., description="Layer index (0 is embedding output, 1..6 are Transformer blocks)")
    layer_name: str = Field(..., description="Descriptive layer label")
    top_candidates: List[LogitLensCandidate] = Field(..., description="Top-ranked candidate tokens at this intermediate layer")
    target_rank: int = Field(..., description="Rank of the final layer's #1 predicted token at this layer")
    target_probability: float = Field(..., description="Probability of the final layer's #1 predicted token at this layer")
    entropy: float = Field(..., description="Shannon entropy (nats) across vocabulary at this layer")
    cosine_sim_to_previous: Optional[float] = Field(None, description="Cosine similarity of residual hidden state to previous layer")
    residual_norm: float = Field(..., description="L2 norm of residual hidden state vector")


class LogitLensResponse(BaseModel):
    prompt: str
    token_count: int
    selected_token_index: int
    selected_token_str: str
    final_top_token_id: int
    final_top_token_str: str
    layers: List[LogitLensLayerStep]
    explanation: str = Field(..., description="Educational explanation of Logit Lens unembedding and prediction evolution")
    explanation_origin: DataOriginType = "Educational explanation"
    data_origin: DataOriginType = "Real model output"


class AttentionRolloutRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text to analyze via Attention Rollout")
    residual_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight (alpha) allocated to attention vs identity residual connection")
    head_reduction: Literal["mean", "max", "min"] = Field("mean", description="Reduction strategy across attention heads")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only.")
        return v


class TokenAttributionItem(BaseModel):
    token_index: int
    token_str: str
    attribution_score: float
    percentage: float


class LayerRolloutStep(BaseModel):
    layer: int = Field(..., description="Layer index (0..5 for distilgpt2)")
    layer_name: str
    rollout_matrix: List[List[float]] = Field(..., description="Cumulative flow matrix [seq_len, seq_len] up to this layer")
    last_token_attribution: List[TokenAttributionItem] = Field(..., description="Credit attribution from the last token to all input tokens")


class AttentionRolloutResponse(BaseModel):
    prompt: str
    tokens: List[str]
    token_count: int
    residual_weight: float
    head_reduction: str
    layer_steps: List[LayerRolloutStep]
    final_attribution: List[TokenAttributionItem] = Field(..., description="End-to-end token attribution for the next-token prediction position")
    top_attributed_token: str
    top_attributed_score: float
    explanation: str = Field(..., description="Educational explanation of cumulative attention flow, residual identity, and attention sinks")
    explanation_origin: DataOriginType = "Educational explanation"
    data_origin: DataOriginType = "Real model output"


