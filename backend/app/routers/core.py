"""Core API router exposing real Transformer model endpoints.
By Hammad, AI Engineer.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from ..schemas import (
    HealthResponse,
    ModelInfoResponse,
    TokenizeRequest,
    TokenizeResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    PredictNextTokenRequest,
    PredictNextTokenResponse,
    GenerateRequest,
    GenerateResponse,
    AttentionRequest,
    AttentionResponse,
    LayerInspectionRequest,
    LayerInspectionResponse
)
from ..model_loader import model_manager
from ..services.tokenizer_service import tokenize_text
from ..services.embedding_service import extract_embeddings
from ..services.generation_service import predict_next_token, generate_text
from ..services.attention_service import extract_attention
from ..services.inspection_service import inspect_layer
from ..services.debug_service import scrub_sensitive_text

router = APIRouter(prefix="/api", tags=["core"])

def _ensure_model_ready():
    """Helper to ensure model is ready or attempt initialization."""
    if not model_manager.is_loaded:
        try:
            model_manager.load_model()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Transformer model is not loaded.",
                    "details": str(e),
                    "setup_guidance": model_manager.setup_guidance
                }
            )

def _validate_layer_and_head(layer: int, head: Optional[int] = None):
    """Dynamically validates layer and head indices against the loaded model config."""
    if not model_manager.model:
        _ensure_model_ready()

    config = model_manager.model.config
    num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 6))
    num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", 12))

    if not (0 <= layer < num_layers):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid layer index",
                "message": f"Layer {layer} is out of range. The loaded model has {num_layers} layers (valid indices: 0 to {num_layers - 1}).",
                "valid_range": [0, num_layers - 1]
            }
        )

    if head is not None and not (0 <= head < num_heads):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid head index",
                "message": f"Head {head} is out of range. The loaded model has {num_heads} attention heads (valid indices: 0 to {num_heads - 1}).",
                "valid_range": [0, num_heads - 1]
            }
        )

@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Returns system status, device, and whether the pretrained Transformer model is ready in memory."""
    if model_manager.is_loaded:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=model_manager.model_name,
            device=str(model_manager.device),
            error_message=None,
            setup_guidance=None,
            data_origin="Real model output"
        )
    return HealthResponse(
        status="degraded" if model_manager.error_message else "initializing",
        model_loaded=False,
        model_name=model_manager.model_name,
        device=str(model_manager.device),
        error_message=model_manager.error_message,
        setup_guidance=model_manager.setup_guidance,
        data_origin="Real model output"
    )

@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Returns verified architectural metadata from the active pretrained model."""
    _ensure_model_ready()
    try:
        info = model_manager.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract model metadata: {scrub_sensitive_text(str(e))}"
        )

@router.post("/tokenize", response_model=TokenizeResponse)
def post_tokenize(req: TokenizeRequest) -> TokenizeResponse:
    """Tokenizes input string using the real tokenizer and extracts true token IDs and tensor shape."""
    _ensure_model_ready()
    try:
        return tokenize_text(req.text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tokenization failed: {scrub_sensitive_text(str(e))}"
        )

@router.post("/embeddings", response_model=EmbeddingResponse)
def post_embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    """Extracts real token embeddings (wte), positional embeddings (wpe), and their combined sum."""
    _ensure_model_ready()
    try:
        return extract_embeddings(text=req.text, slice_size=req.slice_size)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding extraction failed: {scrub_sensitive_text(str(e))}"
        )

@router.post("/predict-next-token", response_model=PredictNextTokenResponse)
def post_predict_next_token(req: PredictNextTokenRequest) -> PredictNextTokenResponse:
    """Performs real model forward pass to compute next-token probabilities and top-k candidate rankings."""
    _ensure_model_ready()
    try:
        return predict_next_token(
            text=req.text,
            temperature=req.temperature,
            top_k=req.top_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Next-token prediction failed: {scrub_sensitive_text(str(e))}"
        )

@router.post("/generate", response_model=GenerateResponse)
def post_generate(req: GenerateRequest) -> GenerateResponse:
    """Generates autoregressive token continuation using real PyTorch model sampling."""
    _ensure_model_ready()
    try:
        return generate_text(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            seed=req.seed
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text generation failed: {scrub_sensitive_text(str(e))}"
        )

@router.post("/attention", response_model=AttentionResponse)
def post_attention(req: AttentionRequest) -> AttentionResponse:
    """Extracts real attention weights and Q/K/V tensors for a given layer and head."""
    _ensure_model_ready()
    _validate_layer_and_head(layer=req.layer, head=req.head)
    try:
        return extract_attention(
            text=req.text,
            layer=req.layer,
            head=req.head
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=scrub_sensitive_text(str(ve))
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Attention extraction failed: {scrub_sensitive_text(str(e))}"
        )

@router.post("/layer-inspection", response_model=LayerInspectionResponse)
def post_layer_inspection(req: LayerInspectionRequest) -> LayerInspectionResponse:
    """Captures real intermediate activations at each sub-layer of a Transformer block using hooks."""
    _ensure_model_ready()
    _validate_layer_and_head(layer=req.layer, head=None)
    try:
        return inspect_layer(
            text=req.text,
            layer=req.layer,
            token_index=req.token_index,
            slice_size=req.slice_size
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=scrub_sensitive_text(str(ve))
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Layer inspection failed: {scrub_sensitive_text(str(e))}"
        )
