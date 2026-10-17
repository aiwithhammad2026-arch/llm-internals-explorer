"""Model comparison service for distilgpt2 and gpt2.
By Hammad, AI Engineer.

Safely caches up to 2 models in memory, verifies architecture compatibility (GPT-2 family),
and evaluates side-by-side tokenization, parameter count, and next-token logits.
"""

import logging
from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import settings, ALLOWED_MODELS
from ..schemas import (
    ModelComparisonStats,
    CandidateToken,
    CompareModelsResponse,
)

logger = logging.getLogger("transformer_explorer.comparison")

class ModelOutOfMemoryError(RuntimeError):
    """Raised when host system RAM or GPU memory is exhausted during model loading."""
    def __init__(self, model_name: str, message: str = ""):
        self.model_name = model_name
        self.remediation = [
            "Close memory-heavy applications to free host system RAM.",
            "Run single-model inspector endpoints with distilgpt2 (~350MB RAM) instead of dual-model comparison.",
            "Increase system paging file / virtual memory swap size."
        ]
        clean_msg = message or f"Insufficient memory to load model '{model_name}'."
        super().__init__(clean_msg)

# Cache up to 2 models in memory
_comparison_cache: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}
MAX_CACHED_MODELS = 2

def _get_or_load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Lazily loads and caches a model from the allowed models list."""
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not in the allowed models list: {ALLOWED_MODELS}. "
            "Only trusted GPT-2 family models are permitted."
        )

    if model_name in _comparison_cache:
        return _comparison_cache[model_name]

    # Enforce memory safety cap of at most 2 models
    if len(_comparison_cache) >= MAX_CACHED_MODELS:
        # Evict a model that is not the primary distilgpt2 if possible
        evict_key = next((k for k in _comparison_cache if k != "distilgpt2"), next(iter(_comparison_cache)))
        logger.info("Evicting '%s' from comparison model cache to maintain memory limits.", evict_key)
        del _comparison_cache[evict_key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Lazily loading comparison model '%s' with eager attention...", model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            dtype=torch.float32,
        )
        model.eval()

        # Check model_type compatibility (both must be gpt2 architecture)
        model_type = getattr(model.config, "model_type", None)
        if model_type != "gpt2":
            raise ValueError(
                f"Model '{model_name}' has architecture type '{model_type}'. "
                "Only 'gpt2' architecture models are supported for direct internal comparison."
            )

        # Populate real commit hash on model config from snapshot cache
        try:
            import re
            from huggingface_hub import try_to_load_from_cache
            cached_file = str(try_to_load_from_cache(model_name, "config.json"))
            match = re.search(r'snapshots[\\/]([0-9a-fA-F]{40})', cached_file)
            if match:
                model.config._commit_hash = match.group(1)
        except Exception:
            pass

        _comparison_cache[model_name] = (model, tokenizer)
        return model, tokenizer

    except (MemoryError, torch.cuda.OutOfMemoryError) as oom:
        logger.error("Out of memory while loading comparison model '%s': %s", model_name, oom)
        raise ModelOutOfMemoryError(
            model_name=model_name,
            message=f"Host system or GPU ran out of memory while loading '{model_name}'."
        ) from oom
    except RuntimeError as re:
        if "out of memory" in str(re).lower():
            logger.error("RuntimeError out-of-memory while loading model '%s': %s", model_name, re)
            raise ModelOutOfMemoryError(
                model_name=model_name,
                message=f"PyTorch allocator exhausted memory while loading '{model_name}'."
            ) from re
        logger.error("Failed to load comparison model '%s': %s", model_name, re)
        raise RuntimeError(
            f"Could not load comparison model '{model_name}'. Please verify internet connectivity and disk space."
        ) from re
    except Exception as e:
        logger.error("Failed to load comparison model '%s': %s", model_name, e)
        raise RuntimeError(
            f"Could not load comparison model '{model_name}'. Please verify internet connectivity and disk space."
        ) from e

def _evaluate_model_on_text(model_name: str, text: str) -> ModelComparisonStats:
    """Evaluates a single model on text: returns parameters, shapes, tokens, and top-10 candidates."""
    model, tokenizer = _get_or_load_model(model_name)

    # 1. Parameter count and architectural config
    param_count = sum(p.numel() for p in model.parameters())
    num_layers = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", 6))
    num_heads = getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", 12))
    hidden_size = getattr(model.config, "n_embd", getattr(model.config, "hidden_size", 768))
    vocab_size = getattr(tokenizer, "vocab_size", 50257)

    # Display formatting for params (e.g. "81.9M (~82M)" or "124.4M (~124M)")
    params_millions = round(param_count / 1_000_000, 1)
    param_display = f"{params_millions}M (~{round(params_millions)}M)"

    # 2. Tokenize text
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    token_ids_list = input_ids[0].tolist()
    token_strings = [tokenizer.decode([tid]) for tid in token_ids_list]

    # 3. Predict next token probabilities
    with torch.no_grad():
        outputs = model(input_ids)
        final_logits = outputs.logits[0, -1, :] # [vocab_size]
        probs = F.softmax(final_logits, dim=-1)

    top_k_values, top_k_indices = torch.topk(probs, k=10)
    top_logits = final_logits[top_k_indices].tolist()

    top_candidates: List[CandidateToken] = []
    cum_prob = 0.0

    for rank, (tid_t, p_t, logit_val) in enumerate(zip(top_k_indices, top_k_values, top_logits), start=1):
        tid = int(tid_t.item())
        prob = float(p_t.item())
        cum_prob += prob

        tok_str = tokenizer.decode([tid])
        tok_bytes = repr(tok_str).strip("'")

        top_candidates.append(
            CandidateToken(
                rank=rank,
                token_id=tid,
                token_str=tok_str,
                token_bytes_display=tok_bytes,
                logit=round(logit_val, 4),
                probability=round(prob, 6),
                cumulative_probability=round(cum_prob, 6),
            )
        )

    top_1 = top_candidates[0]

    return ModelComparisonStats(
        model_name=model_name,
        param_count=param_count,
        param_count_display=param_display,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        tokens=token_strings,
        token_ids=token_ids_list,
        top_candidates=top_candidates,
        top_1_token_id=top_1.token_id,
        top_1_token_str=top_1.token_str,
        top_1_probability=top_1.probability,
        total_top_k_prob=round(cum_prob, 6),
    )

def compare_models(text: str, model_a_name: str = "distilgpt2", model_b_name: str = "gpt2") -> CompareModelsResponse:
    """Executes side-by-side evaluation between two allowed models."""
    if model_a_name not in ALLOWED_MODELS:
        raise ValueError(f"Model '{model_a_name}' is not in the allowed models list: {ALLOWED_MODELS}")
    if model_b_name not in ALLOWED_MODELS:
        raise ValueError(f"Model '{model_b_name}' is not in the allowed models list: {ALLOWED_MODELS}")

    stats_a = _evaluate_model_on_text(model_a_name, text)
    stats_b = _evaluate_model_on_text(model_b_name, text)

    differences = {
        "parameter_delta": stats_b.param_count - stats_a.param_count,
        "parameter_ratio": round(stats_b.param_count / stats_a.param_count, 2),
        "layer_delta": stats_b.num_layers - stats_a.num_layers,
        "shared_hidden_size": stats_a.hidden_size == stats_b.hidden_size,
        "shared_vocab_size": stats_a.vocab_size == stats_b.vocab_size,
        "top_1_agreement": stats_a.top_1_token_id == stats_b.top_1_token_id,
        "top_1_token": stats_a.top_1_token_str if stats_a.top_1_token_id == stats_b.top_1_token_id else None,
    }

    explanation = (
        f"Comparison between '{stats_a.model_name}' and '{stats_b.model_name}': "
        f"distilgpt2 is a knowledge-distilled compression of gpt2 created by Hugging Face. "
        f"It halves the depth from 12 transformer blocks to 6 layers while retaining the exact same "
        f"768 hidden dimension and 12 attention heads. This reduces parameters from 124,439,808 to 81,912,576 (~34% smaller), "
        f"halving the sequential transformer layer count per forward pass; actual wall-clock speedup depends on CPU/GPU hardware and batch size."
    )

    return CompareModelsResponse(
        text=text,
        model_a=stats_a,
        model_b=stats_b,
        differences=differences,
        explanation=explanation,
        explanation_origin="Educational explanation",
        data_origin="Real model output",
    )
