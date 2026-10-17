"""Embedding service for extracting real token embeddings, positional embeddings, and their sum.
By Hammad, AI Engineer.
"""

from typing import List
import torch
from ..schemas import (
    EmbeddingResponse,
    TokenEmbeddingItem,
    VectorStats
)
from ..model_loader import model_manager

def _compute_vector_stats(vec: torch.Tensor) -> VectorStats:
    """Computes summary statistics across full hidden dimension."""
    v = vec.detach().float()
    return VectorStats(
        min=round(float(torch.min(v).item()), 6),
        max=round(float(torch.max(v).item()), 6),
        mean=round(float(torch.mean(v).item()), 6),
        std=round(float(torch.std(v).item()), 6),
        norm=round(float(torch.linalg.norm(v).item()), 6)
    )

def extract_embeddings(text: str, slice_size: int = 16) -> EmbeddingResponse:
    """Extracts real token embedding (wte), positional embedding (wpe), and their combined sum.
    Returns bounded slices and full-vector summary statistics.
    """
    if not model_manager.is_loaded or model_manager.model is None or model_manager.tokenizer is None:
        model_manager.load_model()

    model = model_manager.model
    tokenizer = model_manager.tokenizer
    device = model_manager.device

    if model is None or tokenizer is None:
        raise RuntimeError("Model and tokenizer must be initialized.")

    # 1. Encode text to input IDs
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    # 2. Extract real learned token embeddings (wte)
    with torch.no_grad():
        wte = model.transformer.wte(input_ids) # [1, seq_len, 768]

        # 3. Extract real learned positional embeddings (wpe)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        wpe = model.transformer.wpe(position_ids).unsqueeze(0) # [1, seq_len, 768]

        # 4. Combined sum (the exact representation that enters Transformer Block 0)
        combined = wte + wpe # [1, seq_len, 768]

    tokens_list: List[TokenEmbeddingItem] = []
    token_ids_list = input_ids[0].tolist()

    for idx, token_id in enumerate(token_ids_list):
        token_str = tokenizer.decode([token_id])

        t_vec = wte[0, idx]
        p_vec = wpe[0, idx]
        c_vec = combined[0, idx]

        # Slices bounded to the requested slice_size (e.g. first 16 values)
        t_slice = [round(float(x), 6) for x in t_vec[:slice_size].tolist()]
        p_slice = [round(float(x), 6) for x in p_vec[:slice_size].tolist()]
        c_slice = [round(float(x), 6) for x in c_vec[:slice_size].tolist()]

        tokens_list.append(
            TokenEmbeddingItem(
                index=idx,
                token_id=token_id,
                token_str=token_str,
                token_embedding_slice=t_slice,
                token_stats=_compute_vector_stats(t_vec),
                positional_embedding_slice=p_slice,
                positional_stats=_compute_vector_stats(p_vec),
                combined_embedding_slice=c_slice,
                combined_stats=_compute_vector_stats(c_vec)
            )
        )

    explanation = (
        "In GPT-2 style architectures, the initial input representation is formed by two learned embedding matrices: "
        "(1) Word Token Embedding (wte) maps vocabulary token IDs to continuous vectors. "
        "(2) Word Positional Embedding (wpe) maps sequence positions (0 to seq_len-1) to learned position vectors. "
        "These two vectors are added element-wise: combined = wte + wpe. In evaluation mode (dropout disabled), "
        "this sum is identically equal to the model's initial hidden state (hidden_states[0]) feeding into Block 0."
    )

    return EmbeddingResponse(
        text=text,
        token_count=seq_len,
        hidden_dimension=wte.shape[-1],
        slice_size=slice_size,
        token_embedding_shape=list(wte.shape),
        positional_embedding_shape=list(wpe.shape),
        combined_embedding_shape=list(combined.shape),
        tokens=tokens_list,
        explanation=explanation,
        explanation_origin="Educational explanation",
        data_origin="Real model output"
    )
