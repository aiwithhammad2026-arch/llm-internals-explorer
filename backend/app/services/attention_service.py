"""Attention and Q/K/V extraction service.
By Hammad, AI Engineer.

Extracts real attention weights from eager attention outputs, captures fused c_attn
projections via forward hooks, and mathematically validates that:
softmax((Q @ K^T) / sqrt(d_k) + causal_mask) == real_attention_weights
"""

import math
from typing import List, Dict, Any
import torch
from ..schemas import (
    AttentionResponse,
    AttentionWeightItem,
    QKVExtractionSummary
)
from ..model_loader import model_manager

def extract_attention(text: str, layer: int, head: int) -> AttentionResponse:
    """Extracts real attention matrix, top-15 weights, and Q/K/V vectors for a specific layer and head."""
    if not model_manager.is_loaded or model_manager.model is None or model_manager.tokenizer is None:
        model_manager.load_model()

    model = model_manager.model
    tokenizer = model_manager.tokenizer
    device = model_manager.device

    if model is None or tokenizer is None:
        raise RuntimeError("Model and tokenizer must be initialized.")

    config = model.config
    num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 6))
    num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", 12))
    hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
    head_dim = hidden_size // num_heads # 64 for distilgpt2

    if not (0 <= layer < num_layers):
        raise ValueError(f"Layer {layer} out of range [0, {num_layers - 1}].")
    if not (0 <= head < num_heads):
        raise ValueError(f"Head {head} out of range [0, {num_heads - 1}].")

    # 1. Tokenize prompt
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # 2. Hook onto fused c_attn projection of the selected layer
    c_attn_storage: Dict[str, torch.Tensor] = {}
    def hook_fn(module, inp, output):
        c_attn_storage["fused"] = output

    target_block = model.transformer.h[layer]
    hook_handle = target_block.attn.c_attn.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
    finally:
        hook_handle.remove()

    # 3. Extract real attention weights: shape [1, num_heads, seq_len, seq_len]
    real_attentions = outputs.attentions[layer]
    selected_head_attn = real_attentions[0, head] # shape [seq_len, seq_len]

    # Convert to 2D list
    matrix: List[List[float]] = [
        [round(float(v), 6) for v in row] for row in selected_head_attn.tolist()
    ]

    # Check causal mask property: strictly upper triangle (j > i) must be 0.0
    upper_triangle_zero = True
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if matrix[i][j] != 0.0:
                upper_triangle_zero = False
                break

    # 4. Extract Top 15 weights
    weight_items: List[AttentionWeightItem] = []
    for i in range(seq_len):
        for j in range(seq_len):
            # Only consider valid causal connections
            if j <= i:
                weight_items.append(
                    AttentionWeightItem(
                        source_index=i,
                        source_token=tokens_str[i],
                        target_index=j,
                        target_token=tokens_str[j],
                        weight=matrix[i][j]
                    )
                )

    weight_items.sort(key=lambda x: x.weight, reverse=True)
    top_15_weights = weight_items[:15]

    # 5. Q/K/V extraction from captured fused c_attn projection
    fused_c_attn = c_attn_storage["fused"] # shape [1, seq_len, 2304]
    q, k, v = fused_c_attn.split(hidden_size, dim=2) # each [1, seq_len, 768]

    # Reshape each into [1, num_heads, seq_len, head_dim]
    q_heads = q.view(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k_heads = k.view(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    v_heads = v.view(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    # 6. Verify mathematical recomputation: softmax((Q @ K^T) / sqrt(64) + causal_mask)
    with torch.no_grad():
        scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) / math.sqrt(head_dim)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )
        scores = scores + causal_mask
        recomputed_attn = torch.softmax(scores, dim=-1)

        # Verify: computed directly against outputs.attentions[layer][0, head] (the model's real output)
        diff = torch.max(torch.abs(selected_head_attn - recomputed_attn[0, head])).item()
        max_diff = round(float(diff), 6)

    # Slices of Q, K, V for selected head (final token position)
    q_selected = q_heads[0, head] # [seq_len, 64]
    k_selected = k_heads[0, head] # [seq_len, 64]
    v_selected = v_heads[0, head] # [seq_len, 64]

    q_slice = [round(float(x), 6) for x in q_selected[-1, :16].tolist()]
    k_slice = [round(float(x), 6) for x in k_selected[-1, :16].tolist()]
    v_slice = [round(float(x), 6) for x in v_selected[-1, :16].tolist()]

    qkv_summary = QKVExtractionSummary(
        layer=layer,
        head=head,
        head_dim=head_dim,
        fused_c_attn_width=fused_c_attn.shape[-1],
        q_shape=list(q_selected.shape),
        k_shape=list(k_selected.shape),
        v_shape=list(v_selected.shape),
        q_slice=q_slice,
        k_slice=k_slice,
        v_slice=v_slice,
        max_recomputed_difference=max_diff
    )

    explanation = (
        f"In Layer {layer}, Head {head}, self-attention computes query-key alignment scores: "
        "A = softmax((Q @ K^T) / sqrt(64) + M), where M is the causal attention mask. "
        "Because GPT-2 is an autoregressive decoder, the causal mask sets all upper-triangle positions "
        "(where target position > source position) to -infinity before softmax, ensuring tokens can only attend "
        "to past and current tokens (upper triangle = 0.0). "
        "In GPT-2, many tokens put most attention on the first token, so high weight on it does not mean it is the most important word. "
        "Notice: Attention weights represent dynamic routing of information across token positions, "
        "but attention alone is NOT a complete explanation of model reasoning; downstream non-linear activations "
        "(GELU) and feed-forward projections deeply reshape the final hidden representations."
    )

    return AttentionResponse(
        text=text,
        layer=layer,
        head=head,
        token_count=seq_len,
        tokens=tokens_str,
        matrix=matrix,
        top_weights=top_15_weights,
        is_causal_mask_applied=True,
        upper_triangle_zero=upper_triangle_zero,
        qkv=qkv_summary,
        explanation=explanation,
        explanation_origin="Educational explanation",
        data_origin="Real model output"
    )
