"""Transformer Block inspection service using PyTorch hooks.
By Hammad, AI Engineer.

Captures real intermediate activations at each stage of a Transformer block:
LayerNorm 1 -> Multi-Head Attention -> Residual 1 -> LayerNorm 2 -> MLP -> Residual 2
"""

from typing import List, Dict
import torch
from ..schemas import (
    LayerInspectionResponse,
    LayerActivationComponent,
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

def inspect_layer(text: str, layer: int, slice_size: int = 16, token_index: int = 0) -> LayerInspectionResponse:
    """Captures real intermediate activations through a chosen Transformer block for a selected token."""
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

    if not (0 <= layer < num_layers):
        raise ValueError(f"Layer {layer} out of range [0, {num_layers - 1}].")

    # 1. Tokenize prompt
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    if not (0 <= token_index < seq_len):
        raise ValueError(f"Token index {token_index} out of range [0, {seq_len - 1}].")

    # 2. Register temporary forward hooks on target block components
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    target_block = model.transformer.h[layer]

    def hook_ln1(mod, inp, out): captured["ln_1"] = out
    def hook_attn_proj(mod, inp, out): captured["attn_proj"] = out
    def hook_ln2(mod, inp, out): captured["ln_2"] = out
    def hook_mlp_proj(mod, inp, out): captured["mlp_proj"] = out

    handles.append(target_block.ln_1.register_forward_hook(hook_ln1))
    handles.append(target_block.attn.c_proj.register_forward_hook(hook_attn_proj))
    handles.append(target_block.ln_2.register_forward_hook(hook_ln2))
    handles.append(target_block.mlp.c_proj.register_forward_hook(hook_mlp_proj))

    try:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()

    # 3. Extract captured tensor states
    # Input hidden state entering this block:
    input_hidden = outputs.hidden_states[layer]
    ln_1_out = captured["ln_1"]
    attn_proj_out = captured["attn_proj"]
    residual_attn = input_hidden + attn_proj_out
    ln_2_out = captured["ln_2"]
    mlp_proj_out = captured["mlp_proj"]
    residual_mlp = residual_attn + mlp_proj_out
    final_hidden = outputs.hidden_states[-1] # After ln_f
    logits_shape = list(outputs.logits.shape)

    # Helper to build component item for the selected token index
    def make_comp(name: str, desc: str, tensor: torch.Tensor) -> LayerActivationComponent:
        tok_vec = tensor[0, token_index, :]
        return LayerActivationComponent(
            name=name,
            description=desc,
            shape=list(tensor.shape),
            slice_values=[round(float(x), 6) for x in tok_vec[:slice_size].tolist()],
            stats=_compute_vector_stats(tok_vec),
            is_captured=True
        )

    components: List[LayerActivationComponent] = [
        make_comp(
            "input_hidden_state",
            "Hidden representation entering this block (output from previous block or embedding)",
            input_hidden
        ),
        make_comp(
            "ln_1_output",
            "Normalized activations from LayerNorm 1 prior to multi-head self-attention (Pre-LN)",
            ln_1_out
        ),
        make_comp(
            "attn_output",
            "Projected multi-head attention output (after c_proj)",
            attn_proj_out
        ),
        make_comp(
            "residual_after_attn",
            "First residual stream addition: input + Attention(LN_1(input))",
            residual_attn
        ),
        make_comp(
            "ln_2_output",
            "Normalized activations from LayerNorm 2 prior to Feed-Forward Network (Pre-LN)",
            ln_2_out
        ),
        make_comp(
            "mlp_output",
            "Feed-forward network output after GELU activation and dimension projection (c_proj)",
            mlp_proj_out
        ),
        make_comp(
            "residual_after_mlp",
            "Second residual stream addition: residual_1 + MLP(LN_2(residual_1)), exiting the block",
            residual_mlp
        )
    ]

    explanation = (
        f"Transformer Block {layer} operates on a continuous residual stream. "
        "GPT-2 uses Pre-Layer Normalization (Pre-LN): activations are normalized before each sub-layer. "
        "The first sub-layer applies multi-head self-attention and adds its result back to the residual stream (x + Attn(LN1(x))). "
        "The second sub-layer applies a position-wise feed-forward network (768 -> 3072 -> 768 with GELU) "
        "and adds its result back to the stream (x + MLP(LN2(x))). "
        "Notice that residual connections act as high-speed information highways, preserving gradients and identity across depth."
    )

    return LayerInspectionResponse(
        text=text,
        layer=layer,
        token_index=token_index,
        token_str=tokens_str[token_index],
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        token_count=seq_len,
        tokens=tokens_str,
        components=components,
        final_hidden_state_shape=list(final_hidden.shape),
        logits_shape=logits_shape,
        explanation=explanation,
        explanation_origin="Educational explanation",
        data_origin="Real model output"
    )
