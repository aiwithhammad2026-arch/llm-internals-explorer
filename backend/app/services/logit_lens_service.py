"""Logit Lens service for inspecting residual stream prediction evolution.
By Hammad, AI Engineer.

The Logit Lens applies the final unembedding projection (ln_f + lm_head)
to each intermediate hidden state in the residual stream (Layer 0 to Layer N),
revealing how token predictions and candidate rankings evolve layer by layer.
"""

import time
from typing import Optional, List
import torch
import torch.nn.functional as F

from ..schemas import (
    LogitLensResponse,
    LogitLensLayerStep,
    LogitLensCandidate,
)
from ..model_loader import model_manager


def compute_logit_lens(
    text: str,
    token_index: Optional[int] = None,
    top_k: int = 5
) -> LogitLensResponse:
    """Executes a real forward pass and applies the unembedding head to each residual layer."""
    start_time = time.time()

    if not model_manager.is_loaded or model_manager.model is None or model_manager.tokenizer is None:
        model_manager.load_model()

    model = model_manager.model
    tokenizer = model_manager.tokenizer
    device = model_manager.device

    if model is None or tokenizer is None:
        raise RuntimeError("Model and tokenizer must be initialized.")

    # 1. Tokenize prompt
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # Resolve token index (default to the last token where next-token prediction happens)
    target_pos = seq_len - 1 if token_index is None or token_index == -1 else token_index
    if not (0 <= target_pos < seq_len):
        raise ValueError(
            f"Token position {target_pos} out of sequence range [0, {seq_len - 1}]."
        )

    selected_token_str = tokens_str[target_pos]

    # 2. Forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model(**encoded.to(device), output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors [1, seq_len, hidden_size]
    num_layers = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", 6))

    # Final model prediction at the target position
    final_logits = outputs.logits[0, target_pos, :]
    final_top_id = int(torch.argmax(final_logits).item())
    final_top_str = tokenizer.decode([final_top_id])

    layer_steps: List[LogitLensLayerStep] = []

    # 3. Project each intermediate residual state through the LM head
    for l, h in enumerate(hidden_states):
        # Hugging Face GPT2Model applies ln_f after the last block, so hidden_states[-1] is already post-ln_f.
        # Intermediate hidden states (l < num_layers) are pre-ln_f and must be normalized through ln_f first.
        if l < num_layers:
            h_norm = model.transformer.ln_f(h)
        else:
            h_norm = h

        # Project through linear lm_head: [1, seq_len, vocab_size] -> [vocab_size]
        logits_l = model.lm_head(h_norm)[0, target_pos, :]
        probs_l = F.softmax(logits_l, dim=-1)

        # Extract top_k candidates
        top_probs, top_ids = torch.topk(probs_l, top_k)
        candidates: List[LogitLensCandidate] = []
        for rank_idx in range(top_k):
            tid = int(top_ids[rank_idx].item())
            candidates.append(
                LogitLensCandidate(
                    token_id=tid,
                    token_str=tokenizer.decode([tid]),
                    probability=round(float(top_probs[rank_idx].item()), 6),
                    logit=round(float(logits_l[tid].item()), 4),
                    rank=rank_idx + 1
                )
            )

        # Determine target token's rank and probability at this layer
        sorted_ids = torch.argsort(logits_l, descending=True)
        target_rank = int((sorted_ids == final_top_id).nonzero(as_tuple=True)[0].item()) + 1
        target_prob = round(float(probs_l[final_top_id].item()), 6)

        # Shannon entropy across vocabulary: H = -sum(p * ln(p))
        entropy = round(float(-torch.sum(probs_l * torch.log(probs_l + 1e-12)).item()), 4)

        # Cosine similarity to preceding layer's hidden state vector at this token position
        curr_vec = h[0, target_pos, :].float()
        if l > 0:
            prev_vec = hidden_states[l - 1][0, target_pos, :].float()
            cos_sim = round(
                float(F.cosine_similarity(curr_vec.unsqueeze(0), prev_vec.unsqueeze(0)).item()),
                6
            )
        else:
            cos_sim = None

        residual_norm = round(float(torch.linalg.norm(curr_vec).item()), 4)

        layer_name = "Embedding (Layer 0)" if l == 0 else f"Block {l - 1} (Layer {l})"
        layer_steps.append(
            LogitLensLayerStep(
                layer=l,
                layer_name=layer_name,
                top_candidates=candidates,
                target_rank=target_rank,
                target_probability=target_prob,
                entropy=entropy,
                cosine_sim_to_previous=cos_sim,
                residual_norm=residual_norm
            )
        )

    explanation = (
        "The Logit Lens applies the model's final unembedding projection (ln_f + lm_head) to each intermediate "
        "hidden state along the residual stream. In Transformer architectures, representations evolve hierarchically: "
        "early layers prioritize syntactic and positional structure, intermediate layers assemble semantic and "
        "relational context, and late layers refine the final vocabulary distribution. Tracking the rank and probability "
        f"of the final output token '{final_top_str}' shows the exact layer where the model's final prediction crystallized."
    )

    return LogitLensResponse(
        prompt=text,
        token_count=seq_len,
        selected_token_index=target_pos,
        selected_token_str=selected_token_str,
        final_top_token_id=final_top_id,
        final_top_token_str=final_top_str,
        layers=layer_steps,
        explanation=explanation,
        explanation_origin="Educational explanation",
        data_origin="Real model output"
    )
