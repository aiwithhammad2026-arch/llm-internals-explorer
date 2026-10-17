"""Attention Rollout service for quantifying cumulative information flow.
By Hammad, AI Engineer.

Implements the Attention Rollout formulation (Abnar & Zuidema, ACL 2020):
Recursively computes cumulative attention flow across depth, accounting for
residual connection identity highways:
    A_hat_l = alpha * A_l + (1 - alpha) * I
    R_l = A_hat_l * R_{l-1}
"""

from typing import List
import torch

from ..schemas import (
    AttentionRolloutResponse,
    LayerRolloutStep,
    TokenAttributionItem,
)
from ..model_loader import model_manager


def compute_attention_rollout(
    text: str,
    residual_weight: float = 0.5,
    head_reduction: str = "mean"
) -> AttentionRolloutResponse:
    """Computes recursive attention rollout across all transformer layers."""
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
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # 2. Forward pass extracting eager attention weights
    with torch.no_grad():
        outputs = model(**encoded.to(device), output_attentions=True)

    attentions = outputs.attentions  # Tuple of num_layers tensors [1, num_heads, seq_len, seq_len]
    num_layers = len(attentions)
    eye = torch.eye(seq_len, device=device)

    layer_steps: List[LayerRolloutStep] = []
    curr_rollout: torch.Tensor | None = None

    alpha = max(0.0, min(1.0, residual_weight))

    for l in range(num_layers):
        # Raw layer attention across all heads: [num_heads, seq_len, seq_len]
        head_attns = attentions[l][0]

        # Reduce across attention heads
        if head_reduction == "max":
            a_l, _ = head_attns.max(dim=0)
        elif head_reduction == "min":
            a_l, _ = head_attns.min(dim=0)
        else:
            # Default to mean across heads
            a_l = head_attns.mean(dim=0)

        # Interpolate with residual connection identity
        # A_hat = alpha * A_l + (1 - alpha) * I
        a_hat = alpha * a_l + (1.0 - alpha) * eye

        # Re-normalize rows so each row sums strictly to 1.0
        row_sums = a_hat.sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1.0  # Avoid zero division
        a_hat = a_hat / row_sums

        # Recursive matrix multiplication: R_l = A_hat * R_{l-1}
        if curr_rollout is None:
            curr_rollout = a_hat
        else:
            curr_rollout = torch.matmul(a_hat, curr_rollout)

        # Enforce exact causal upper triangle zeroing and row re-normalization
        curr_rollout = torch.tril(curr_rollout)
        curr_rollout = curr_rollout / curr_rollout.sum(dim=-1, keepdim=True)

        # Extract attribution for the last token position
        last_row = curr_rollout[-1, :].tolist()
        last_attribution_items = [
            TokenAttributionItem(
                token_index=idx,
                token_str=tok,
                attribution_score=round(float(score), 6),
                percentage=round(float(score) * 100, 2)
            )
            for idx, (tok, score) in enumerate(zip(tokens, last_row))
        ]

        matrix_2d = [
            [round(float(val), 6) for val in row]
            for row in curr_rollout.tolist()
        ]

        layer_steps.append(
            LayerRolloutStep(
                layer=l,
                layer_name=f"Layer {l} (Block {l})",
                rollout_matrix=matrix_2d,
                last_token_attribution=last_attribution_items
            )
        )

    final_step = layer_steps[-1]
    final_attribution = final_step.last_token_attribution

    # Determine highest-attributed input token for the next-token prediction position
    top_item = max(final_attribution, key=lambda item: item.attribution_score)

    explanation = (
        "Attention Rollout (Abnar & Zuidema, 2020) quantifies cumulative information flow through multi-layer Transformers. "
        "Because representations pass through both multi-head attention and residual bypass highways (x = x + f(x)), "
        "isolated single-layer attention matrices only reflect local routing. By recursively accumulating adjusted attention "
        f"matrices across all {num_layers} layers, rollout reveals how much information from each input token reached the final "
        f"prediction position. In GPT-2 architectures, position 0 ('{tokens[0]}') frequently acts as an attention sink, absorbing "
        f"{top_item.percentage:.1f}% of cumulative flow, while later tokens contribute fine-grained syntactic and semantic context."
    )

    return AttentionRolloutResponse(
        prompt=text,
        tokens=tokens,
        token_count=seq_len,
        residual_weight=residual_weight,
        head_reduction=head_reduction,
        layer_steps=layer_steps,
        final_attribution=final_attribution,
        top_attributed_token=top_item.token_str,
        top_attributed_score=top_item.attribution_score,
        explanation=explanation,
        explanation_origin="Educational explanation",
        data_origin="Real model output"
    )
