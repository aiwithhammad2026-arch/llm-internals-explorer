"""Generation and next-token prediction service using real PyTorch Transformer logits.
By Hammad, AI Engineer.
"""

from typing import List, Optional
import torch
from ..schemas import (
    PredictNextTokenResponse,
    CandidateToken,
    GenerateResponse,
    GeneratedTokenItem
)
from ..model_loader import model_manager

def _format_token_display(token_id: int, tokenizer) -> str:
    """Formats raw BPE vocabulary tokens with readable whitespace markers."""
    try:
        vocab_symbol = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(vocab_symbol, bytes):
            vocab_symbol = vocab_symbol.decode("utf-8", errors="replace")
        if vocab_symbol.startswith("Ġ"):
            return f"␣{vocab_symbol[1:]}"
        elif vocab_symbol == "Ċ":
            return "\\n"
        elif vocab_symbol == "č":
            return "\\t"
        return vocab_symbol
    except Exception:
        return tokenizer.decode([token_id])

def predict_next_token(text: str, temperature: float = 1.0, top_k: int = 10) -> PredictNextTokenResponse:
    """Performs real model forward pass to compute next-token logits and softmax probabilities."""
    if not model_manager.is_loaded or model_manager.model is None or model_manager.tokenizer is None:
        model_manager.load_model()

    model = model_manager.model
    tokenizer = model_manager.tokenizer
    device = model_manager.device

    if model is None or tokenizer is None:
        raise RuntimeError("Model and tokenizer must be initialized.")

    # 1. Encode prompt
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    prompt_token_ids = input_ids[0].tolist()
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    # 2. Forward pass for logits
    with torch.no_grad():
        outputs = model(input_ids)
        # Raw unscaled logits for the very last token position in the sequence: shape [vocab_size]
        final_logits = outputs.logits[0, -1, :]

        # Scale by temperature: z / T
        scaled_logits = final_logits / temperature

        # Softmax over vocabulary: sum(probs) == 1.0
        probabilities = torch.softmax(scaled_logits, dim=-1)

        # Top-k selection
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

    # Top-1 candidate matches argmax of unscaled logits (since temperature scaling is monotonic)
    top_1_token_id = int(torch.argmax(final_logits).item())
    top_1_token_str = tokenizer.decode([top_1_token_id])

    candidates: List[CandidateToken] = []
    running_prob = 0.0

    for rank_idx in range(top_k):
        cid = int(top_indices[rank_idx].item())
        prob = float(top_probs[rank_idx].item())
        raw_logit = float(final_logits[cid].item())
        running_prob += prob

        candidates.append(
            CandidateToken(
                rank=rank_idx + 1,
                token_id=cid,
                token_str=tokenizer.decode([cid]),
                token_bytes_display=_format_token_display(cid, tokenizer),
                logit=round(raw_logit, 4),
                probability=round(prob, 6),
                cumulative_probability=round(running_prob, 6)
            )
        )

    explanation = (
        f"Temperature ({temperature}) scales the raw output logits before the softmax function: "
        "P(w_i) = exp(z_i / T) / sum_j exp(z_j / T). "
        "Higher temperature (> 1.0) flattens the distribution, producing more diverse and unpredictable tokens. "
        "Lower temperature (< 1.0) sharpens the probability peaks, favoring high-confidence tokens. "
        "Notice that temperature changes only the probability distribution during inference; "
        "it does NOT retrain or modify the model's weights."
    )

    return PredictNextTokenResponse(
        prompt=text,
        prompt_token_ids=prompt_token_ids,
        prompt_tokens=prompt_tokens,
        vocab_size=len(probabilities),
        temperature_used=temperature,
        top_k_used=top_k,
        temperature_explanation=explanation,
        explanation=explanation,
        explanation_origin="Educational explanation",
        top_candidates=candidates,
        top_1_token_id=top_1_token_id,
        top_1_token_str=top_1_token_str,
        total_probability_top_k=round(float(torch.sum(top_probs).item()), 6),
        data_origin="Real model output"
    )

def generate_text(
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.7,
    top_k: int = 50,
    seed: Optional[int] = None
) -> GenerateResponse:
    """Generates autoregressive continuation using real model sampling."""
    if not model_manager.is_loaded or model_manager.model is None or model_manager.tokenizer is None:
        model_manager.load_model()

    model = model_manager.model
    tokenizer = model_manager.tokenizer
    device = model_manager.device

    if model is None or tokenizer is None:
        raise RuntimeError("Model and tokenizer must be initialized.")

    if seed is not None:
        torch.manual_seed(seed)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    # Decide generation strategy based on temperature
    do_sample = temperature > 0.0
    effective_top_k = top_k if (top_k > 0 and do_sample) else None

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        if effective_top_k is not None:
            gen_kwargs["top_k"] = effective_top_k

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    generated_sequence = output_ids[0].tolist()
    new_token_ids = generated_sequence[prompt_len:]

    new_tokens_items: List[GeneratedTokenItem] = []
    for step, tid in enumerate(new_token_ids, start=1):
        new_tokens_items.append(
            GeneratedTokenItem(
                step=step,
                token_id=tid,
                token_str=tokenizer.decode([tid])
            )
        )

    continuation = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)

    finish_reason = "max_tokens"
    if new_token_ids and new_token_ids[-1] == tokenizer.eos_token_id:
        finish_reason = "eos_token"

    return GenerateResponse(
        prompt=prompt,
        prompt_token_count=prompt_len,
        generated_text=full_text,
        continuation=continuation,
        new_token_ids=new_token_ids,
        new_tokens=new_tokens_items,
        new_token_count=len(new_token_ids),
        finish_reason=finish_reason,
        settings={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "do_sample": do_sample,
            "seed": seed
        },
        data_origin="Real model output"
    )
