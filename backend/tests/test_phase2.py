"""Phase 2 verification tests: Embeddings, prediction, and generation.
By Hammad, AI Engineer.

Tests verify:
1. token embedding (wte) + positional embedding (wpe) equals the model's hidden_states[0] for the same input.
2. Next-token probabilities sum to 1.0 (within numerical tolerance).
3. The top-1 candidate matches argmax of the model's final-position logits.
4. Input validation rejects invalid temperature, top-k, empty input, and overlong generation.
5. All tests needing the real pretrained model are marked with @pytest.mark.real_model.
"""

import pytest
import torch
from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import model_manager

client = TestClient(app)


# =====================================================================
# Real Model Tests
# =====================================================================

@pytest.mark.real_model
def test_embedding_identity_wte_plus_wpe_equals_hidden_states():
    """Mathematical identity test:
    wte[input_ids] + wpe[positions] == hidden_states[0] in model eval mode.
    """
    text = "The capital of France is"
    response = client.post("/api/embeddings", json={"text": text, "slice_size": 16})
    assert response.status_code == 200
    data = response.json()

    assert data["token_count"] == 5
    assert data["hidden_dimension"] == 768
    assert data["slice_size"] == 16
    assert data["token_embedding_shape"] == [1, 5, 768]
    assert data["positional_embedding_shape"] == [1, 5, 768]
    assert data["combined_embedding_shape"] == [1, 5, 768]
    assert data["data_origin"] == "Real model output"
    assert len(data["tokens"]) == 5

    # Verify that slice combined equals slice wte + slice wpe
    for t in data["tokens"]:
        wte_s = t["token_embedding_slice"]
        wpe_s = t["positional_embedding_slice"]
        comb_s = t["combined_embedding_slice"]
        for i in range(len(wte_s)):
            assert pytest.approx(wte_s[i] + wpe_s[i], abs=1e-5) == comb_s[i]

    # Verify directly against PyTorch model's hidden_states[0]
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    inputs = tokenizer(text, return_tensors="pt").to(model_manager.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h0 = outputs.hidden_states[0] # Initial hidden state entering block 0

        pos = torch.arange(0, inputs["input_ids"].shape[1], dtype=torch.long, device=model_manager.device)
        wte = model.transformer.wte(inputs["input_ids"])
        wpe = model.transformer.wpe(pos).unsqueeze(0)
        expected_combined = wte + wpe

        # Mathematical equality check
        diff = torch.max(torch.abs(expected_combined - h0)).item()
        assert diff < 1e-5, f"Difference between wte+wpe and hidden_states[0] was {diff}"

@pytest.mark.real_model
def test_predict_next_token_probabilities_and_argmax():
    """Probability test:
    - Softmax probabilities across full vocabulary sum to 1.0.
    - Top-1 candidate matches argmax of the model's logits.
    """
    prompt = "The capital of France is"
    response = client.post("/api/predict-next-token", json={"text": prompt, "temperature": 1.0, "top_k": 10})
    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == prompt
    assert data["vocab_size"] == 50257
    assert data["temperature_used"] == 1.0
    assert data["top_k_used"] == 10
    assert data["data_origin"] == "Real model output"
    assert len(data["top_candidates"]) == 10

    # Independently compute with PyTorch
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(model_manager.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # 1. Probabilities sum to 1.0 (within numerical float32 tolerance)
        total_prob = torch.sum(probs).item()
        assert pytest.approx(total_prob, abs=1e-4) == 1.0

        # 2. Top-1 candidate matches argmax
        expected_top1_id = int(torch.argmax(logits).item())
        expected_top1_str = tokenizer.decode([expected_top1_id])

    assert data["top_1_token_id"] == expected_top1_id
    assert data["top_1_token_str"] == expected_top1_str
    assert data["top_candidates"][0]["token_id"] == expected_top1_id

    # Check that candidate probabilities are in descending order
    for i in range(len(data["top_candidates"]) - 1):
        assert data["top_candidates"][i]["probability"] >= data["top_candidates"][i + 1]["probability"]

@pytest.mark.real_model
def test_generation_produces_real_tokens():
    """Generation test:
    - Generates new tokens up to max_new_tokens.
    - Full text starts with prompt and contains continuation.
    """
    prompt = "The capital of France is"
    response = client.post("/api/generate", json={
        "prompt": prompt,
        "max_new_tokens": 10,
        "temperature": 0.7,
        "top_k": 50,
        "seed": 42
    })
    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == prompt
    assert data["new_token_count"] <= 10
    assert len(data["new_token_ids"]) == data["new_token_count"]
    assert len(data["new_tokens"]) == data["new_token_count"]
    assert data["generated_text"].startswith(prompt)
    assert data["data_origin"] == "Real model output"
    assert data["finish_reason"] in ["max_tokens", "eos_token"]

# =====================================================================
# Validation & Error Handling Tests
# =====================================================================

def test_validation_invalid_temperature():
    """Temperature <= 0 or > 5.0 must be rejected."""
    # Temperature 0.0 on predict-next-token must fail (gt=0.0)
    res_zero = client.post("/api/predict-next-token", json={"text": "Hello", "temperature": 0.0})
    assert res_zero.status_code == 422

    # Negative temperature must fail
    res_neg = client.post("/api/predict-next-token", json={"text": "Hello", "temperature": -1.0})
    assert res_neg.status_code == 422

    # Excessively high temperature must fail
    res_high = client.post("/api/predict-next-token", json={"text": "Hello", "temperature": 10.0})
    assert res_high.status_code == 422

def test_validation_invalid_top_k():
    """top_k < 1 or > 100 on predict-next-token must be rejected."""
    res_low = client.post("/api/predict-next-token", json={"text": "Hello", "top_k": 0})
    assert res_low.status_code == 422

    res_high = client.post("/api/predict-next-token", json={"text": "Hello", "top_k": 500})
    assert res_high.status_code == 422

def test_validation_overlong_generation_limit():
    """max_new_tokens > 128 must be rejected."""
    res = client.post("/api/generate", json={"prompt": "Hello", "max_new_tokens": 500})
    assert res.status_code == 422

def test_validation_empty_prompt_rejected():
    """Empty or whitespace-only prompts must be rejected with 422."""
    res_emb = client.post("/api/embeddings", json={"text": "   "})
    assert res_emb.status_code == 422

    res_pred = client.post("/api/predict-next-token", json={"text": ""})
    assert res_pred.status_code == 422

    res_gen = client.post("/api/generate", json={"prompt": "    "})
    assert res_gen.status_code == 422
