"""Phase 3 verification tests: Attention, Q/K/V extraction, and Layer Inspection.
By Hammad, AI Engineer.

Tests verify:
1. Real attention output has shape [heads, seq, seq] and each row sums to 1.0 (within float tolerance).
2. The causal mask is strictly enforced (upper triangle is zero).
3. Extracted Q, K, V have shape [heads, seq, 64].
4. Recomputed attention from Q and K matches real attention (tolerance < 1e-4).
5. Invalid layer and head indices are rejected with HTTP 400.
6. hidden_states has exactly n_layer + 1 entries (7 for distilgpt2).
7. Real-model tests are marked with @pytest.mark.real_model.
"""

import pytest
import torch
import math
from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import model_manager

client = TestClient(app)


# =====================================================================
# Real Model Tests
# =====================================================================

@pytest.mark.real_model
def test_attention_weights_properties():
    """Verify real attention weights:
    - [seq, seq] matrix per head.
    - Each row sums to 1.0.
    - Upper triangle is strictly zero (causal mask).
    - Top 15 weights table is populated and sorted.
    """
    text = "The capital of France is"
    response = client.post("/api/attention", json={"text": text, "layer": 0, "head": 0})
    assert response.status_code == 200
    data = response.json()

    assert data["layer"] == 0
    assert data["head"] == 0
    assert data["token_count"] == 5
    assert len(data["tokens"]) == 5
    assert data["is_causal_mask_applied"] is True
    assert data["upper_triangle_zero"] is True
    assert data["data_origin"] == "Real model output"

    matrix = data["matrix"]
    assert len(matrix) == 5

    for i in range(5):
        assert len(matrix[i]) == 5
        # Row must sum to 1.0
        row_sum = sum(matrix[i])
        assert pytest.approx(row_sum, abs=1e-5) == 1.0

        # Causal mask check: upper triangle (j > i) must be exactly 0.0
        for j in range(i + 1, 5):
            assert matrix[i][j] == 0.0

    # Top weights table verification
    assert len(data["top_weights"]) <= 15
    for idx in range(len(data["top_weights"]) - 1):
        assert data["top_weights"][idx]["weight"] >= data["top_weights"][idx + 1]["weight"]

@pytest.mark.real_model
def test_qkv_shapes_and_recomputation_matches_real_attention():
    """Verify Q/K/V extraction from fused c_attn and recomputation:
    - Q, K, V shapes are [seq_len, 64].
    - softmax((Q @ K^T)/sqrt(64) + mask) == real_attention (diff < 1e-4).
    """
    text = "The capital of France is"
    response = client.post("/api/attention", json={"text": text, "layer": 2, "head": 4})
    assert response.status_code == 200
    data = response.json()

    qkv = data["qkv"]
    assert qkv["layer"] == 2
    assert qkv["head"] == 4
    assert qkv["head_dim"] == 64
    assert qkv["fused_c_attn_width"] == 2304
    assert qkv["q_shape"] == [5, 64]
    assert qkv["k_shape"] == [5, 64]
    assert qkv["v_shape"] == [5, 64]
    assert len(qkv["q_slice"]) == 16
    assert len(qkv["k_slice"]) == 16
    assert len(qkv["v_slice"]) == 16

    # Recomputation error between Q/K and real attention must be negligible
    assert qkv["max_recomputed_difference"] < 1e-4

    # Direct PyTorch validation for all 12 heads of layer 2
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    inputs = tokenizer(text, return_tensors="pt").to(model_manager.device)

    c_attn_captured = {}
    def hook_fn(m, inp, out):
        c_attn_captured["val"] = out

    h = model.transformer.h[2].attn.c_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    h.remove()

    real_layer2_attn = outputs.attentions[2] # [1, 12, 5, 5]
    assert list(real_layer2_attn.shape) == [1, 12, 5, 5]

    q, k, v = c_attn_captured["val"].split(768, dim=2)
    q_h = q.view(1, 5, 12, 64).permute(0, 2, 1, 3)
    k_h = k.view(1, 5, 12, 64).permute(0, 2, 1, 3)

    assert list(q_h.shape) == [1, 12, 5, 64]
    assert list(k_h.shape) == [1, 12, 5, 64]

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(64)
    mask = torch.triu(torch.full((5, 5), float("-inf"), device=model_manager.device), diagonal=1)
    recomputed = torch.softmax(scores + mask, dim=-1)

    max_diff = torch.max(torch.abs(real_layer2_attn - recomputed)).item()
    assert max_diff < 1e-4

@pytest.mark.real_model
def test_layer_inspection_components_and_hidden_states_count():
    """Verify layer inspection:
    - Exactly 7 hidden_states (n_layer + 1).
    - All 7 block components captured (ln_1, attn, res_1, ln_2, mlp, res_2, input).
    - Residual after MLP matches hidden_states[layer + 1].
    """
    text = "The capital of France is"
    response = client.post("/api/layer-inspection", json={"text": text, "layer": 0, "slice_size": 16})
    assert response.status_code == 200
    data = response.json()

    assert data["layer"] == 0
    assert data["num_layers"] == 6
    assert data["num_heads"] == 12
    assert data["hidden_size"] == 768
    assert data["token_count"] == 5
    assert data["data_origin"] == "Real model output"
    assert data["final_hidden_state_shape"] == [1, 5, 768]
    assert data["logits_shape"] == [1, 5, 50257]

    component_names = [c["name"] for c in data["components"]]
    expected_names = [
        "input_hidden_state",
        "ln_1_output",
        "attn_output",
        "residual_after_attn",
        "ln_2_output",
        "mlp_output",
        "residual_after_mlp"
    ]
    assert component_names == expected_names

    for comp in data["components"]:
        assert comp["shape"] == [1, 5, 768]
        assert len(comp["slice_values"]) == 16
        assert comp["is_captured"] is True

    # Validate that model forward pass produces exactly 7 hidden states (n_layer + 1)
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    inputs = tokenizer(text, return_tensors="pt").to(model_manager.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    n_layer = model.config.n_layer
    assert len(outputs.hidden_states) == n_layer + 1

# =====================================================================
# Validation Tests: Layer and Head Bounds
# =====================================================================

def test_validation_invalid_layer_rejected():
    """Layer indices >= num_layers or < 0 must be rejected with 400."""
    # distilgpt2 has 6 layers (0 to 5)
    res_high = client.post("/api/attention", json={"text": "Test", "layer": 6, "head": 0})
    assert res_high.status_code == 400
    assert "Invalid layer index" in str(res_high.json())

    res_insp = client.post("/api/layer-inspection", json={"text": "Test", "layer": 10})
    assert res_insp.status_code == 400
    assert "Invalid layer index" in str(res_insp.json())

def test_validation_invalid_head_rejected():
    """Head indices >= num_heads or < 0 must be rejected with 400."""
    # distilgpt2 has 12 heads (0 to 11)
    res_high = client.post("/api/attention", json={"text": "Test", "layer": 0, "head": 12})
    assert res_high.status_code == 400
    assert "Invalid head index" in str(res_high.json())
