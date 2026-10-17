"""Tests for Phase 11: Attention Rollout & Token Attribution Flow.
By Hammad, AI Engineer.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import model_manager

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """Ensure distilgpt2 model is loaded."""
    model_manager.load_model("distilgpt2")


def test_attention_rollout_endpoint_success():
    """Verify POST /api/attention-rollout returns 6 layers and valid attribution metrics."""
    prompt = "The capital of France is"
    resp = client.post("/api/attention-rollout", json={"text": prompt})
    assert resp.status_code == 200
    data = resp.json()

    assert data["prompt"] == prompt
    assert data["token_count"] == 5
    assert len(data["tokens"]) == 5
    assert data["data_origin"] == "Real model output"
    assert data["explanation_origin"] == "Educational explanation"

    # distilgpt2 has 6 layers
    assert len(data["layer_steps"]) == 6
    assert len(data["final_attribution"]) == 5

    # Check top attributed token
    assert data["top_attributed_token"] in data["tokens"]
    assert 0.0 < data["top_attributed_score"] <= 1.0


def test_attention_rollout_row_stochasticity_and_causality():
    """Each rollout matrix row must sum to 1.0 and upper triangle must be strictly zero."""
    prompt = "The capital of France is"
    resp = client.post("/api/attention-rollout", json={"text": prompt, "residual_weight": 0.5})
    assert resp.status_code == 200
    data = resp.json()

    seq_len = data["token_count"]

    for step in data["layer_steps"]:
        matrix = step["rollout_matrix"]
        assert len(matrix) == seq_len

        for r_idx, row in enumerate(matrix):
            assert len(row) == seq_len
            # Row sum must equal 1.0 within numerical float tolerance
            assert abs(sum(row) - 1.0) < 1e-4

            # Upper triangle (j > r_idx) must be strictly 0.0 (causal mask preservation)
            for c_idx in range(r_idx + 1, seq_len):
                assert row[c_idx] == 0.0

    # Final attribution must sum to 1.0
    total_attribution = sum(item["attribution_score"] for item in data["final_attribution"])
    assert abs(total_attribution - 1.0) < 1e-4


def test_attention_rollout_head_reduction_strategies():
    """Verify mean, max, and min head reductions produce valid rollout representations."""
    prompt = "The capital of France is"
    for strategy in ["mean", "max", "min"]:
        resp = client.post(
            "/api/attention-rollout",
            json={"text": prompt, "head_reduction": strategy, "residual_weight": 0.6}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["head_reduction"] == strategy
        assert len(data["layer_steps"]) == 6


def test_attention_rollout_validation_boundaries():
    """Verify validation: empty text, overlong text, and out-of-range residual weight."""
    # 1. Empty text
    r1 = client.post("/api/attention-rollout", json={"text": "   "})
    assert r1.status_code == 422

    # 2. Overlong text (>512 chars)
    r2 = client.post("/api/attention-rollout", json={"text": "x" * 513})
    assert r2.status_code == 422

    # 3. Residual weight out of bounds
    r3 = client.post("/api/attention-rollout", json={"text": "test", "residual_weight": 1.5})
    assert r3.status_code == 422
