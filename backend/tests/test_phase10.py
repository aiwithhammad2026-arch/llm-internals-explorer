"""Tests for Phase 10: Logit Lens & Residual Stream Evolution.
By Hammad, AI Engineer.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import model_manager

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """Ensure the distilgpt2 model is loaded for real tests."""
    model_manager.load_model("distilgpt2")


def test_logit_lens_endpoint_success():
    """Verify POST /api/logit-lens returns valid real layers, metrics, and origins."""
    prompt = "The capital of France is"
    resp = client.post("/api/logit-lens", json={"text": prompt, "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()

    assert data["prompt"] == prompt
    assert data["token_count"] == 5
    assert data["selected_token_index"] == 4
    assert data["data_origin"] == "Real model output"
    assert data["explanation_origin"] == "Educational explanation"
    assert len(data["explanation"]) > 20

    # distilgpt2 has 6 blocks + 1 embedding output = 7 layer representations
    layers = data["layers"]
    assert len(layers) == 7

    assert layers[0]["layer"] == 0
    assert "Embedding" in layers[0]["layer_name"]
    assert layers[0]["cosine_sim_to_previous"] is None

    # Check each layer step structure
    for idx, step in enumerate(layers):
        assert step["layer"] == idx
        assert len(step["top_candidates"]) == 5
        assert step["target_rank"] >= 1
        assert 0.0 <= step["target_probability"] <= 1.0
        assert step["entropy"] >= 0.0
        assert step["residual_norm"] > 0.0
        if idx > 0:
            assert step["cosine_sim_to_previous"] is not None
            assert -1.0 <= step["cosine_sim_to_previous"] <= 1.0


def test_logit_lens_final_layer_matches_predict_next_token():
    """Layer 6 of the Logit Lens must mathematically match the model's top candidates."""
    prompt = "The capital of France is"
    
    # 1. Fetch Logit Lens
    ll_resp = client.post("/api/logit-lens", json={"text": prompt, "top_k": 5})
    assert ll_resp.status_code == 200
    ll_data = ll_resp.json()
    layer6_top = ll_data["layers"][-1]["top_candidates"]

    # 2. Fetch standard next-token prediction
    pred_resp = client.post("/api/predict-next-token", json={"text": prompt, "temperature": 1.0, "top_k": 5})
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    pred_top = pred_data["top_candidates"]

    # Compare Layer 6 against prediction candidates
    assert ll_data["final_top_token_id"] == pred_data["top_1_token_id"]
    assert ll_data["layers"][-1]["target_rank"] == 1

    for i in range(5):
        assert layer6_top[i]["token_id"] == pred_top[i]["token_id"]
        assert abs(layer6_top[i]["probability"] - pred_top[i]["probability"]) < 1e-4


def test_logit_lens_custom_token_position():
    """Verify Logit Lens can inspect any arbitrary token position in the sequence."""
    prompt = "The capital of France is"
    # Inspect position 0 ("The")
    resp = client.post("/api/logit-lens", json={"text": prompt, "token_index": 0, "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_token_index"] == 0
    assert len(data["layers"]) == 7


def test_logit_lens_validation_errors():
    """Verify validation boundaries: empty text, overlong text, and out-of-range position."""
    # 1. Empty text
    r1 = client.post("/api/logit-lens", json={"text": "   "})
    assert r1.status_code == 422

    # 2. Overlong text (>512 chars)
    r2 = client.post("/api/logit-lens", json={"text": "a" * 513})
    assert r2.status_code == 422

    # 3. Invalid token index out of range
    r3 = client.post("/api/logit-lens", json={"text": "test", "token_index": 99})
    assert r3.status_code == 400
    assert "out of sequence range" in r3.json()["detail"]
