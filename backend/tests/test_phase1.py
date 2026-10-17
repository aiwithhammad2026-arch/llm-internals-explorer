"""Phase 1 verification tests.
By Hammad, AI Engineer.

Tests verify:
1. /api/health returns healthy status and active device.
2. /api/model-info returns 6 layers, 12 heads, 768 hidden size for distilgpt2.
3. /api/tokenize returns ground-truth token IDs [15496, 995] for 'Hello world'.
4. Tokenization request validation rejects empty strings and overlong input.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import ModelManager

client = TestClient(app)

# =====================================================================
# Real Model Tests
# =====================================================================

@pytest.mark.real_model
def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "distilgpt2"
    assert "cpu" in data["device"] or "cuda" in data["device"]
    assert data["data_origin"] == "Real model output"

@pytest.mark.real_model
def test_model_info_distilgpt2():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "distilgpt2"
    assert data["num_layers"] == 6
    assert data["num_heads"] == 12
    assert data["hidden_size"] == 768
    assert data["vocab_size"] == 50257
    assert data["max_position_embeddings"] == 1024
    assert data["attn_implementation"] == "eager"
    assert data["trust_remote_code"] is False
    assert data["data_origin"] == "Real model output"

@pytest.mark.real_model
def test_tokenize_hello_world():
    response = client.post("/api/tokenize", json={"text": "Hello world"})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["token_count"] == 2
    # Exact real token IDs for GPT-2 BPE on 'Hello world'
    assert data["token_ids"] == [15496, 995]
    assert data["input_tensor_shape"] == [1, 2]
    assert len(data["tokens"]) == 2
    assert data["tokens"][0]["token_id"] == 15496
    assert data["tokens"][0]["token_str"] == "Hello"
    assert data["tokens"][1]["token_id"] == 995
    assert data["tokens"][1]["token_str"] == " world"
    assert data["tokens"][1]["token_bytes_display"] == "␣world"
    assert data["data_origin"] == "Real model output"

# =====================================================================
# Mocked / Offline Unit Tests (no model download or loading required)
# =====================================================================

def test_tokenize_empty_rejected():
    response = client.post("/api/tokenize", json={"text": "   "})
    assert response.status_code == 422

def test_tokenize_overlong_rejected():
    overlong = "a" * 513
    response = client.post("/api/tokenize", json={"text": overlong})
    assert response.status_code == 422

def test_model_loading_failure_unallowed_model():
    """Model loader rejects unapproved models to prevent unsafe code execution."""
    mgr = ModelManager()
    with pytest.raises(ValueError) as excinfo:
        mgr.load_model("unauthorized/malicious-model")
    assert "not in the allowed models list" in str(excinfo.value)
    assert mgr.is_loaded is False

def test_model_loading_failure_download_error():
    """Model loader produces actionable setup guidance when a download fails."""
    mgr = ModelManager()
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=Exception("Connection timed out")):
        with pytest.raises(RuntimeError) as excinfo:
            mgr.load_model("distilgpt2")
        assert "Could not download or load the model" in str(excinfo.value)
        assert mgr.setup_guidance is not None
        assert "huggingface" in mgr.setup_guidance.lower() or "internet" in mgr.setup_guidance.lower()
        assert mgr.is_loaded is False

