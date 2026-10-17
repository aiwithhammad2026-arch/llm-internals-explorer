"""Phase 7 Tests: Debug Console & Model Comparison endpoints.
By Hammad, AI Engineer.

Validates:
1. Debug logs never contain API keys, environment variables, or file paths.
2. Compare-models endpoint rejects non-allow-listed models with HTTP 400.
3. Compare-models rejects incompatible model architectures.
4. Real-model evaluation verifies parameter counts (distilgpt2 ~82M, gpt2 ~124M)
   and ensures top-10 next-token probabilities are valid distributions.
"""

from unittest.mock import patch, MagicMock
import pytest
import torch
from fastapi.testclient import TestClient
from app.main import app
from app.services import debug_service

client = TestClient(app)

SECRET_KEY_FOR_TEST = "DUMMY_SECRET_KEY_FOR_DEBUG_TEST_8888"

def test_debug_logs_capture_and_clear():
    """Verify that requests are recorded in the ring buffer and can be cleared."""
    client.get("/api/health")
    response = client.get("/api/debug/logs")
    assert response.status_code == 200
    data = response.json()
    assert data["total_captured"] > 0
    assert len(data["logs"]) > 0

    # Clear logs
    r_clear = client.post("/api/debug/clear")
    assert r_clear.status_code == 200
    r_empty = client.get("/api/debug/logs")
    assert r_empty.json()["total_captured"] == 0

def test_debug_logs_never_leak_secrets_or_paths():
    """Verify that file paths, environment variables, and API keys are scrubbed."""
    debug_service.clear_debug_logs()

    # Manually record an entry with sensitive raw data
    raw_path_win = r"C:\Users\SecretUser\Desktop\taransfermore\backend\.env"
    raw_path_unix = "/home/secret/project/backend/.env"
    
    debug_service.record_request(
        endpoint="/api/test-debug",
        method="POST",
        status_code=200,
        duration_ms=12.34,
        warnings=[f"Failed to read file from {raw_path_win}"],
        errors=[f"Error accessing {raw_path_unix} with key {SECRET_KEY_FOR_TEST}"],
        gemini_status=f"error: {SECRET_KEY_FOR_TEST}",
    )

    response = client.get("/api/debug/logs")
    assert response.status_code == 200
    content_str = response.text

    # Assert that actual paths and keys were scrubbed
    assert raw_path_win not in content_str
    assert raw_path_unix not in content_str
    assert SECRET_KEY_FOR_TEST not in content_str
    assert "[FILE_PATH]" in content_str
    assert "[REDACTED_KEY]" in content_str

@pytest.mark.real_model
def test_debug_info_content_and_sanitization():
    """GET /api/debug/info returns versions and device info without paths or keys."""
    response = client.get("/api/debug/info")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "distilgpt2"
    assert "torch_version" in data
    assert "transformers_version" in data
    assert "device" in data
    assert data["data_origin"] == "Real model output"

    content_str = response.text
    assert "AI" + "za" not in content_str
    assert "\\" not in data["torch_version"]

def test_compare_models_rejects_unallowed_model():
    """POST /api/compare-models rejects any model not in ALLOWED_MODELS with 400."""
    response = client.post(
        "/api/compare-models",
        json={"text": "Test prompt", "model_a": "bert-base-uncased", "model_b": "gpt2"}
    )
    assert response.status_code == 400
    data = response.json()
    assert "not in the allowed models list" in str(data)

def test_compare_models_rejects_incompatible_architecture():
    """Rejects models whose config.model_type != 'gpt2'."""
    mock_model = MagicMock()
    mock_model.config.model_type = "llama"
    mock_tok = MagicMock()

    with patch("app.services.comparison_service._get_or_load_model", side_effect=ValueError("Model has architecture type 'llama'")):
        response = client.post(
            "/api/compare-models",
            json={"text": "Test prompt", "model_a": "distilgpt2", "model_b": "gpt2"}
        )
        assert response.status_code == 400
        assert "architecture type" in str(response.json())

@pytest.mark.real_model
def test_compare_models_real_models():
    """Real model test comparing distilgpt2 and gpt2 parameter counts, layers, and probabilities."""
    test_text = "The capital of France is"
    response = client.post(
        "/api/compare-models",
        json={"text": test_text, "model_a": "distilgpt2", "model_b": "gpt2"}
    )
    assert response.status_code == 200
    data = response.json()

    model_a = data["model_a"]
    model_b = data["model_b"]

    # Parameter count verification
    # distilgpt2 is ~82M (81,912,576), gpt2 is ~124M (124,439,808)
    assert 81_000_000 < model_a["param_count"] < 83_000_000
    assert 124_000_000 < model_b["param_count"] < 125_000_000

    # Layer count: distilgpt2 has 6 blocks, gpt2 has 12 blocks
    assert model_a["num_layers"] == 6
    assert model_b["num_layers"] == 12
    assert model_a["hidden_size"] == 768
    assert model_b["hidden_size"] == 768

    # Probability sanity checks: each candidate > 0, top-10 sum <= 1.0
    for cand in model_a["top_candidates"]:
        assert 0.0 < cand["probability"] <= 1.0
    assert 0.0 < model_a["total_top_k_prob"] <= 1.0

    for cand in model_b["top_candidates"]:
        assert 0.0 < cand["probability"] <= 1.0
    assert 0.0 < model_b["total_top_k_prob"] <= 1.0

    # Architectural differences summary
    diffs = data["differences"]
    assert diffs["layer_delta"] == 6
    assert diffs["parameter_delta"] > 40_000_000
    assert diffs["shared_hidden_size"] is True
    assert diffs["shared_vocab_size"] is True

def test_compare_models_cpu_memory_error_returns_503():
    """Verifies that a host CPU MemoryError when loading the comparison model returns 503 with remediation and no paths."""
    with patch("app.services.comparison_service.AutoModelForCausalLM.from_pretrained", side_effect=MemoryError("Unable to allocate 512MB for tensor storage in C:\\Users\\test\\cache")):
        # Ensure model is not already cached
        with patch.dict("app.services.comparison_service._comparison_cache", {}, clear=True):
            response = client.post(
                "/api/compare-models",
                json={"text": "Hello world", "model_a": "distilgpt2", "model_b": "gpt2"}
            )
            assert response.status_code == 503
            data = response.json()
            assert data["detail"]["error"] == "Out of Memory"
            assert "remediation" in data["detail"]
            assert len(data["detail"]["remediation"]) > 0
            # Path must be completely scrubbed
            assert "C:\\Users" not in str(data)
            assert "C:/" not in str(data)

def test_compare_models_runtime_oom_returns_503():
    """Verifies that PyTorch CPU allocator out-of-memory RuntimeError returns 503 with remediation."""
    with patch("app.services.comparison_service.AutoModelForCausalLM.from_pretrained", side_effect=RuntimeError("[CPU allocator] DefaultCPUAllocator: out of memory attempting to allocate 450MB")):
        with patch.dict("app.services.comparison_service._comparison_cache", {}, clear=True):
            response = client.post(
                "/api/compare-models",
                json={"text": "Hello world", "model_a": "distilgpt2", "model_b": "gpt2"}
            )
            assert response.status_code == 503
            data = response.json()
            assert data["detail"]["error"] == "Out of Memory"
            assert "remediation" in data["detail"]

def test_compare_models_cuda_oom_returns_503():
    """Verifies that torch.cuda.OutOfMemoryError returns 503 with remediation."""
    with patch("app.services.comparison_service.AutoModelForCausalLM.from_pretrained", side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory in /workspace/cuda_alloc.cpp")):
        with patch.dict("app.services.comparison_service._comparison_cache", {}, clear=True):
            response = client.post(
                "/api/compare-models",
                json={"text": "Hello world", "model_a": "distilgpt2", "model_b": "gpt2"}
            )
            assert response.status_code == 503
            data = response.json()
            assert data["detail"]["error"] == "Out of Memory"
            assert "remediation" in data["detail"]
            assert "/workspace" not in str(data)

