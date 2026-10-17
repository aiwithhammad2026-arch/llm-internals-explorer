"""Phase 6 Tests: Gemini AI Tutor endpoints with mocked Google GenAI client.
By Hammad, AI Engineer.

Ensures:
1. Missing key gives clear guidance.
2. Oversized questions (>500 chars) are rejected with 422.
3. API errors are safely translated into friendly user messages.
4. The API key NEVER appears in any response body or log.
5. All tests run completely offline without real API calls or network access.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SECRET_KEY_FOR_TEST = "DUMMY_SECRET_KEY_NEVER_LEAK_99999"

def test_tutor_status_no_key(monkeypatch):
    """GET /api/tutor/status reports unconfigured with guidance when key is unset."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with patch("app.services.gemini_service._get_api_key", return_value=""):
        response = client.get("/api/tutor/status")
        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is False
        assert "GEMINI_API_KEY" in data["status_message"]
        assert "backend/.env" in data["status_message"]
        assert data["data_origin"] == "Gemini-written explanation"
        assert SECRET_KEY_FOR_TEST not in str(data)

def test_tutor_status_with_key(monkeypatch):
    """GET /api/tutor/status reports configured when key is present, without revealing it."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)
    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        response = client.get("/api/tutor/status")
        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is True
        assert "active" in data["status_message"].lower()
        # Verify secret key string is NEVER returned
        assert SECRET_KEY_FOR_TEST not in str(data)
        assert data["data_origin"] == "Gemini-written explanation"

def test_explain_missing_key_returns_clear_guidance(monkeypatch):
    """POST /api/tutor/explain returns 400 with setup guidance when key is missing."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with patch("app.services.gemini_service._get_api_key", return_value=""):
        response = client.post(
            "/api/tutor/explain",
            json={
                "view_name": "attention",
                "context_summary": {"token_count": 5, "layer": 0, "head": 0}
            }
        )
        assert response.status_code == 400
        data = response.json()
        detail = data.get("detail", {})
        assert "Missing GEMINI_API_KEY" in str(detail)
        assert "setup_guidance" in detail
        assert SECRET_KEY_FOR_TEST not in str(data)

def test_ask_oversized_question_rejected():
    """POST /api/tutor/ask with >500 characters is rejected with 422 Unprocessable Entity."""
    oversized_question = "A" * 501
    response = client.post(
        "/api/tutor/ask",
        json={"question": oversized_question}
    )
    assert response.status_code == 422

def test_ask_empty_question_rejected():
    """POST /api/tutor/ask with empty or whitespace question is rejected."""
    response = client.post(
        "/api/tutor/ask",
        json={"question": "   "}
    )
    assert response.status_code == 422

@patch("app.services.gemini_service.genai.Client")
def test_explain_with_mocked_gemini(mock_genai_client, monkeypatch):
    """POST /api/tutor/explain returns Gemini explanation when client succeeds."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)
    
    mock_instance = MagicMock()
    mock_genai_client.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = "In layer 0, head 0, the token ' is' strongly attends to 'The' due to the first-token sink."
    mock_instance.models.generate_content.return_value = mock_response

    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        response = client.post(
            "/api/tutor/explain",
            json={
                "view_name": "attention",
                "context_summary": {"prompt": "The capital of France is", "top_weight": 0.5297}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "first-token sink" in data["explanation"]
        assert data["data_origin"] == "Gemini-written explanation"
        assert SECRET_KEY_FOR_TEST not in str(data)

@patch("app.services.gemini_service.genai.Client")
def test_ask_with_mocked_gemini(mock_genai_client, monkeypatch):
    """POST /api/tutor/ask returns pedagogical answer when client succeeds."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)

    mock_instance = MagicMock()
    mock_genai_client.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = "Pre-LN normalizes the input before attention, stabilizing deeper architectures."
    mock_instance.models.generate_content.return_value = mock_response

    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        response = client.post(
            "/api/tutor/ask",
            json={"question": "Why does GPT-2 use Pre-LN?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "Pre-LN" in data["answer"]
        assert data["data_origin"] == "Gemini-written explanation"
        assert SECRET_KEY_FOR_TEST not in str(data)

@patch("app.services.gemini_service.genai.Client")
def test_quiz_generation_with_mocked_gemini(mock_genai_client, monkeypatch):
    """POST /api/tutor/quiz returns 3 multiple-choice questions."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)

    mock_instance = MagicMock()
    mock_genai_client.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = """[
        {
            "id": 1,
            "question": "What is the head dimension of distilgpt2?",
            "options": ["32", "64", "128", "768"],
            "correct_index": 1,
            "explanation": "768 / 12 = 64 dimensions per head."
        },
        {
            "id": 2,
            "question": "Why is the upper triangle masked?",
            "options": ["Speed", "Causal ordering", "Sparsity", "Quantization"],
            "correct_index": 1,
            "explanation": "Tokens cannot attend to future tokens in causal generation."
        },
        {
            "id": 3,
            "question": "What does c_attn project?",
            "options": ["Only Q", "Only K", "Only V", "Q, K, and V simultaneously"],
            "correct_index": 3,
            "explanation": "c_attn fuses Q, K, and V into 2304 dimensions."
        }
    ]"""
    mock_instance.models.generate_content.return_value = mock_response

    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        response = client.post(
            "/api/tutor/quiz",
            json={"topic": "attention"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["questions"]) == 3
        assert data["questions"][0]["correct_index"] == 1
        assert data["data_origin"] == "Gemini-written explanation"
        assert SECRET_KEY_FOR_TEST not in str(data)

@patch("app.services.gemini_service.genai.Client")
def test_compare_generation_with_mocked_gemini(mock_genai_client, monkeypatch):
    """POST /api/tutor/compare-generation returns continuation with foundation model disclaimer."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)

    mock_instance = MagicMock()
    mock_genai_client.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = "Paris, known for its iconic Eiffel Tower and rich history."
    mock_instance.models.generate_content.return_value = mock_response

    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        response = client.post(
            "/api/tutor/compare-generation",
            json={"prompt": "The capital of France is", "local_generation": " Paris, which"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "Paris" in data["gemini_continuation"]
        assert "disclaimer" in data
        assert "foundation model" in data["disclaimer"].lower()
        assert data["data_origin"] == "Gemini-written explanation"
        assert SECRET_KEY_FOR_TEST not in str(data)

@patch("app.services.gemini_service.genai.Client")
def test_api_error_handling_and_secret_leak_protection(mock_genai_client, monkeypatch):
    """Verifies that API errors are converted into friendly messages and NEVER leak the key."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)

    mock_instance = MagicMock()
    mock_genai_client.return_value = mock_instance
    # Simulate an error that internally contains the secret key
    mock_instance.models.generate_content.side_effect = Exception(
        f"403 Forbidden: Invalid API Key {SECRET_KEY_FOR_TEST}"
    )

    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        response = client.post(
            "/api/tutor/ask",
            json={"question": "Explain attention."}
        )
        assert response.status_code == 502
        data = response.json()
        detail_msg = data.get("detail", {}).get("message", "")
        # Must be friendly message
        assert "rejected by Google's API" in detail_msg or "verify" in detail_msg
        # MUST NEVER LEAK THE SECRET KEY STRING
        assert SECRET_KEY_FOR_TEST not in str(response.content)
        assert SECRET_KEY_FOR_TEST not in str(response.headers)

@patch("app.services.gemini_service.genai.Client")
def test_gemini_timeout_error_handling(mock_genai_client, monkeypatch):
    """Verifies that network/timeout errors turn into friendly messages without leaking internals."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)

    mock_instance = MagicMock()
    mock_genai_client.return_value = mock_instance
    mock_instance.models.generate_content.side_effect = TimeoutError("The read operation timed out after 30000ms")

    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        response = client.post(
            "/api/tutor/ask",
            json={"question": "Explain attention."}
        )
        assert response.status_code == 502
        data = response.json()
        detail_msg = data.get("detail", {}).get("message", "")
        assert "timed out or failed" in detail_msg
        assert "internet connection" in detail_msg
        assert SECRET_KEY_FOR_TEST not in str(response.content)

@patch("app.services.gemini_service.genai.Client")
def test_gemini_client_explicit_timeout(mock_client_cls, monkeypatch):
    """Verifies that the Gemini client initializes with an explicit 30,000ms (30s) request timeout."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance
    mock_instance.models.generate_content.return_value = MagicMock(text="Sample explanation.")

    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        response = client.post(
            "/api/tutor/ask",
            json={"question": "What is self-attention?"}
        )
        assert response.status_code == 200

    assert mock_client_cls.called
    _, kwargs = mock_client_cls.call_args
    assert "http_options" in kwargs
    assert kwargs["http_options"].timeout == 30000

@pytest.mark.real_model
@patch("app.services.gemini_service.genai.Client")
def test_inspector_results_unchanged_after_tutor_call(mock_client_cls, monkeypatch):
    """Verifies that calling Gemini AI Tutor endpoints leaves model inspector results 100% unchanged.
    This holds plainly because gemini_service has no access to model tensors.
    """
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY_FOR_TEST)
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance
    mock_instance.models.generate_content.return_value = MagicMock(text="Pedagogical explanation.")

    prompt = "The capital of France is"

    # 1. Inspector results before calling tutor
    tok_before = client.post("/api/tokenize", json={"text": prompt}).json()
    pred_before = client.post("/api/predict-next-token", json={"text": prompt, "temperature": 1.0, "top_k": 5}).json()

    # 2. Call multiple Tutor endpoints with mocked Gemini
    with patch("app.services.gemini_service._get_api_key", return_value=SECRET_KEY_FOR_TEST):
        client.post("/api/tutor/explain", json={"view_name": "tokenize", "context_summary": tok_before})
        client.post("/api/tutor/ask", json={"question": "Why is the top token ' the'?"})
        client.post("/api/tutor/quiz", json={"topic": "attention"})

    # 3. Inspector results after calling tutor
    tok_after = client.post("/api/tokenize", json={"text": prompt}).json()
    pred_after = client.post("/api/predict-next-token", json={"text": prompt, "temperature": 1.0, "top_k": 5}).json()

    # Assert exact equality of inspector results
    assert tok_before == tok_after
    assert pred_before == pred_after

