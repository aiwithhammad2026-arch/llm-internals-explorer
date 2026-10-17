import os
import json
import torch
from fastapi.testclient import TestClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.main import app
from app.model_loader import model_manager

def main():
    client = TestClient(app)
    out_dir = os.path.join(os.path.dirname(__file__), "example_outputs")
    os.makedirs(out_dir, exist_ok=True)

    test_prompt = "The capital of France is"

    print("=== Step 1: Run Endpoints and Save Example Outputs ===")
    
    # 1. Health
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    with open(os.path.join(out_dir, "health.json"), "w", encoding="utf-8") as f:
        json.dump(r_health.json(), f, indent=2)
    print("Saved health.json")

    # 2. Model Info
    r_model_info = client.get("/api/model-info")
    assert r_model_info.status_code == 200
    with open(os.path.join(out_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(r_model_info.json(), f, indent=2)
    print("Saved model_info.json")

    # 3. Tokenize
    r_tokenize = client.post("/api/tokenize", json={"text": test_prompt})
    assert r_tokenize.status_code == 200
    with open(os.path.join(out_dir, "tokenize.json"), "w", encoding="utf-8") as f:
        json.dump(r_tokenize.json(), f, indent=2)
    print("Saved tokenize.json")

    # 4. Embeddings
    r_embeddings = client.post("/api/embeddings", json={"text": test_prompt, "slice_size": 16})
    assert r_embeddings.status_code == 200
    emb_data = r_embeddings.json()
    with open(os.path.join(out_dir, "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(emb_data, f, indent=2)
    print("Saved embeddings.json")

    # 5. Predict Next Token
    r_predict = client.post("/api/predict-next-token", json={"text": test_prompt, "temperature": 1.0, "top_k": 10})
    assert r_predict.status_code == 200
    with open(os.path.join(out_dir, "predict_next_token.json"), "w", encoding="utf-8") as f:
        json.dump(r_predict.json(), f, indent=2)
    print("Saved predict_next_token.json")

    # 6. Generate
    r_generate = client.post("/api/generate", json={"prompt": test_prompt, "max_new_tokens": 10, "temperature": 0.7, "top_k": 50, "seed": 42})
    assert r_generate.status_code == 200
    with open(os.path.join(out_dir, "generate.json"), "w", encoding="utf-8") as f:
        json.dump(r_generate.json(), f, indent=2)
    print("Saved generate.json")

    # 7. Attention
    r_attention = client.post("/api/attention", json={"text": test_prompt, "layer": 0, "head": 0})
    assert r_attention.status_code == 200
    with open(os.path.join(out_dir, "attention.json"), "w", encoding="utf-8") as f:
        json.dump(r_attention.json(), f, indent=2)
    print("Saved attention.json")

    # 8. Layer Inspection
    r_inspection = client.post("/api/layer-inspection", json={"text": test_prompt, "layer": 0, "slice_size": 16, "token_index": 0})
    assert r_inspection.status_code == 200
    insp_data = r_inspection.json()
    with open(os.path.join(out_dir, "layer_inspection.json"), "w", encoding="utf-8") as f:
        json.dump(insp_data, f, indent=2)
    print("Saved layer_inspection.json")

    # 9. Tutor Status
    r_tutor_status = client.get("/api/tutor/status")
    assert r_tutor_status.status_code == 200
    with open(os.path.join(out_dir, "tutor_status.json"), "w", encoding="utf-8") as f:
        json.dump(r_tutor_status.json(), f, indent=2)
    print("Saved tutor_status.json")

    # 10. Debug Info
    r_debug_info = client.get("/api/debug/info")
    assert r_debug_info.status_code == 200
    with open(os.path.join(out_dir, "debug_info.json"), "w", encoding="utf-8") as f:
        json.dump(r_debug_info.json(), f, indent=2)
    print("Saved debug_info.json")

    # 11. Debug Logs
    r_debug_logs = client.get("/api/debug/logs")
    assert r_debug_logs.status_code == 200
    with open(os.path.join(out_dir, "debug_logs.json"), "w", encoding="utf-8") as f:
        json.dump(r_debug_logs.json(), f, indent=2)
    print("Saved debug_logs.json")

    # 12. Compare Models (distilgpt2 vs gpt2)
    r_compare = client.post("/api/compare-models", json={"text": test_prompt, "model_a": "distilgpt2", "model_b": "gpt2"})
    assert r_compare.status_code == 200
    with open(os.path.join(out_dir, "compare_models.json"), "w", encoding="utf-8") as f:
        json.dump(r_compare.json(), f, indent=2)
    print("Saved compare_models.json")

    # 13. Logit Lens (Phase 10)
    r_logit_lens = client.post("/api/logit-lens", json={"text": test_prompt, "top_k": 5})
    assert r_logit_lens.status_code == 200
    with open(os.path.join(out_dir, "logit_lens.json"), "w", encoding="utf-8") as f:
        json.dump(r_logit_lens.json(), f, indent=2)
    print("Saved logit_lens.json")

    print("\n=== Step 2: Numerical Consistency Verification ===")
    # Token 0 combined embedding slice from /api/embeddings
    emb_token0_slice = emb_data["tokens"][0]["combined_embedding_slice"]
    print(f"/api/embeddings token 0 combined slice (first 16 values):\n{emb_token0_slice}")

    # Token 0 input_hidden_state from /api/layer-inspection (layer 0)
    # Finding component input_hidden_state:
    input_hidden_comp = next(c for c in insp_data["components"] if c["name"] == "input_hidden_state")
    insp_token0_slice = input_hidden_comp["slice_values"]
    print(f"\n/api/layer-inspection (layer 0) input_hidden_state (first 16 values):\n{insp_token0_slice}")

    # Direct model call
    if not model_manager.is_loaded or model_manager.model is None or model_manager.tokenizer is None:
        model_manager.load_model()
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    inputs = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        direct_outputs = model(**inputs, output_hidden_states=True)
    direct_hidden_0_slice = [round(float(x), 6) for x in direct_outputs.hidden_states[0][0, 0, :16].tolist()]
    print(f"\nDirect model call outputs.hidden_states[0][0, 0, :16] (first 16 values):\n{direct_hidden_0_slice}")

    # Compute differences
    diff_emb_insp = max(abs(a - b) for a, b in zip(emb_token0_slice, insp_token0_slice))
    diff_emb_direct = max(abs(a - b) for a, b in zip(emb_token0_slice, direct_hidden_0_slice))
    diff_insp_direct = max(abs(a - b) for a, b in zip(insp_token0_slice, direct_hidden_0_slice))

    print(f"\nMax difference (|embeddings - layer-inspection|): {diff_emb_insp:.10e}")
    print(f"Max difference (|embeddings - direct model|):      {diff_emb_direct:.10e}")
    print(f"Max difference (|layer-inspection - direct model|): {diff_insp_direct:.10e}")

    assert diff_emb_insp < 1e-5, f"Embedding vs Inspection difference {diff_emb_insp} >= 1e-5"
    assert diff_emb_direct < 1e-5, f"Embedding vs Direct difference {diff_emb_direct} >= 1e-5"
    assert diff_insp_direct < 1e-5, f"Inspection vs Direct difference {diff_insp_direct} >= 1e-5"

    print("\nSUCCESS: All 3 sources match within 1e-5!")

if __name__ == "__main__":
    main()
