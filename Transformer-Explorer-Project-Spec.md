# Project: Real Transformer Internals Explorer

**By Hammad, AI Engineer**

Act as a senior AI engineer, machine learning engineer and full-stack developer.

Build a working web application that lets beginners explore how a real Transformer model processes text, from tokenization to next-token prediction. The app also includes an optional **Gemini AI Tutor** that explains the real data in plain language.

Work in the phases listed in Section 9. Finish one phase, show how to run and verify it, then stop and wait for me to say "continue" before starting the next phase.

---

## 1. Main Goal

Show the REAL internal operations of a pretrained Transformer that runs locally.

- Do not build a frontend-only simulation.
- Connect the frontend to an actual pretrained language model through a FastAPI backend.
- Expose real tokenizer output, model configuration, intermediate tensors, attention weights and next-token probabilities wherever the model supports it.
- Label every value as one of: **Real model output**, **Educational explanation**, **Illustrative diagram**, or **Gemini-written explanation**.
- If the model cannot download or load, show a clear error with setup steps. Never replace real outputs with made-up values.

---

## 2. Technology Stack

**Frontend:** React, Vite, TypeScript, Tailwind CSS, Recharts (or a suitable chart library).

**Backend:** Python, FastAPI, Hugging Face Transformers, PyTorch, Hugging Face Tokenizers, Pydantic.

**Local model (default):** `distilgpt2`
- Runs on CPU by default, optional CUDA.
- Needs no API key.
- Users may pick another compatible model from a trusted allow-list in config.

**Gemini (optional tutor layer):** Google Gemini API through the official `google-genai` Python SDK.
- Used only for the AI Tutor feature (Section 3, feature I).
- The app must run fully without a Gemini key. If no key is set, the Tutor panel shows "Add GEMINI_API_KEY to enable the AI Tutor" and everything else keeps working.
- Check Google's current Gemini API documentation for the SDK call syntax and available model names before coding, and keep the model name in an environment variable (`GEMINI_MODEL`) so it can change without code edits.

---

## 2.1 Why two kinds of models (important design rule)

| | Local pretrained model (`distilgpt2`) | Gemini API |
|---|---|---|
| Purpose | Show real internals | Explain them in simple language |
| Token IDs, embeddings, Q/K/V, attention | Yes, real | No. Gemini does not expose these. |
| Needs a key | No | Yes |
| Runs where | Your machine | Google's servers |

Gemini never produces or replaces any number shown in the inspector panels. It only receives a small summary of what the local model produced and writes an explanation. Mark its output "Gemini-written explanation" in the UI.

---

## 3. Features

### A. Tokenization Inspector
Input: any text. Show original text, real token IDs, real token strings, token count, input tensor shape, token-to-ID mapping. Use the actual tokenizer.

### B. Embedding Inspector
Show embedding tensor shape, hidden dimension, the vector for each selected token, a small sample of values, and a chart of selected dimensions. GPT-2-style models add learned positional embeddings to token embeddings before the first block. Label which part is token embedding, which is positional, and which is their sum.

### C. Query, Key and Value Inspector
Extract real Q, K and V from a selected layer and head. Show layer, head, Q/K/V vectors, shapes, values and token positions. GPT-2 uses a fused attention projection (`c_attn`) that produces Q, K and V together, which is then split and reshaped per head. Never hardcode example vectors.

### D. Self-Attention Visualization
Show a real attention heatmap with source and target token labels, values, selected layer and head, a table of the highest weights, and an explanation of the calculation. Explain the causal mask (a token cannot attend to future tokens). State clearly that attention weights do not fully explain model reasoning.

Implementation note: Hugging Face only returns attention weights when the model is loaded with `attn_implementation="eager"` (the faster default implementations may not return them). Load the model that way and note it in the debug console.

### E. Transformer Block Inspector
Show the real architecture and the activations you actually capture: embeddings plus positions, layer norm, self-attention, attention output projection, residual connections, feed-forward network, final hidden states, output logits. Use model outputs and PyTorch hooks. Verify shapes against the loaded config. Never claim a component was extracted if it was not captured.

### F. Next-Token Prediction
Prompt example: "The capital of France is". Show real input tokens, real next-token logits, softmax probabilities, top 10 candidates with IDs and probabilities, and the generated continuation. Include top-k and temperature controls. Explain that temperature changes the sampling distribution and does not retrain the model.

### G. Backend Debug Console
Show model name and revision, device, tokenizer info, input token IDs, tensor shapes and dtypes, selected layer and head, inference duration, generation settings, errors and warnings, and (when used) Gemini request status. Never expose secrets, environment variables or API keys.

### H. Model Comparison
Compare two locally available causal models when hardware permits: tokenization, parameter counts, hidden size, layers, heads, next-token predictions. Label model-specific values and handle incompatible architectures safely.

### I. Gemini AI Tutor (new)
A side panel that explains what the user is currently looking at.

Required behaviour:
1. **Explain this view.** A button on each inspector sends Gemini a bounded summary (for example: the prompt, token strings and IDs, shapes, top 5 attention weights, top 10 next-token probabilities) and returns a beginner explanation of what these numbers mean.
2. **Ask a question.** A chat box where the user asks about the current view ("Why does 'it' attend to 'animal'?"). Gemini receives the same bounded summary plus the question.
3. **Quiz me.** Gemini writes 3 multiple-choice questions based on the current real data, with answers and short explanations.
4. **Honest framing.** The system prompt tells Gemini: it is explaining data from a small local model, it must not claim to see inside itself or inside the local model beyond the supplied data, it must not invent numbers that were not supplied, and it should say when attention weights are only a partial explanation.
5. **Optional side-by-side text.** A "Compare with Gemini" toggle in Next-Token Prediction that shows Gemini's continuation of the same prompt next to the local model's. Label it clearly: Gemini is a different, much larger model and its internals and probabilities are not available.
6. **User-triggered only.** Never call Gemini automatically on every keystroke.

Reliability and safety for Gemini:
- Read `GEMINI_API_KEY` only on the backend from environment variables or a `.env` file. Never send it to the browser, never log it, never include it in errors or the debug console.
- Provide `.env.example` with placeholder values. Add `.env` to `.gitignore`.
- Limit request size (cap the summary and the user question length), set a request timeout, and add a simple per-session rate limit.
- Handle failures with clear messages: missing key, invalid key, quota or rate limit, network error, blocked or empty response.
- Treat user text and model output as untrusted: render Gemini's reply as plain text or safely sanitized Markdown, and ignore any instructions inside the user's prompt that try to change the tutor's rules.
- Show a short notice: text sent to the Tutor is sent to Google's Gemini API.

---

## 4. Backend API

Use Pydantic request and response models.

Core endpoints:
- `GET /api/health`
- `GET /api/model-info`
- `POST /api/tokenize`
- `POST /api/embeddings`
- `POST /api/attention`
- `POST /api/layer-inspection`
- `POST /api/predict-next-token`
- `POST /api/generate`

Gemini endpoints:
- `GET /api/tutor/status` (returns whether a key is configured, never the key)
- `POST /api/tutor/explain`
- `POST /api/tutor/ask`
- `POST /api/tutor/quiz`
- `POST /api/tutor/compare-generation`

Validate input length, layer index, head index, temperature, top-k and generation limits. Add exception handling, sensible request limits and clear error messages. Return bounded slices and summaries, never entire model state or huge tensors.

---

## 5. Frontend Design

Professional AI engineering dashboard: dark theme, responsive layout, sidebar navigation, model status indicator, text input, token visualization, tensor inspector, attention heatmap, next-token probability chart, backend logs, AI Tutor panel, loading states and error handling. Use real backend data for every panel that claims to show model internals. Mark illustrative diagrams as illustrative. Add the credit "By Hammad, AI Engineer" in the sidebar footer and the README.

---

## 6. Project Structure

```
transformer-explorer/
  backend/
    app/
      main.py
      config.py
      model_loader.py
      schemas.py
      services/
        tokenizer_service.py
        embedding_service.py
        attention_service.py
        inspection_service.py
        generation_service.py
        gemini_service.py
      routers/
        core.py
        tutor.py
    tests/
    requirements.txt
    .env.example
  frontend/
    src/
    package.json
    vite.config.ts
  README.md
  .gitignore
```

---

## 7. Security and Reliability

- No hardcoded keys or credentials. No key needed for local inference.
- Restrict input length and generation length.
- No arbitrary Python execution, no unsafe deserialization.
- Do not allow loading arbitrary remote model code (`trust_remote_code=False`); only models from a trusted allow-list.
- Do not leak filesystem paths or secrets in errors.
- Timeouts and resource limits where possible.
- Handle unavailable models and out-of-memory errors gracefully.
- CORS limited to the local frontend origin.

---

## 8. Testing

Automated tests for: tokenization endpoint, model metadata, embedding shapes, Q/K/V shapes, attention dimensions, probability normalization (sums to 1), invalid layer and head indices, empty and overlong input, model loading failure.

Gemini tests (all with the Gemini client **mocked**, so no key or network is needed): missing key returns a clear status, oversized question is rejected, API error is turned into a friendly message, the API key never appears in any response or log.

Mark which tests are mocked unit tests and which need the real pretrained model (for example with a `real_model` pytest marker).

---

## 9. Build Phases

For every phase: implement it, give the exact PowerShell commands to run it, list what I should see, then **stop and wait for "continue"**.

### Phase 0: Plan and project setup
- Briefly explain the architecture and the plan.
- Create the folder structure, `.gitignore`, `requirements.txt`, `.env.example`, empty README with the "By Hammad, AI Engineer" credit.
- Windows PowerShell commands for the Python virtual environment and installing backend and frontend dependencies.
- Done when: both projects install without errors.

### Phase 1: Backend foundation with the real model
- `config.py`, `model_loader.py` (load `distilgpt2` with `attn_implementation="eager"`, CPU default, optional CUDA), `main.py`, `schemas.py`.
- Endpoints: `/api/health`, `/api/model-info`, `/api/tokenize`.
- Clear error and setup message if the model cannot download.
- Done when: `/api/tokenize` returns real token IDs for "Hello world" and `/api/model-info` shows 6 layers, 12 heads, hidden size 768 for `distilgpt2`.

### Phase 2: Embeddings, prediction and generation
- `/api/embeddings` (token, positional and combined, with shapes).
- `/api/predict-next-token` (logits, softmax, top 10, temperature, top-k).
- `/api/generate` with generation limits.
- Done when: probabilities sum to 1 and "The capital of France is" returns a ranked list of real candidates.

### Phase 3: Attention, Q/K/V and block inspection
- `/api/attention` with real weights, shape `[heads, seq, seq]` per layer, causal mask visible (upper triangle is zero).
- Q/K/V extraction from the fused `c_attn` output, split and reshaped per head.
- `/api/layer-inspection` using hooks for layer norm, attention output, residual, feed-forward and final hidden states.
- Validate layer and head indices.
- Done when: shapes match the config and invalid indices return clear errors.

### Phase 4: Frontend shell and first views
- Vite + React + TypeScript + Tailwind dark dashboard, sidebar, model status indicator, API client with loading and error states.
- Views: Tokenization Inspector, Embedding Inspector, Next-Token Prediction with probability chart and controls.
- Done when: all three views show live backend data.

### Phase 5: Attention, Q/K/V and block views
- Attention heatmap with layer and head selectors and top-weights table.
- Q/K/V inspector.
- Transformer Block Inspector with an "Illustrative" label on any diagram that is not live data.
- Done when: changing layer or head updates the view from real backend data.

### Phase 6: Gemini AI Tutor
- `gemini_service.py` using the `google-genai` SDK, `tutor` router, `/api/tutor/*` endpoints.
- Frontend Tutor panel: Explain this view, Ask a question, Quiz me, optional Compare with Gemini.
- All Gemini safety rules from feature I.
- Done when: with a key set, the Tutor explains the current view; with no key, the panel shows the "add key" message and the rest of the app still works.

### Phase 7: Debug console and model comparison
- Backend Debug Console panel (model, device, shapes, dtypes, timings, settings, warnings, Gemini request status without secrets).
- Model comparison (feature H) with safe handling of incompatible architectures.
- Done when: the console shows real timings and comparison works for two compatible models.

### Phase 8: Tests, hardening and documentation
- All tests from Section 8, mocked and real-model separated.
- Security pass against Section 7.
- Full README with PowerShell setup: virtual environment, backend and frontend installs, model download, setting `GEMINI_API_KEY` in `.env`, starting FastAPI, starting Vite, opening the browser, running tests.
- Final section: **How to verify the data is real** (below).
- Done when: tests pass and the README steps work from a clean machine.

---

## 10. How to Verify the Data Is Real (include in the README)

Explain how I can check each claim myself:
- **Token IDs:** run the same text through `AutoTokenizer.from_pretrained("distilgpt2")` in a separate Python shell and compare.
- **Embeddings:** the combined vector should equal `wte[token_id] + wpe[position]` from the model weights (and match `hidden_states[0]`).
- **Q/K/V:** recompute from the first block's `c_attn` weights and compare.
- **Attention:** rows sum to 1 and the upper triangle is 0 because of the causal mask. Compare with `output_attentions=True`.
- **Probabilities:** softmax of the final-position logits sums to 1, and the top token matches `argmax`.
- **Gemini:** its output is labelled as a written explanation, it never changes any number in the inspectors, and turning off the key removes only the Tutor panel.

---

## 11. Final Requirements

- Do not omit essential files, replace backend functionality with static JSON, or present simulated values as real.
- Generate complete working source code, file by file, inside each phase.
- If a phase is too large for one response, split it further and continue until it works.
