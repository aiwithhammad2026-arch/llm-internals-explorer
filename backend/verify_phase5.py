import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.services.attention_service import extract_attention
from app.services.inspection_service import inspect_layer

text = "The capital of France is"
layer = 0
head = 0

# 1. Direct call to attention_service
attn_res = extract_attention(text, layer=layer, head=head)
print("=== Real Output from Attention Service ===")
print(f"Tokens: {attn_res.tokens}")
print(f"Matrix shape: {len(attn_res.matrix)}x{len(attn_res.matrix[0])}")
# Cell [4, 0]: Source ' is' (pos 4), Target 'The' (pos 0)
cell_weight_api = attn_res.matrix[4][0]
print(f"Heatmap cell [4, 0] (source='{attn_res.tokens[4]}', target='{attn_res.tokens[0]}'): {cell_weight_api:.8f}")

# Q/K/V values
q0 = attn_res.qkv.q_slice[0]
k0 = attn_res.qkv.k_slice[0]
v0 = attn_res.qkv.v_slice[0]
print(f"Q slice dim 0: {q0:.8f}")
print(f"K slice dim 0: {k0:.8f}")
print(f"V slice dim 0: {v0:.8f}")
print(f"Max recomputed diff: {attn_res.qkv.max_recomputed_difference:.10f}")

# 2. Independent direct call to AutoModelForCausalLM
print("\n=== Real Output from Direct AutoModelForCausalLM ===")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2", attn_implementation="eager")
model.eval()

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# outputs.attentions is tuple of [batch, num_heads, seq_len, seq_len]
direct_attn = outputs.attentions[layer][0, head]
direct_cell_weight = float(direct_attn[4, 0].item())
print(f"Direct PyTorch cell [4, 0]: {direct_cell_weight:.8f}")
print(f"Difference: {abs(cell_weight_api - direct_cell_weight):.10e}")

# Direct Q projection verification
block = model.transformer.h[layer]
norm_input = block.ln_1(model.transformer.wte(inputs["input_ids"]) + model.transformer.wpe(torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0)))
fused_qkv = block.attn.c_attn(norm_input) # [1, seq, 2304]
q_direct, k_direct, v_direct = fused_qkv.split(768, dim=-1)
# Reshape to [1, seq, 12, 64] -> permute to [1, 12, seq, 64]
q_direct = q_direct.view(1, -1, 12, 64).permute(0, 2, 1, 3)
direct_q0 = float(q_direct[0, head, -1, 0].item())
print(f"Direct PyTorch Q[head=0, pos=final, dim=0]: {direct_q0:.8f}")
print(f"Difference: {abs(q0 - direct_q0):.10e}")
