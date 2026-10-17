"""Tokenizer service for extracting real token IDs, token strings, and tensor shapes.
By Hammad, AI Engineer.
"""

from typing import List
from ..schemas import TokenizeResponse, TokenItem
from ..model_loader import model_manager

def tokenize_text(text: str) -> TokenizeResponse:
    """Tokenizes text using the actual pretrained tokenizer.
    Returns real token IDs, byte displays, and input tensor shape.
    """
    if not model_manager.is_loaded or model_manager.tokenizer is None:
        model_manager.load_model()

    tokenizer = model_manager.tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not available.")

    # 1. Encode into real PyTorch tensors
    encoded = tokenizer(text, return_tensors="pt")
    input_ids_tensor = encoded["input_ids"]
    token_ids_list: List[int] = input_ids_tensor[0].tolist()

    # 2. Extract detailed token-by-token metadata
    tokens: List[TokenItem] = []
    for idx, token_id in enumerate(token_ids_list):
        # Decode single token ID
        token_str = tokenizer.decode([token_id])
        
        # Raw subword token string as represented in GPT-2 vocabulary (includes 'Ġ' for leading spaces)
        token_vocab_symbol = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token_vocab_symbol, bytes):
            token_vocab_symbol = token_vocab_symbol.decode("utf-8", errors="replace")

        # Create informative byte display representation
        if token_vocab_symbol.startswith("Ġ"):
            token_bytes_display = f"␣{token_vocab_symbol[1:]}"
        elif token_vocab_symbol == "Ċ":
            token_bytes_display = "\\n"
        elif token_vocab_symbol == "č":
            token_bytes_display = "\\t"
        else:
            token_bytes_display = token_vocab_symbol

        tokens.append(
            TokenItem(
                index=idx,
                token_id=token_id,
                token_str=token_str,
                token_bytes_display=token_bytes_display
            )
        )

    return TokenizeResponse(
        text=text,
        token_count=len(token_ids_list),
        token_ids=token_ids_list,
        tokens=tokens,
        input_tensor_shape=list(input_ids_tensor.shape),
        data_origin="Real model output"
    )
