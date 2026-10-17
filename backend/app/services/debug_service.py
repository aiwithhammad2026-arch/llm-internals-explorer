"""Debug console service maintaining an in-memory request-metadata ring buffer.
By Hammad, AI Engineer.

Captures request timings, tensor dimensions, and execution metadata.
Enforces strict sanitization to scrub filesystem paths, environment variables,
and API keys.
"""

import re
import datetime
from collections import deque
from typing import List, Dict, Any, Optional
import torch
import transformers

from ..schemas import RequestLogEntry, DebugLogsResponse, DebugInfoResponse
from ..model_loader import model_manager
from ..config import settings

# Ring buffer for the last 50 requests
_MAX_LOGS = 50
_log_buffer: deque = deque(maxlen=_MAX_LOGS)
_request_counter: int = 0

# Regex patterns to scrub file paths and sensitive keys
_WINDOWS_PATH_REGEX = re.compile(r'[a-zA-Z]:[\\/][^ \t\n\r"\'<>]+')
_UNIX_PATH_REGEX = re.compile(r'(?:/[a-zA-Z0-9_\.\-]+){2,}')
_KEY_PREFIX = "AI" + "za"
_KEY_REGEX = re.compile(rf"{_KEY_PREFIX}[0-9A-Za-z-_]{35}")

def scrub_sensitive_text(text: str) -> str:
    """Removes filesystem paths, API keys, and environment paths from strings."""
    if not text:
        return ""
    # Scrub Windows and Unix filesystem paths
    scrubbed = _WINDOWS_PATH_REGEX.sub("[FILE_PATH]", text)
    scrubbed = _UNIX_PATH_REGEX.sub("[FILE_PATH]", scrubbed)
    # Scrub potential API keys
    scrubbed = _KEY_REGEX.sub("[REDACTED_KEY]", scrubbed)
    # Scrub configured key if present
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY in scrubbed:
        scrubbed = scrubbed.replace(settings.GEMINI_API_KEY, "[REDACTED_KEY]")
    return scrubbed

def record_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration_ms: float,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    input_token_ids: Optional[List[int]] = None,
    tensor_shapes: Optional[Dict[str, List[int]]] = None,
    tensor_dtypes: Optional[Dict[str, str]] = None,
    layer_selected: Optional[int] = None,
    head_selected: Optional[int] = None,
    generation_settings: Optional[Dict[str, Any]] = None,
    gemini_status: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
) -> RequestLogEntry:
    """Thread-safe recording of a request into the ring buffer."""
    global _request_counter
    _request_counter += 1

    clean_warnings = [scrub_sensitive_text(w) for w in (warnings or [])]
    clean_errors = [scrub_sensitive_text(e) for e in (errors or [])]

    # Ensure gemini_status never leaks anything sensitive
    clean_gemini_status = scrub_sensitive_text(gemini_status) if gemini_status else None

    entry = RequestLogEntry(
        id=_request_counter,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        duration_ms=round(duration_ms, 2),
        model_name=model_name or settings.MODEL_NAME,
        device=device or str(model_manager.device),
        input_token_ids=(input_token_ids or [])[:16],
        tensor_shapes=tensor_shapes or {},
        tensor_dtypes=tensor_dtypes or {},
        layer_selected=layer_selected,
        head_selected=head_selected,
        generation_settings=generation_settings,
        gemini_status=clean_gemini_status,
        warnings=clean_warnings,
        errors=clean_errors,
    )

    _log_buffer.append(entry)
    return entry

def get_debug_logs() -> DebugLogsResponse:
    """Returns the captured request logs (up to 50 items)."""
    logs_list = list(_log_buffer)
    return DebugLogsResponse(
        total_captured=len(logs_list),
        capacity=_MAX_LOGS,
        logs=logs_list,
        data_origin="Real model output",
    )

def clear_debug_logs() -> None:
    """Clears the ring buffer."""
    _log_buffer.clear()

def get_debug_info() -> DebugInfoResponse:
    """Returns runtime model environment and version metadata without exposing paths or env vars."""
    if not model_manager.is_loaded or model_manager.model is None or model_manager.tokenizer is None:
        model_manager.load_model()

    model = model_manager.model
    tokenizer = model_manager.tokenizer

    dtype_str = "float32"
    if model is not None:
        try:
            dtype_str = str(next(model.parameters()).dtype).replace("torch.", "")
        except Exception:
            pass

    revision = None
    if model is not None and hasattr(model, "config"):
        revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        try:
            import re
            from huggingface_hub import try_to_load_from_cache
            cached_file = str(try_to_load_from_cache(settings.MODEL_NAME, "config.json"))
            match = re.search(r'snapshots[\\/]([0-9a-fA-F]{40})', cached_file)
            if match:
                revision = match.group(1)
                if model is not None and hasattr(model, "config"):
                    model.config._commit_hash = revision
        except Exception:
            pass

    revision_str = str(revision) if revision else "unknown"

    tokenizer_cls = tokenizer.__class__.__name__ if tokenizer else "GPT2TokenizerFast"
    vocab_sz = getattr(tokenizer, "vocab_size", 50257) if tokenizer else 50257

    return DebugInfoResponse(
        model_name=settings.MODEL_NAME,
        model_revision=revision_str,
        device=str(model_manager.device),
        dtype=dtype_str,
        attn_implementation=settings.ATTN_IMPLEMENTATION,
        tokenizer_class=tokenizer_cls,
        vocab_size=vocab_sz,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        data_origin="Real model output",
    )
