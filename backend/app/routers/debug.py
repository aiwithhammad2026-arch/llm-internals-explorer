"""Router for Debug Console and Model Comparison endpoints.
By Hammad, AI Engineer.
"""

import torch
from fastapi import APIRouter, HTTPException, status
from ..schemas import (
    DebugLogsResponse,
    DebugInfoResponse,
    CompareModelsRequest,
    CompareModelsResponse,
)
from ..services import debug_service, comparison_service

router = APIRouter(prefix="/api", tags=["Debug & Comparison"])

@router.get("/debug/logs", response_model=DebugLogsResponse)
def get_logs() -> DebugLogsResponse:
    """Returns the last 50 captured request logs with timings, tensor shapes, and errors."""
    return debug_service.get_debug_logs()

@router.post("/debug/clear")
def clear_logs():
    """Clears the debug request ring buffer."""
    debug_service.clear_debug_logs()
    return {"status": "cleared", "total_captured": 0}

@router.get("/debug/info", response_model=DebugInfoResponse)
def get_debug_info() -> DebugInfoResponse:
    """Returns runtime model environment and version metadata. Never reveals file paths or API keys."""
    return debug_service.get_debug_info()

@router.post("/compare-models", response_model=CompareModelsResponse)
def compare_models_endpoint(req: CompareModelsRequest) -> CompareModelsResponse:
    """Compares distilgpt2 and gpt2 side-by-side on parameters, architecture, and next-token logits."""
    try:
        return comparison_service.compare_models(req.text, req.model_a, req.model_b)
    except comparison_service.ModelOutOfMemoryError as oom:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Out of Memory",
                "message": debug_service.scrub_sensitive_text(str(oom)),
                "remediation": oom.remediation
            }
        )
    except (MemoryError, torch.cuda.OutOfMemoryError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Out of Memory",
                "message": f"Host system memory exhausted: {debug_service.scrub_sensitive_text(str(e))}",
                "remediation": [
                    "Close memory-heavy applications to free host system RAM.",
                    "Run single-model inspector endpoints with distilgpt2 (~350MB RAM).",
                    "Increase system paging file / virtual memory swap size."
                ]
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid Model", "message": debug_service.scrub_sensitive_text(str(e))}
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Out of Memory",
                    "message": f"Host system memory exhausted: {debug_service.scrub_sensitive_text(str(e))}",
                    "remediation": [
                        "Close memory-heavy applications to free host system RAM.",
                        "Run single-model inspector endpoints with distilgpt2 (~350MB RAM).",
                        "Increase system paging file / virtual memory swap size."
                    ]
                }
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Model Comparison Error", "message": debug_service.scrub_sensitive_text(str(e))}
        )
