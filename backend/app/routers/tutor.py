"""FastAPI router for Gemini AI Tutor endpoints.
By Hammad, AI Engineer.
"""

from fastapi import APIRouter, HTTPException, status
from ..schemas import (
    TutorStatusResponse,
    TutorExplainRequest,
    TutorExplainResponse,
    TutorAskRequest,
    TutorAskResponse,
    TutorQuizRequest,
    TutorQuizResponse,
    TutorCompareGenerationRequest,
    TutorCompareGenerationResponse,
)
from ..services import gemini_service

router = APIRouter(prefix="/api/tutor", tags=["Gemini AI Tutor"])

@router.get("/status", response_model=TutorStatusResponse)
def get_status() -> TutorStatusResponse:
    """Returns whether Gemini API key is configured and active model name. Never reveals key."""
    return gemini_service.get_tutor_status()

@router.post("/explain", response_model=TutorExplainResponse)
def explain(req: TutorExplainRequest) -> TutorExplainResponse:
    """Explains real view data and numbers in plain English."""
    try:
        explanation_text = gemini_service.explain_view(req.view_name, req.context_summary)
        return TutorExplainResponse(
            explanation=explanation_text,
            model=gemini_service._get_model_name(),
            data_origin="Gemini-written explanation"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing GEMINI_API_KEY",
                "message": str(e),
                "setup_guidance": "Add GEMINI_API_KEY to backend/.env to enable the AI Tutor"
            }
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "Gemini API Error", "message": str(e)}
        )

@router.post("/ask", response_model=TutorAskResponse)
def ask(req: TutorAskRequest) -> TutorAskResponse:
    """Answers a user natural language question regarding the model or current view."""
    try:
        answer_text = gemini_service.answer_question(req.question, req.context_summary)
        return TutorAskResponse(
            answer=answer_text,
            model=gemini_service._get_model_name(),
            data_origin="Gemini-written explanation"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing GEMINI_API_KEY",
                "message": str(e),
                "setup_guidance": "Add GEMINI_API_KEY to backend/.env to enable the AI Tutor"
            }
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "Gemini API Error", "message": str(e)}
        )

@router.post("/quiz", response_model=TutorQuizResponse)
def quiz(req: TutorQuizRequest) -> TutorQuizResponse:
    """Generates 3 interactive multiple-choice questions on Transformer internals."""
    try:
        questions = gemini_service.generate_quiz(req.topic or "general")
        return TutorQuizResponse(
            topic=req.topic or "general",
            questions=questions,
            model=gemini_service._get_model_name(),
            data_origin="Gemini-written explanation"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing GEMINI_API_KEY",
                "message": str(e),
                "setup_guidance": "Add GEMINI_API_KEY to backend/.env to enable the AI Tutor"
            }
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "Gemini API Error", "message": str(e)}
        )

@router.post("/compare-generation", response_model=TutorCompareGenerationResponse)
def compare_generation(req: TutorCompareGenerationRequest) -> TutorCompareGenerationResponse:
    """Generates Gemini's continuation for comparison against distilgpt2."""
    try:
        continuation = gemini_service.compare_generation(req.prompt, req.local_generation)
        return TutorCompareGenerationResponse(
            prompt=req.prompt,
            gemini_continuation=continuation,
            model_name=gemini_service._get_model_name(),
            disclaimer=(
                "Gemini is a separate, massive-scale foundation model hosted by Google. "
                "Its internal weights, attention matrices, and logits cannot be inspected directly. "
                "Contrast its output with distilgpt2's local 82M parameter generation."
            ),
            data_origin="Gemini-written explanation"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing GEMINI_API_KEY",
                "message": str(e),
                "setup_guidance": "Add GEMINI_API_KEY to backend/.env to enable the AI Tutor"
            }
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "Gemini API Error", "message": str(e)}
        )
