"""Main FastAPI application for Real Transformer Internals Explorer.
By Hammad, AI Engineer.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .model_loader import model_manager
import time
from .routers import core, tutor, debug
from .services import debug_service

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("transformer_explorer")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to pre-warm model on startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} ({settings.AUTHOR})...")
    logger.info(f"Attempting to load model '{settings.MODEL_NAME}' with eager attention...")
    try:
        model_manager.load_model()
        logger.info("Model loaded successfully and ready to serve requests.")
    except Exception as e:
        print("\n" + "="*80)
        print("  WARNING: MODEL COULD NOT BE DOWNLOADED OR LOADED ON STARTUP")
        print("="*80)
        print(f"Details: {e}\n")
        print("Actionable Setup Steps:")
        print("  1. Verify your Internet connection (for initial download from Hugging Face).")
        print("  2. Ensure PyTorch and Transformers are installed in the Python virtual environment:")
        print("       .\\venv\\Scripts\\pip.exe install -r requirements.txt")
        print("  3. Pre-download distilgpt2 directly via Python if offline or in restricted environments:")
        print("       .\\venv\\Scripts\\python.exe -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2')\"")
        print("  4. Check that sufficient RAM and disk storage are available on your system.")
        print("="*80 + "\n")
        logger.warning(
            f"Pre-warming model failed on startup: {e}. "
            "Endpoints will attempt to load on first request or report setup guidance."
        )
    yield
    logger.info("Shutting down Transformer Explorer backend.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Exposes real Transformer internal operations for educational exploration. By Hammad, AI Engineer.",
    lifespan=lifespan
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request metadata recording middleware
@app.middleware("http")
async def record_request_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    errors_captured = []
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        errors_captured.append(debug_service.scrub_sensitive_text(str(exc)))
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        path = request.url.path
        # Avoid self-referential logging of debug log retrieval or clear
        if path.startswith("/api") and path not in ("/api/debug/logs", "/api/debug/clear"):
            debug_service.record_request(
                endpoint=path,
                method=request.method,
                status_code=status_code,
                duration_ms=duration_ms,
                errors=errors_captured,
            )

# Include Routers with /api prefix (for frontend) and root prefix (for direct CLI accessibility)
app.include_router(core.router, prefix="/api")
app.include_router(tutor.router, prefix="/api")
app.include_router(debug.router, prefix="/api")

app.include_router(core.router, prefix="")
app.include_router(tutor.router, prefix="")
app.include_router(debug.router, prefix="")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to prevent path or secret leakage."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during model computation.",
            "data_origin": "Educational explanation"
        }
    )

@app.get("/")
def get_root():
    return {
        "project": settings.PROJECT_NAME,
        "author": settings.AUTHOR,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/api/health",
        "status": "online"
    }
