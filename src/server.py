"""
FastAPI server for vLLM inference.
CPU-only compatible setup.
"""

import os
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.vllm_engine import VLLMEngine
from config.settings import settings, get_vllm_engine_args

# Force CPU-only mode
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_CPU_ONLY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Disable v1 engine to avoid sampled_ids tensor bug with Phi-3
os.environ["VLLM_USE_V1"] = "0"

# --- Logging ---
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Lifespan context for FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting vLLM API server...")
    try:
        engine_args = get_vllm_engine_args()
        logger.info(f"vLLM configuration: {engine_args}")

        engine = VLLMEngine(engine_args)
        await engine.initialize()
        app.state.engine = engine

        logger.info(f"Server running on http://{settings.host}:{settings.port}")
        yield

    finally:
        logger.info("Shutting down vLLM API server...")
        if hasattr(app.state, "engine"):
            await app.state.engine.shutdown()
        logger.info("Shutdown complete")

# --- FastAPI app ---
app = FastAPI(
    title="vLLM CPU API Server",
    description="OpenAI-compatible API server using vLLM (CPU-only)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {
        "name": "vLLM CPU API Server",
        "version": "1.0.0",
        "model": settings.model_name,
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "completions": "/v1/completions",
            "chat_completions": "/v1/chat/completions",
            "docs": "/docs",
        },
    }

def main():
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )

if __name__ == "__main__":
    main()
