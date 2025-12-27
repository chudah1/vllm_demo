"""
API routes for the vLLM server.
OpenAI-compatible endpoints for completions and chat.
"""

import logging
import time
from typing import AsyncIterator
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from vllm.utils import random_uuid

from src.api.models import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    CompletionStreamResponse,
    CompletionStreamChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatMessage,
    ModelListResponse,
    ModelInfo,
    HealthResponse,
)
from src.vllm_engine import VLLMEngine
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def get_engine(request: Request) -> VLLMEngine:
    """Dependency to get the vLLM engine from app state."""
    return request.app.state.engine


# ============================================================================
# Health Check Endpoint
# ============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check(engine: VLLMEngine = Depends(get_engine)):
    """
    Check the health status of the API server.
    """
    try:
        model_info = await engine.get_model_info()
        return HealthResponse(
            status="healthy",
            model=model_info.get("id"),
            message="Server is running and ready to accept requests",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy", message=f"Server error: {str(e)}"
        )


# ============================================================================
# Models Endpoint
# ============================================================================


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(engine: VLLMEngine = Depends(get_engine)):
    """
    List available models.
    OpenAI-compatible endpoint.
    """
    try:
        model_info = await engine.get_model_info()
        return ModelListResponse(
            object="list",
            data=[
                ModelInfo(
                    id=model_info.get("id", "unknown"),
                    object="model",
                    owned_by=model_info.get("owned_by", "vllm"),
                    created=int(time.time()),
                )
            ],
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Completion Endpoint
# ============================================================================


@router.post("/v1/completions")
async def create_completion(
    request: CompletionRequest, engine: VLLMEngine = Depends(get_engine)
):
    """
    Create a text completion.
    OpenAI-compatible endpoint with streaming support.
    """
    try:
        # Handle single or multiple prompts
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

        if request.stream:
            return StreamingResponse(
                _stream_completion(engine, prompts[0], request),
                media_type="text/event-stream",
            )
        else:
            return await _non_stream_completion(engine, prompts[0], request)

    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _non_stream_completion(
    engine: VLLMEngine, prompt: str, request: CompletionRequest
) -> CompletionResponse:
    """Handle non-streaming completion."""
    completion_id = f"cmpl-{random_uuid()}"
    created = int(time.time())

    # Generate completion
    output = await engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        stream=False,
    )

    # Calculate token usage (approximation)
    prompt_tokens = len(prompt.split())
    completion_tokens = len(output.split())

    return CompletionResponse(
        id=completion_id,
        object="text_completion",
        created=created,
        model=engine.model_name,
        choices=[
            CompletionChoice(
                text=output,
                index=0,
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_completion(
    engine: VLLMEngine, prompt: str, request: CompletionRequest
) -> AsyncIterator[str]:
    """Handle streaming completion."""
    completion_id = f"cmpl-{random_uuid()}"
    created = int(time.time())

    # Generate streaming completion
    generator = await engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        stream=True,
    )

    async for text_chunk in generator:
        chunk = CompletionStreamResponse(
            id=completion_id,
            object="text_completion.chunk",
            created=created,
            model=engine.model_name,
            choices=[
                CompletionStreamChoice(
                    text=text_chunk,
                    index=0,
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = CompletionStreamResponse(
        id=completion_id,
        object="text_completion.chunk",
        created=created,
        model=engine.model_name,
        choices=[
            CompletionStreamChoice(
                text="",
                index=0,
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================================
# Chat Completion Endpoint
# ============================================================================


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest, engine: VLLMEngine = Depends(get_engine)
):
    """
    Create a chat completion.
    OpenAI-compatible endpoint with streaming support.
    """
    try:
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(engine, request),
                media_type="text/event-stream",
            )
        else:
            return await _non_stream_chat_completion(engine, request)

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _non_stream_chat_completion(
    engine: VLLMEngine, request: ChatCompletionRequest
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""
    completion_id = f"chatcmpl-{random_uuid()}"
    created = int(time.time())

    # Convert messages to dict format
    messages = [msg.model_dump() for msg in request.messages]

    # Generate chat completion
    output = await engine.chat_completion(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        stream=False,
    )

    # Calculate token usage (approximation)
    prompt_text_parts = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            prompt_text_parts.append(content)
        elif isinstance(content, list):
            # Extract text from multimodal content parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    prompt_text_parts.append(part.get("text", ""))
    
    prompt_text = " ".join(prompt_text_parts)
    prompt_tokens = len(prompt_text.split())
    completion_tokens = len(output.split())

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=engine.model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_chat_completion(
    engine: VLLMEngine, request: ChatCompletionRequest
) -> AsyncIterator[str]:
    """Handle streaming chat completion."""
    completion_id = f"chatcmpl-{random_uuid()}"
    created = int(time.time())

    # Convert messages to dict format
    messages = [msg.model_dump() for msg in request.messages]

    # Send initial chunk with role
    initial_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        object="chat.completion.chunk",
        created=created,
        model=engine.model_name,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatCompletionStreamDelta(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Generate streaming chat completion
    generator = await engine.chat_completion(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        stream=True,
    )

    async for text_chunk in generator:
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=engine.model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(content=text_chunk),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        object="chat.completion.chunk",
        created=created,
        model=engine.model_name,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatCompletionStreamDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
