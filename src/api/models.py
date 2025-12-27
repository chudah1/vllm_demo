"""
Pydantic models for API request and response validation.
OpenAI-compatible API schemas.
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Completion API Models
# ============================================================================


class CompletionRequest(BaseModel):
    """Request model for text completion endpoint."""

    prompt: Union[str, List[str]] = Field(
        ..., description="The prompt(s) to generate completions for"
    )
    model: Optional[str] = Field(
        None, description="Model to use (optional, uses default if not specified)"
    )
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(
        0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(-1, description="Top-k sampling parameter (-1 to disable)")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")
    n: int = Field(1, description="Number of completions to generate")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Once upon a time",
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": False,
            }
        }


class CompletionChoice(BaseModel):
    """Single completion choice in response."""

    text: str = Field(..., description="Generated text")
    index: int = Field(..., description="Index of this choice")
    finish_reason: Optional[str] = Field(
        None, description="Reason for completion finish (stop, length, etc.)"
    )


class CompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in prompt")
    completion_tokens: int = Field(..., description="Number of tokens in completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class CompletionResponse(BaseModel):
    """Response model for text completion endpoint."""

    id: str = Field(..., description="Unique identifier for this completion")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for generation")
    choices: List[CompletionChoice] = Field(..., description="List of completion choices")
    usage: Optional[CompletionUsage] = Field(None, description="Token usage statistics")


# ============================================================================
# Chat Completion API Models
# ============================================================================


class ChatMessage(BaseModel):
    """Single message in a chat conversation."""

    role: str = Field(
        ...,
        description="Role of the message sender (system, user, or assistant)",
        pattern="^(system|user|assistant)$",
    )
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the sender (optional)")

    class Config:
        json_schema_extra = {
            "example": {"role": "user", "content": "Hello, how are you?"}
        }


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion endpoint."""

    messages: List[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    model: Optional[str] = Field(
        None, description="Model to use (optional, uses default if not specified)"
    )
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(-1, description="Top-k sampling parameter (-1 to disable)")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")
    n: int = Field(1, description="Number of chat completions to generate")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                "max_tokens": 100,
                "temperature": 0.7,
            }
        }


class ChatCompletionChoice(BaseModel):
    """Single chat completion choice in response."""

    index: int = Field(..., description="Index of this choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Optional[str] = Field(
        None, description="Reason for completion finish"
    )


class ChatCompletionResponse(BaseModel):
    """Response model for chat completion endpoint."""

    id: str = Field(..., description="Unique identifier for this chat completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for generation")
    choices: List[ChatCompletionChoice] = Field(
        ..., description="List of chat completion choices"
    )
    usage: Optional[CompletionUsage] = Field(None, description="Token usage statistics")


# ============================================================================
# Streaming Response Models
# ============================================================================


class CompletionStreamChoice(BaseModel):
    """Single choice in a streaming completion response."""

    text: str = Field(..., description="Generated text delta")
    index: int = Field(..., description="Index of this choice")
    finish_reason: Optional[str] = Field(None, description="Reason for completion finish")


class CompletionStreamResponse(BaseModel):
    """Streaming response chunk for text completion."""

    id: str = Field(..., description="Unique identifier")
    object: str = Field("text_completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model name")
    choices: List[CompletionStreamChoice] = Field(..., description="Completion choices")


class ChatCompletionStreamDelta(BaseModel):
    """Delta content in streaming chat response."""

    role: Optional[str] = Field(None, description="Role (only in first chunk)")
    content: Optional[str] = Field(None, description="Content delta")


class ChatCompletionStreamChoice(BaseModel):
    """Single choice in a streaming chat completion response."""

    index: int = Field(..., description="Index of this choice")
    delta: ChatCompletionStreamDelta = Field(..., description="Delta content")
    finish_reason: Optional[str] = Field(None, description="Reason for completion finish")


class ChatCompletionStreamResponse(BaseModel):
    """Streaming response chunk for chat completion."""

    id: str = Field(..., description="Unique identifier")
    object: str = Field("chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatCompletionStreamChoice] = Field(..., description="Completion choices")


# ============================================================================
# Model Information
# ============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type")
    owned_by: str = Field(..., description="Organization that owns the model")
    created: Optional[int] = Field(None, description="Unix timestamp of model creation")


class ModelListResponse(BaseModel):
    """Response containing list of available models."""

    object: str = Field("list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of available models")


# ============================================================================
# Health Check
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status (healthy, unhealthy)")
    model: Optional[str] = Field(None, description="Loaded model name")
    message: Optional[str] = Field(None, description="Additional information")


# ============================================================================
# Error Response
# ============================================================================


class ErrorDetail(BaseModel):
    """Error detail information."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: ErrorDetail = Field(..., description="Error details")
