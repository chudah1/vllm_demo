"""
Configuration settings for the vLLM API server.
Loads settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Model Configuration
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    model_path: Optional[str] = None

    # vLLM Configuration
    gpu_memory_utilization: float = 0.95  # High utilization for vision model
    tensor_parallel_size: int = 1
    max_model_len: int = 32768  # Safe default for Qwen2-VL on T4/L4
    # API Security
    api_key: Optional[str] = None

    # HuggingFace Configuration
    hugging_face_hub_token: Optional[str] = None

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra environment variables (for vLLM-specific vars)


# Global settings instance
settings = Settings()


def get_model_name_or_path() -> str:
    """
    Returns the model path if specified, otherwise returns the model name.
    This allows using either HuggingFace model names or local model paths.
    """
    return settings.model_path if settings.model_path else settings.model_name


def get_vllm_engine_args() -> dict:
    """
    Returns a dictionary of vLLM engine arguments based on settings.
    """
    import torch
    import logging

    logger = logging.getLogger(__name__)

    # Check if CUDA is available
    has_cuda = torch.cuda.is_available()

    # Force CPU mode via environment variable (vLLM reads this)
    if not has_cuda:
        logger.warning("⚠️  No CUDA GPU detected - running in CPU mode (slower)")
        logger.warning("⚠️  For better performance, use a smaller model (e.g., Phi-2 or TinyLlama)")
        # Set environment variables for CPU execution BEFORE creating args
        os.environ["VLLM_TARGET_DEVICE"] = "cpu"
        os.environ["VLLM_CPU_ONLY"] = "1"  # Critical: Force CPU-only mode
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_USE_V1"] = "0"  # Disable v1 engine to avoid bugs

    args = {
        "model": get_model_name_or_path(),
        "max_model_len": settings.max_model_len,
        "max_num_seqs": settings.max_num_seqs,
        "trust_remote_code": True,  # Allow model code execution
        "tensor_parallel_size": 1,  # Always 1 for CPU and single GPU
        "quantization": None,  # Disable quantization (override model's default FP8 if present)
    }

    # GPU-specific settings (only if CUDA is available)
    if has_cuda:
        logger.info(f"CUDA detected! Using GPU acceleration")
        logger.info(f"GPU count: {torch.cuda.device_count()}")

        # GPU optimization parameters
        args["gpu_memory_utilization"] = settings.gpu_memory_utilization
        args["dtype"] = "auto"  # Let vLLM choose optimal dtype (bfloat16/float16)
        args["max_num_batched_tokens"] = 4096  # Further reduced for 7B models
        args["max_num_seqs"] = settings.max_num_seqs
        args["enable_chunked_prefill"] = True  # Enable for better latency
        args["enable_prefix_caching"] = False  # Disable to save memory

        # Enable swap space (now have 15GB RAM - can afford 6GB for overflow)
        args["swap_space"] = 6  # 6GB CPU swap space for KV cache overflow

        # Multi-GPU settings
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and settings.tensor_parallel_size > 1:
            args["tensor_parallel_size"] = min(settings.tensor_parallel_size, gpu_count)
            logger.info(f"Tensor parallelism enabled: {args['tensor_parallel_size']} GPUs")

    else:
        # CPU-only mode - explicit settings
        logger.info("Running in CPU-only mode")
        args["dtype"] = "float32"  # CPU doesn't support bfloat16 efficiently
        # Disable GPU acceleration explicitly
        args["cpu_offload_gb"] = 0

    # Add HuggingFace token if provided
    if settings.hugging_face_hub_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hugging_face_hub_token

    logger.info(f"vLLM engine args: {args}")
    return args
