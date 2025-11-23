"""
vLLM Engine wrapper for handling model inference.
Provides async methods for text generation with streaming support.
"""

import logging
from typing import AsyncIterator, Dict, List, Optional, Union
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)

class VLLMEngine:
    """Wrapper class for vLLM AsyncLLMEngine."""

    def __init__(self, engine_args: Dict):
        """
        Initialize the vLLM engine with the given arguments.
        """
        self.engine: Optional[AsyncLLMEngine] = None
        self.engine_args = engine_args.copy()

    
        self.model_name = self.engine_args.get("model", "unknown")
        logger.info(f"Initializing vLLM engine with model: {self.model_name}")

    async def initialize(self):
        """Initialize the async vLLM engine in CPU-only mode."""
        try:
      
            # --- Clean engine args: remove unsupported keys ---
            engine_args_dict = self.engine_args.copy()
            unsupported_keys = ["device", "cpu_only", "disable_v1_engine", "enable_chunked_prefill"]
            for key in unsupported_keys:
                if key in engine_args_dict:
                    logger.info(f"Removing unsupported key '{key}' from engine args")
                    engine_args_dict.pop(key)
            logger.info(f"Engine args used for AsyncEngineArgs: {engine_args_dict}")

            # --- Create AsyncEngineArgs ---
            args = AsyncEngineArgs(**engine_args_dict)
            logger.info("AsyncEngineArgs created successfully")

            # --- Initialize AsyncLLMEngine ---
            self.engine = AsyncLLMEngine.from_engine_args(args)
            logger.info("vLLM engine initialized successfully")

        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize vLLM engine: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise


    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate text from a prompt."""
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
            repetition_penalty=1.1,  # Penalize repetition
            frequency_penalty=0.5,   # Reduce frequency of repeated tokens
            presence_penalty=0.5,    # Encourage diverse token selection
        )

        request_id = random_uuid()

        if stream:
            return self._stream_generate(prompt, sampling_params, request_id)
        else:
            return await self._non_stream_generate(prompt, sampling_params, request_id)

    async def _non_stream_generate(
        self, prompt: str, sampling_params: SamplingParams, request_id: str
    ) -> str:
        final_output = ""
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        async for request_output in results_generator:
            if request_output.outputs:
                final_output = request_output.outputs[0].text
        return final_output

    async def _stream_generate(
        self, prompt: str, sampling_params: SamplingParams, request_id: str
    ) -> AsyncIterator[str]:
        previous_text = ""
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        async for request_output in results_generator:
            if request_output.outputs:
                current_text = request_output.outputs[0].text
                new_text = current_text[len(previous_text):]
                if new_text:
                    yield new_text
                previous_text = current_text

    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        if self.engine:
            logger.info("Shutting down vLLM engine")
            self.engine = None
