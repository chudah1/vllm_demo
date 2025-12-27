"""
vLLM Engine wrapper for handling model inference.
Provides async methods for text generation with streaming support.
"""

import logging
import base64
import io
from typing import AsyncIterator, Dict, List, Optional, Union
from PIL import Image
from transformers import AutoTokenizer  # Generic tokenizer support
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
        self.tokenizer = None  # To be initialized

    
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

            # --- Initialize Tokenizer for Chat Templates ---
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                logger.info(f"Tokenizer initialized for {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}. Chat templating might fail.")

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
        multi_modal_data: Optional[Dict] = None,
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

        # Construct inputs for vLLM.
        # Use a dict for multimodal inputs to pass 'multi_modal_data' correctly.
        inputs = {"prompt": prompt}
        if multi_modal_data:
            inputs["multi_modal_data"] = multi_modal_data
        
        engine_prompt = inputs if multi_modal_data else prompt

        if stream:
            return self._stream_generate(engine_prompt, sampling_params, request_id)
        else:
            return await self._non_stream_generate(engine_prompt, sampling_params, request_id)

    async def _non_stream_generate(
        self, prompt: Union[str, Dict], sampling_params: SamplingParams, request_id: str
    ) -> str:
        final_output = ""
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        async for request_output in results_generator:
            if request_output.outputs:
                final_output = request_output.outputs[0].text
        return final_output

    async def _stream_generate(
        self, prompt: Union[str, Dict], sampling_params: SamplingParams, request_id: str
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

    async def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "id": self.model_name,
            "object": "model",
            "owned_by": "vllm",
        }

    async def chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate chat completion from messages."""
    async def chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate chat completion from messages."""
        
        # Prepare messages for tokenizer and extract images for vLLM
        tokenizer_messages = []
        multi_modal_data = {}
        image_count = 0

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            new_msg = {"role": role, "content": []}
            
            if isinstance(content, str):
                new_msg["content"] = content # AutoTokenizer handles str content usually, or we wrap in list
                # Actually standardize to list for robust multimodal handling if template supports it
                # But simple string is safer for text-only turns if template expects that.
                # Let's keep it simple: if str, pass str.
                pass
            elif isinstance(content, list):
                new_content = []
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                         new_content.append({"type": "text", "text": part.get("text", "")})
                    elif part_type == "image_url" or part_type == "image" or  (part.get("source", {}).get("type") == "base64"):
                        # Extract image data
                        image_data = None
                        try:
                            if part_type == "image_url":
                                url_data = part.get("image_url", {}).get("url", "")
                                if url_data.startswith("data:image"):
                                    base64_str = url_data.split(",")[1]
                                    image_bytes = base64.b64decode(base64_str)
                                    image_data = Image.open(io.BytesIO(image_bytes))
                            elif part.get("source", {}).get("type") == "base64":
                                base64_str = part["source"]["data"]
                                image_bytes = base64.b64decode(base64_str)
                                image_data = Image.open(io.BytesIO(image_bytes))
                        except Exception as e:
                            logger.error(f"Failed to decode image: {e}")

                        if image_data:
                            # Add generic image placeholder for tokenizer
                            # Check if Qwen2-VL needs specific placeholder? 
                            # Most AutoTokenizers handle {"type": "image"} by inserting their token (e.g. <|image_pad|>)
                            new_content.append({"type": "image"}) 
                            
                            # Add logic to append to multi_modal_data
                            if "image" not in multi_modal_data:
                                multi_modal_data["image"] = image_data
                            else:
                                if not isinstance(multi_modal_data["image"], list):
                                    multi_modal_data["image"] = [multi_modal_data["image"]]
                                multi_modal_data["image"].append(image_data)
                            image_count += 1
                        else:
                             new_content.append({"type": "text", "text": "[Image Failed to Load]"})
                
                new_msg["content"] = new_content
            
            tokenizer_messages.append(new_msg)

        # Apply chat template
        if hasattr(self, "tokenizer") and self.tokenizer:
            # We must ensure apply_chat_template logic matches what model expects
            # Qwen2-VL usually works with {"type": "image"} entries
            full_prompt = self.tokenizer.apply_chat_template(
                tokenizer_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback (shouldn't happen if initialized correctly)
            logger.warning("Tokenizer not initialized, falling back to simple join")
            full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        final_mm_data = multi_modal_data if image_count > 0 else None

        return await self.generate(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stream=stream,
            multi_modal_data=final_mm_data
        )

    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        if self.engine:
            logger.info("Shutting down vLLM engine")
            self.engine = None
