"""
models/model_manager.py
========================
Manages both fine-tuned models (0.6B and 1.5B).
Lazy-loads on first request, caches in memory.
Supports hot-swapping between models per request.
"""
import logging
import time
from enum import Enum
from typing import Optional
import torch

from core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


class ModelSize(str, Enum):
    SMALL = "0.6b"    # Qwen3-0.6B  — faster, less VRAM
    LARGE = "1.5b"    # Qwen2.5-1.5B — slower, better quality


MODEL_REGISTRY = {
    ModelSize.SMALL: {
        "repo": settings.MODEL_0_6B,
        "description": "Qwen 0.6B — Fast & efficient",
        "vram": "~4GB",
        "speed": "~3s/query",
    },
    ModelSize.LARGE: {
        "repo": settings.MODEL_1_5B,
        "description": "Qwen 1.5B — High quality",
        "vram": "~8GB",
        "speed": "~7s/query",
    },
}


class ModelManager:
    """
    Singleton model manager.
    Keeps both models in memory if VRAM allows,
    otherwise unloads previous before loading next.
    """
    _instance: Optional["ModelManager"] = None
    _loaded: dict = {}          # {ModelSize: (model, tokenizer)}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, size: ModelSize) -> tuple:
        """Load model if not cached, return (model, tokenizer)."""
        if size in self._loaded:
            log.info(f"Model {size} already loaded — using cache")
            return self._loaded[size]

        repo = MODEL_REGISTRY[size]["repo"]
        log.info(f"Loading model {size} from {repo} ...")
        t0 = time.time()

        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name     = repo,
                max_seq_length = settings.MAX_SEQ_LENGTH,
                dtype          = torch.float16,
                load_in_4bit   = True,
            )
            FastLanguageModel.for_inference(model)

        except ImportError:
            # Fallback if unsloth not available (e.g. local dev without GPU)
            log.warning("Unsloth not found — falling back to transformers")
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel

            bnb = BitsAndBytesConfig(
                load_in_4bit           = True,
                bnb_4bit_compute_dtype = torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                quantization_config = bnb,
                device_map          = {"": 0},
                torch_dtype         = torch.float16,
                trust_remote_code   = True,
            )

        elapsed = time.time() - t0
        log.info(f"Model {size} loaded in {elapsed:.1f}s")
        self._loaded[size] = (model, tokenizer)
        return model, tokenizer

    def unload(self, size: ModelSize):
        """Free VRAM for a specific model."""
        if size in self._loaded:
            del self._loaded[size]
            torch.cuda.empty_cache()
            log.info(f"Model {size} unloaded")

    def list_loaded(self) -> list[str]:
        return [s.value for s in self._loaded.keys()]

    @property
    def available_models(self) -> dict:
        return {k.value: v for k, v in MODEL_REGISTRY.items()}


# Module-level singleton
model_manager = ModelManager()