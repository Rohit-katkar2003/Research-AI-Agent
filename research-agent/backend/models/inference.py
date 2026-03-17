"""
models/inference.py
====================
All generation logic in one place.
Supports: summarize, extract_facts, decompose_query, synthesize_report
"""
import logging
import re
import json
import torch
from typing import Optional

from models.model_manager import ModelManager, ModelSize, model_manager
from core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# ── Prompt Templates (ChatML format matching fine-tune) ─────────────────────
SYSTEM_RESEARCH = (
    "You are a research analysis assistant. "
    "Extract precise facts, summarize clearly, and return structured information. "
    "Be concise and accurate. Do not hallucinate."
)

PROMPTS = {
    "summarize": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n"
        "Summarize the following text concisely, preserving all key facts:\n\n{text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "extract_facts": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n"
        "Extract the 3-5 most important facts from this text as a numbered list:\n\n{text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "decompose": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n"
        "Break this research question into 3-5 specific searchable sub-queries. "
        "Return as a numbered list:\n\n{text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "synthesize": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n"
        "Based on the following research findings, write a comprehensive summary "
        "with key insights and conclusions:\n\n{text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


class InferenceEngine:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    def _generate(
        self,
        prompt: str,
        model_size: ModelSize,
        max_new_tokens: int = None,
    ) -> str:
        model, tokenizer = self.manager.load(model_size)
        max_tokens = max_new_tokens or settings.MAX_NEW_TOKENS

        inputs = tokenizer(
            prompt,
            return_tensors       = "pt",
            truncation           = True,
            max_length           = settings.MAX_SEQ_LENGTH - max_tokens,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = max_tokens,
                temperature    = settings.TEMPERATURE,
                top_p          = settings.TOP_P,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def summarize(self, text: str, model_size: ModelSize) -> str:
        prompt = PROMPTS["summarize"].format(system=SYSTEM_RESEARCH, text=text[:3000])
        return self._generate(prompt, model_size, max_new_tokens=256)

    def extract_facts(self, text: str, model_size: ModelSize) -> list[str]:
        prompt = PROMPTS["extract_facts"].format(system=SYSTEM_RESEARCH, text=text[:3000])
        raw = self._generate(prompt, model_size, max_new_tokens=256)
        # Parse numbered list → python list
        facts = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|\Z)", raw, re.DOTALL)
        return [f.strip() for f in facts if f.strip()] or [raw]

    def decompose_query(self, query: str, model_size: ModelSize) -> list[str]:
        prompt = PROMPTS["decompose"].format(system=SYSTEM_RESEARCH, text=query)
        raw = self._generate(prompt, model_size, max_new_tokens=200)
        queries = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|\Z)", raw, re.DOTALL)
        return [q.strip() for q in queries if q.strip()] or [query]

    def synthesize_report(self, findings: str, model_size: ModelSize) -> str:
        prompt = PROMPTS["synthesize"].format(system=SYSTEM_RESEARCH, text=findings[:4000])
        return self._generate(prompt, model_size, max_new_tokens=512)


# Module-level singleton
inference_engine = InferenceEngine(model_manager)