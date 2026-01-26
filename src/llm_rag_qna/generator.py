from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


class HFText2TextGenerator:
    """
    A small, local generator to keep the project reproducible without API keys.
    Recommended defaults:
    - `google/flan-t5-base` (better but slower)
    - `google/flan-t5-small` (faster)
    """

    def __init__(self, model_name: str = "google/flan-t5-base", device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    @torch.inference_mode()
    def generate(self, prompt: str, *, cfg: GenerationConfig) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        do_sample = cfg.temperature > 0.0
        out = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature if do_sample else None,
            top_p=cfg.top_p if do_sample else None,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

