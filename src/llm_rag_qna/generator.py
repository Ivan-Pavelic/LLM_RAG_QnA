from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


class GeneratorProtocol(Protocol):
    """Protocol for any generator that takes a prompt and returns a string."""

    def generate(self, prompt: str, *, cfg: GenerationConfig) -> str:
        ...


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


class OpenAIGenerator:
    """Generator using OpenAI API (GPT-3.5, GPT-4). Requires openai and OPENAI_API_KEY."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        try:
            import openai
        except ImportError as e:
            raise ImportError("OpenAIGenerator requires the openai package") from e
        self.model_name = model_name
        self._client = openai.OpenAI()

    def generate(self, prompt: str, *, cfg: GenerationConfig) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
        return (resp.choices[0].message.content or "").strip()


class HFCausalLMGenerator:
    """Generator for causal LMs (LLaMA 2, Mistral, TinyLlama, etc.) via Hugging Face."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", device: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        print(f"Loading tokenizer and model: {model_name} ... (first run may download several GB)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Moving model to {self.device} ...", flush=True)
        self.model.to(self.device)
        self._use_chat_template = hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None
        print("Model ready.", flush=True)

    @torch.inference_mode()
    def generate(self, prompt: str, *, cfg: GenerationConfig) -> str:
        if self._use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        do_sample = cfg.temperature > 0.0
        out = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature if do_sample else None,
            top_p=cfg.top_p if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

