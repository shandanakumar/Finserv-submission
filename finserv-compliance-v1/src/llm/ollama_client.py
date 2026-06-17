"""
src/llm/ollama_client.py  —  Local Ollama inference

HOW OLLAMA WORKS:
  Ollama runs as a local HTTP server (default: http://localhost:11434).
  You pull models with `ollama pull mistral` — they're stored on disk (~4GB each).
  Your code sends a POST request with the prompt; Ollama runs the model on
  your GPU (or CPU if no GPU) and streams back tokens.

WHY THE SAME INTERFACE AS bedrock_client.py:
  Both OllamaLLMClient and BedrockLLMClient expose the same methods:
    .invoke(system_prompt, user_message) → LLMResponse
    .invoke_raw(prompt) → str
  So compliance_agent.py doesn't need to change when you switch backends.
  The factory function get_llm_client() picks the right one from settings.

OLLAMA ENDPOINTS USED:
  POST /api/generate   →  text generation (Mistral, Mixtral)
  POST /api/embeddings →  embedding generation (nomic-embed-text)
  GET  /api/tags       →  list downloaded models (used in health check)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests
from requests.exceptions import ConnectionError, Timeout

from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaLLMClient:
    """
    LLM client for locally-running Ollama models.

    Supports any model you've pulled with `ollama pull <name>`:
      mistral        →  Mistral 7B Instruct (fast Q&A, 4.1 GB)
      mixtral        →  Mixtral 8×7B (complex reasoning, 26 GB)
      mistral:latest →  same as mistral (Ollama resolves aliases)

    Request format (Ollama /api/generate):
      {
        "model": "mistral",
        "prompt": "<s>[INST] system\n\nuser [/INST]",
        "stream": false,
        "options": {"temperature": 0.1, "num_predict": 2048}
      }

    Response (when stream=false):
      {
        "response": "The answer is...",
        "done": true,
        "total_duration": 1234567890,   # nanoseconds
        "eval_count": 42                # output tokens
      }
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or settings.LLM_PRIMARY_MODEL  # "mistral"
        self.base_url = settings.OLLAMA_BASE_URL               # "http://localhost:11434"
        self.generate_url = f"{self.base_url}/api/generate"
        logger.info("OllamaLLMClient initialised | model=%s | url=%s",
                    self.model_id, self.base_url)

    def _format_prompt(self, system_prompt: str, user_message: str) -> str:
        """
        Mistral instruct format — same as Bedrock version.
        Must be consistent between local and cloud so eval results are comparable.
        """
        return f"<s>[INST] {system_prompt}\n\n{user_message} [/INST]"

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "LLMResponse":
        """
        Call Ollama and return a structured LLMResponse.

        stream=False: wait for the complete response before returning.
        For interactive demos you'd set stream=True and yield tokens,
        but for a compliance pipeline we want the full answer atomically.
        """
        prompt = self._format_prompt(system_prompt, user_message)

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or settings.LLM_TEMPERATURE,
                "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
                "top_p": 0.9,
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(
                self.generate_url,
                json=payload,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except ConnectionError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}\n"
                "  Fix: make sure Ollama is running.  Start it with:\n"
                "    ollama serve           (in a separate terminal)\n"
                "  Or on macOS/Linux it may already be running as a service."
            )
        except Timeout:
            raise OllamaConnectionError(
                f"Ollama timed out after {settings.LLM_TIMEOUT_SECONDS}s.\n"
                "  Large models (Mixtral) are slow on CPU — increase LLM_TIMEOUT_SECONDS."
            )

        latency_ms = int((time.time() - t0) * 1000)
        data = resp.json()
        text = data.get("response", "").strip()

        # Ollama gives us real token counts — useful for benchmarking
        input_tokens = data.get("prompt_eval_count", len(prompt) // 4)
        output_tokens = data.get("eval_count", len(text) // 4)

        logger.info(
            "Ollama invoke OK | model=%s | latency=%dms | in=%d out=%d tokens",
            self.model_id, latency_ms, input_tokens, output_tokens
        )

        return LLMResponse(
            text=text,
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def invoke_raw(self, prompt: str, **kwargs) -> str:
        """Pre-formatted prompt, returns raw text string."""
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
                "num_predict": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            },
        }
        resp = requests.post(self.generate_url, json=payload,
                             timeout=settings.LLM_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def is_available(self) -> bool:
        """Health check — returns True if Ollama is running and model is pulled."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
            return self.model_id.split(":")[0] in models
        except Exception:
            return False


class OllamaEmbedder:
    """
    Embedding client using nomic-embed-text via Ollama.

    nomic-embed-text specs:
      - 768-dimensional vectors
      - 8192 token context window (same as Titan v2)
      - Apache 2.0 license — fully open source
      - ~274 MB on disk
      - Runs on CPU: ~50ms per embedding

    Ollama /api/embeddings request:
      {"model": "nomic-embed-text", "prompt": "text to embed"}

    Response:
      {"embedding": [0.123, -0.456, ...]}   # 768 floats
    """

    def __init__(self):
        self.model_id = settings.EMBEDDING_MODEL  # "nomic-embed-text"
        self.base_url = settings.OLLAMA_BASE_URL
        self.embed_url = f"{self.base_url}/api/embeddings"
        self.dimensions = settings.EMBEDDING_DIMENSION  # 768
        logger.info("OllamaEmbedder initialised | model=%s | dims=%d",
                    self.model_id, self.dimensions)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string → 768-dim vector."""
        if not text or not text.strip():
            return [0.0] * self.dimensions

        try:
            resp = requests.post(
                self.embed_url,
                json={"model": self.model_id, "prompt": text[:8000]},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except ConnectionError:
            raise OllamaConnectionError(
                "Cannot connect to Ollama for embeddings.\n"
                "  Run: ollama serve  (then: ollama pull nomic-embed-text)"
            )

    def embed_batch(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        """
        Embed a list of texts.
        No rate limiting needed for local — Ollama handles queuing internally.
        batch_size=50 is fine locally (vs 20 for Bedrock Titan).
        """
        vectors = []
        for i, text in enumerate(texts):
            vectors.append(self.embed(text))
            if (i + 1) % 10 == 0:
                logger.debug("Embedded %d / %d", i + 1, len(texts))
        return vectors


# ─── Shared response types ────────────────────────────────────────────────────

class LLMResponse:
    """Same interface as BedrockLLMClient.LLMResponse — swap freely."""
    def __init__(self, text, model_id, input_tokens, output_tokens, latency_ms):
        self.text = text
        self.model_id = model_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms

    def __repr__(self):
        return (f"LLMResponse(model={self.model_id}, "
                f"latency={self.latency_ms}ms, text={self.text[:60]!r}...)")

    @property
    def estimated_cost_usd(self) -> float:
        return 0.0  # Local = free


class OllamaConnectionError(Exception):
    """Raised when Ollama is not running or model not pulled."""
    pass
