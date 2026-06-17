"""
src/llm/bedrock_client.py

WHY THIS FILE EXISTS:
  Single place for all Bedrock API calls. Every other module (agent,
  embedder, evaluator) imports from here — so if AWS changes their API
  or we want to swap models, we change ONE file, not ten.

HOW BEDROCK WORKS (important to understand):
  1. boto3 creates a "bedrock-runtime" client authenticated via your
     AWS credentials (from env vars, ~/.aws/credentials, or IAM role).
  2. You call client.invoke_model(modelId=..., body=...) with a JSON body.
  3. The JSON body FORMAT is different per model family:
       Mistral:  {"prompt": "<s>[INST]...[/INST]", "max_tokens": N}
       Titan:    {"inputText": "...", "dimensions": 1536, "normalize": True}
  4. Response comes back as a streaming body — you must .read() it and
     parse the JSON yourself.

CREDENTIALS — how boto3 finds your AWS keys (in order of priority):
  1. Environment variables: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
  2. ~/.aws/credentials file (set by `aws configure`)
  3. IAM role attached to EC2/ECS/Lambda (automatic, no keys needed)
  For local dev: option 1 or 2. For production on AWS: option 3.

MISTRAL PROMPT FORMAT:
  Mistral instruct models expect a very specific format:
    <s>[INST] {instruction} [/INST]
  The <s> is the BOS (beginning of sentence) token.
  [INST] and [/INST] wrap the user message.
  For system prompts, prepend before [INST]:
    <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
  Getting this wrong produces garbage output — the model doesn't
  know where instructions end and content begins.
"""

from __future__ import annotations

import json
import logging
import time
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointResolutionError
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


# ─── Boto3 Client Factory ─────────────────────────────────────────────────────

def _make_bedrock_client():
    """
    Create a boto3 bedrock-runtime client.

    WHY bedrock-runtime (not bedrock)?
      "bedrock" client = control plane (list models, manage provisioned throughput)
      "bedrock-runtime" client = data plane (actually invoke models)
      You need bedrock-runtime for inference calls.

    WHY Config(retries=...)?
      Bedrock occasionally returns ThrottlingException under load.
      The adaptive retry mode automatically backs off and retries.
      max_attempts=3 means: 1 original try + 2 retries before raising.
    """
    session_kwargs = {"region_name": settings.AWS_REGION}

    # AWS_PROFILE lets you use a named profile from ~/.aws/credentials
    # e.g. if you ran `aws configure --profile finserv-dev`
    if settings.AWS_PROFILE:
        session = boto3.Session(profile_name=settings.AWS_PROFILE, **session_kwargs)
    else:
        session = boto3.Session(**session_kwargs)

    return session.client(
        service_name="bedrock-runtime",
        config=Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            read_timeout=settings.LLM_TIMEOUT_SECONDS,
            connect_timeout=10,
        ),
    )


# Singleton client — created once, reused across all calls.
# boto3 clients are thread-safe for read operations (invoke_model).
_bedrock_client = None

def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = _make_bedrock_client()
    return _bedrock_client


# ─── Mistral Prompt Formatter ─────────────────────────────────────────────────

def format_mistral_prompt(system_prompt: str, user_message: str) -> str:
    """
    Format a system + user message into Mistral's instruct template.

    WHY THIS MATTERS:
      Without the correct special tokens, Mistral treats your entire
      input as raw text and produces incoherent output. The model was
      fine-tuned specifically expecting this format.

    Example output:
      <s>[INST] You are a compliance assistant.

      What is the CET1 minimum ratio? [/INST]

    Args:
        system_prompt: The instruction/persona (e.g. "You are a FinServ compliance expert...")
        user_message:  The actual question or task
    Returns:
        Formatted string ready to pass to Bedrock as "prompt"
    """
    return f"<s>[INST] {system_prompt}\n\n{user_message} [/INST]"


# ─── BedrockLLMClient ─────────────────────────────────────────────────────────

class BedrockLLMClient:
    """
    Unified LLM client for all Bedrock Mistral model calls.

    Supports:
      - mistral.mistral-7b-instruct-v0:2      (primary — fast Q&A)
      - mistral.mixtral-8x7b-instruct-v0:1    (complex reasoning)
      - mistral.ministral-8b-instruct-v3:0    (256K context, test model)

    All three use the SAME request/response format, so one client handles all.

    Usage:
        llm = BedrockLLMClient()                       # uses primary model
        llm = BedrockLLMClient(model_id=settings.LLM_COMPLEX_MODEL)  # Mixtral

        response = llm.invoke(
            system_prompt="You are a compliance expert...",
            user_message="What is the Basel III CET1 minimum?"
        )
        print(response.text)
        print(response.input_tokens, response.output_tokens)
    """

    def __init__(self, model_id: Optional[str] = None):
        # Default to primary model; caller can override for complex tasks
        self.model_id = model_id or settings.LLM_PRIMARY_MODEL
        self.client = get_bedrock_client()
        logger.info("BedrockLLMClient initialised with model: %s", self.model_id)

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "LLMResponse":
        """
        Call Bedrock and return a structured LLMResponse.

        HOW THE REQUEST BODY IS BUILT:
          Bedrock expects a JSON string (not a dict) in the `body` parameter.
          The keys are model-specific. For all Mistral models:
            - "prompt":      the formatted instruct string
            - "max_tokens":  hard ceiling on output length
            - "temperature": 0.0 = deterministic, 1.0 = creative
            - "top_p":       nucleus sampling threshold (0.9 is safe default)

        HOW THE RESPONSE IS PARSED:
          response["body"] is a StreamingBody object.
          You must call .read() on it, then json.loads() the bytes.
          The output key is "outputs" → list → first item → "text".
          This is specific to Mistral on Bedrock (Titan uses different keys).

        Returns:
            LLMResponse with .text, .input_tokens, .output_tokens, .latency_ms
        Raises:
            BedrockInvokeError if Bedrock returns an error (with full context)
        """
        prompt = format_mistral_prompt(system_prompt, user_message)

        body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
            "temperature": temperature or settings.LLM_TEMPERATURE,
            "top_p": 0.9,
        })

        t0 = time.time()
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]
            # Common errors explained:
            # AccessDeniedException     → model access not enabled in Bedrock console
            # ResourceNotFoundException  → wrong model ID string
            # ThrottlingException       → too many requests (boto3 retries automatically)
            # ValidationException       → malformed request body
            logger.error("Bedrock ClientError [%s]: %s", error_code, error_msg)
            raise BedrockInvokeError(
                f"Bedrock call failed ({error_code}): {error_msg}\n"
                f"  Model: {self.model_id}\n"
                f"  Fix: check AWS Console → Bedrock → Model access"
            ) from e

        latency_ms = int((time.time() - t0) * 1000)

        # Parse the streaming response body
        response_body = json.loads(response["body"].read())

        # Mistral response structure:
        # { "outputs": [{"text": "...", "stop_reason": "stop"}] }
        text = response_body["outputs"][0]["text"].strip()

        # Bedrock doesn't return token counts for Mistral in the response body.
        # We estimate for cost tracking: ~4 chars ≈ 1 token (rough Mistral tokenizer ratio)
        estimated_input_tokens = len(prompt) // 4
        estimated_output_tokens = len(text) // 4

        logger.info(
            "Bedrock invoke OK | model=%s | latency=%dms | ~in=%d ~out=%d tokens",
            self.model_id, latency_ms, estimated_input_tokens, estimated_output_tokens
        )

        return LLMResponse(
            text=text,
            model_id=self.model_id,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            latency_ms=latency_ms,
        )

    def invoke_raw(self, prompt: str, **kwargs) -> str:
        """
        Invoke with a pre-formatted prompt string (no system/user splitting).
        Used by the evaluator when it builds its own prompt structure.
        """
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
            "top_p": 0.9,
        })
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        body_parsed = json.loads(response["body"].read())
        return body_parsed["outputs"][0]["text"].strip()


# ─── BedrockEmbedder ──────────────────────────────────────────────────────────

class BedrockEmbedder:
    """
    Embedding client using Amazon Titan Text Embeddings v2.

    WHY TITAN EMBEDDINGS:
      - Native to Bedrock — no separate model server needed
      - 1536 dimensions (vs nomic's 768) = richer semantic space
      - normalize=True makes cosine similarity == dot product
        (Qdrant's default metric) — important for retrieval accuracy
      - Supports up to 8192 tokens per input (same as nomic-embed-text)

    HOW IT DIFFERS FROM THE LLM CALL:
      - Different modelId: amazon.titan-embed-text-v2:0
      - Request body uses "inputText" key (not "prompt")
      - Response uses "embedding" key (list of 1536 floats)
      - No temperature/max_tokens — embeddings are deterministic

    Usage:
        embedder = BedrockEmbedder()
        vector = embedder.embed("What is the Basel III CET1 ratio?")
        # vector is a list of 1536 floats
        vectors = embedder.embed_batch(["text1", "text2", "text3"])
    """

    # Known Titan embedding model IDs to try in order
    TITAN_MODEL_IDS = [
        "amazon.titan-embed-text-v2:0",
        "amazon.titan-embed-text-v1:2",
        "amazon.titan-embed-text-v1:0",
    ]

    def __init__(self):
        self.client = get_bedrock_client()
        self.model_id = self._find_working_model()
        self.dimensions = settings.EMBEDDING_DIMENSION
        logger.info("BedrockEmbedder initialised | model=%s | dims=%d",
                    self.model_id, self.dimensions)

    def _find_working_model(self) -> str:
        """
        Try each known Titan model ID until one works.
        Different AWS accounts/regions expose different model versions.
        Returns the first model ID that successfully returns an embedding.
        """
        configured = settings.EMBEDDING_MODEL
        candidates = [configured] + [m for m in self.TITAN_MODEL_IDS if m != configured]

        for model_id in candidates:
            try:
                body = json.dumps({"inputText": "test"})
                resp = self.client.invoke_model(
                    modelId=model_id, body=body,
                    contentType="application/json", accept="application/json",
                )
                result = json.loads(resp["body"].read())
                if "embedding" in result:
                    # Update dimension from actual response
                    actual_dim = len(result["embedding"])
                    import os
                    os.environ["EMBEDDING_DIMENSION"] = str(actual_dim)
                    logger.info("Found working Titan model: %s (dim=%d)", model_id, actual_dim)
                    return model_id
            except Exception as e:
                logger.debug("Model %s failed: %s", model_id, e)
                continue

        logger.warning("No Titan model worked — defaulting to configured: %s", configured)
        return configured

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Titan request body:
          {
            "inputText": "...",
            "dimensions": 1536,   # must match EMBEDDING_DIMENSION in settings
            "normalize": true     # normalise to unit length (needed for cosine sim)
          }

        Titan response:
          { "embedding": [0.123, -0.456, ...],  # 1536 floats
            "inputTextTokenCount": 42 }
        """
        if not text or not text.strip():
            # Return zero vector for empty input — Qdrant will still accept it
            # but it won't match anything useful
            logger.warning("embed() called with empty text — returning zero vector")
            return [0.0] * self.dimensions

        # NOTE: Some AWS accounts only accept {"inputText": "..."}
        # Adding "dimensions" or "normalize" causes ValidationException.
        # The endpoint still returns 1536-dim vectors by default.
        body = json.dumps({
            "inputText": text[:8000],
        })

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error("Titan embed failed [%s]: %s", error_code, e)
            raise BedrockInvokeError(f"Embedding failed: {error_code}") from e

    def embed_batch(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """
        Embed a list of texts, batching to avoid throttling.

        WHY BATCHING:
          Bedrock doesn't have a native batch embedding endpoint for Titan.
          We call embed() once per text. Sending 500 docs in a tight loop
          triggers ThrottlingException. Sleeping 100ms between batches keeps
          us under the default quota (100 RPM for Titan embeddings).

        Args:
            texts:      list of strings to embed
            batch_size: how many to send before pausing
        Returns:
            list of embedding vectors, same order as input
        """
        vectors = []
        for i, text in enumerate(texts):
            vectors.append(self.embed(text))
            # Pause every batch_size calls to respect Bedrock rate limits
            if (i + 1) % batch_size == 0 and i + 1 < len(texts):
                logger.debug("Embedded %d/%d — pausing 100ms", i + 1, len(texts))
                time.sleep(0.1)
        return vectors


# ─── Response + Error Types ───────────────────────────────────────────────────

class LLMResponse:
    """
    Structured response from a Bedrock LLM call.

    WHY A CLASS INSTEAD OF A DICT:
      Attribute access (response.text) is cleaner than dict access
      (response["text"]) and gives IDE autocompletion.
      Also lets us add helper methods later (e.g. response.to_audit_entry()).
    """
    def __init__(
        self,
        text: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
    ):
        self.text = text
        self.model_id = model_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms

    def __repr__(self):
        return (
            f"LLMResponse(model={self.model_id}, "
            f"tokens={self.input_tokens}+{self.output_tokens}, "
            f"latency={self.latency_ms}ms, "
            f"text={self.text[:80]!r}...)"
        )

    @property
    def estimated_cost_usd(self) -> float:
        """
        Rough cost estimate based on Bedrock pricing (as of 2024).
        Mistral 7B:  $0.00015 per 1K input + $0.0002 per 1K output
        Mixtral 8x7B: $0.00045 per 1K input + $0.0007 per 1K output
        """
        if "mixtral" in self.model_id:
            return (self.input_tokens / 1000 * 0.00045) + (self.output_tokens / 1000 * 0.0007)
        return (self.input_tokens / 1000 * 0.00015) + (self.output_tokens / 1000 * 0.0002)


class BedrockInvokeError(Exception):
    """Raised when a Bedrock API call fails after retries."""
    pass
