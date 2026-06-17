"""
src/llm/__init__.py  —  LLM backend factory

WHY THIS EXISTS:
  compliance_agent.py calls get_llm_client() — it doesn't import
  OllamaLLMClient or BedrockLLMClient directly.
  Switching from local to Bedrock = change one env var (LLM_BACKEND=bedrock).
  Zero code changes in agent, embedder, or evaluator.

USAGE:
  from src.llm import get_llm_client, get_embedder

  llm = get_llm_client()                    # primary model
  llm = get_llm_client(use_complex=True)    # Mixtral
  embedder = get_embedder()
"""

from config.settings import settings


def get_llm_client(use_complex: bool = False):
    """Return the right LLM client based on LLM_BACKEND env var."""
    if settings.LLM_BACKEND == "bedrock":
        from src.llm.bedrock_client import BedrockLLMClient
        model_id = (settings.BEDROCK_COMPLEX_MODEL if use_complex
                    else settings.BEDROCK_PRIMARY_MODEL)
        return BedrockLLMClient(model_id=model_id)
    else:
        from src.llm.ollama_client import OllamaLLMClient
        model_id = (settings.LLM_COMPLEX_MODEL if use_complex
                    else settings.LLM_PRIMARY_MODEL)
        return OllamaLLMClient(model_id=model_id)


def get_embedder():
    """Return the right embedder based on LLM_BACKEND env var."""
    if settings.LLM_BACKEND == "bedrock":
        from src.llm.bedrock_client import BedrockEmbedder
        return BedrockEmbedder()
    else:
        from src.llm.ollama_client import OllamaEmbedder
        return OllamaEmbedder()
