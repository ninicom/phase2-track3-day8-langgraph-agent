"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Read a .env file into os.environ (without overriding already-set vars).

    Stdlib-only parser — no python-dotenv dependency. Walks up from this file to
    find the project root .env if no path is given. Returns the parsed pairs so
    callers (e.g. the web UI) can display configuration.
    """
    if path is None:
        for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
            candidate = parent / ".env"
            if candidate.is_file():
                path = candidate
                break
    if path is None or not Path(path).is_file():
        return {}
    parsed: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        parsed[key] = value
        os.environ.setdefault(key, value)
    return parsed


# Load .env on import so any node that calls get_llm() sees the configured keys.
load_env()


def get_llm(model: str | None = None, temperature: float = 0.0):
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY → ChatGoogleGenerativeAI
    2. OPENAI_API_KEY → ChatOpenAI
    3. ANTHROPIC_API_KEY → ChatAnthropic
    4. DEEPSEEK_API_KEY → ChatOpenAI against the DeepSeek OpenAI-compatible API

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    if os.getenv("DEEPSEEK_API_KEY"):
        # DeepSeek is OpenAI-compatible — reuse ChatOpenAI with its base_url.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "or DEEPSEEK_API_KEY in .env\nSee .env.example for configuration."
    )
