"""
Unified LLM client for the frozen executor and the experience summarizer.

The backend is selected with the ``AGENT_API`` environment variable:

    gemini  Google Gemini REST API          GEMINI_API_KEYS, GEMINI_MODEL
    nvidia  NVIDIA NIM (OpenAI-compatible)  NVIDIA_API_KEYS, NVIDIA_MODEL
    openai  Any OpenAI-compatible endpoint  OPENAI_API_KEYS, OPENAI_MODEL, OPENAI_BASE_URL
            (e.g. a local vLLM server)
    claude  Anthropic API                   ANTHROPIC_API_KEY, CLAUDE_MODEL

``*_API_KEYS`` accept a comma-separated list; requests are spread across the keys
round-robin, which helps with per-key rate limits during parallel rollouts.

The experience summarizer can use a different model than the executor:

    SUMMARIZER_API       backend for summarization (default: AGENT_API)
    SUMMARIZER_MODEL     model name (default: the executor model of that backend)
    SUMMARIZER_BASE_URL  endpoint for SUMMARIZER_API=openai (default: OPENAI_BASE_URL)
    SUMMARIZER_API_KEYS  keys for the summarizer (default: the keys of that backend)
"""

import os
import time
import requests
import threading
from typing import List, Optional

if hasattr(time, "clock_gettime") and hasattr(time, "CLOCK_MONOTONIC"):
    def _real_monotonic() -> float:
        return time.clock_gettime(time.CLOCK_MONOTONIC)
else:
    _REAL_MONOTONIC = time.monotonic

    def _real_monotonic() -> float:
        return _REAL_MONOTONIC()


def _keys_from_env(*names: str) -> List[str]:
    """Read API keys from the first non-empty env var (comma-separated)."""
    for name in names:
        raw = os.environ.get(name, "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return keys
    return []


# Backend: "gemini", "nvidia", "openai", or "claude"
AGENT_API = os.environ.get("AGENT_API", "gemini")

# Claude
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", os.environ.get("CLAUDE_API_KEY", ""))
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Gemini
GEMINI_API_KEYS = _keys_from_env("GEMINI_API_KEYS", "GEMINI_API_KEY")
_gemini_key_idx = 0
_gemini_key_lock = threading.Lock()
_gemini_thread_local = threading.local()  # thread-safe per-worker key
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# NVIDIA NIM
NVIDIA_API_KEYS = _keys_from_env("NVIDIA_API_KEYS", "NVIDIA_API_KEY")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Set to 0 to retry forever.
NVIDIA_MAX_RETRIES = int(os.environ.get("NVIDIA_MAX_RETRIES", "60"))

# Generic OpenAI-compatible endpoint (vLLM, SGLang, OpenAI, Together, ...)
OPENAI_API_KEYS = _keys_from_env("OPENAI_API_KEYS", "OPENAI_API_KEY") or ["EMPTY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

# Per-key cooldown tracking for key rotation (thread-safe)
_key_last_fail = {}  # key -> timestamp of last failure
_key_lock = threading.Lock()


def _require_keys(keys: List[str], env_name: str) -> None:
    if not keys:
        raise RuntimeError(
            f"AGENT_API={AGENT_API} but no API key found. "
            f"Please export {env_name}=<key1>[,<key2>,...]"
        )


def bind_worker(worker_id: int) -> None:
    """Pin a rollout worker process to one API key so that parallel workers
    spread their requests across all available keys."""
    keys = NVIDIA_API_KEYS if AGENT_API == "nvidia" else OPENAI_API_KEYS if AGENT_API == "openai" else []
    if keys:
        os.environ["_EXPHARNESS_WORKER_KEY"] = keys[worker_id % len(keys)]


def _pick_best_key(keys: List[str], start_idx: int) -> str:
    """Pick the key that has been idle the longest (least recently failed)."""
    now = _real_monotonic()
    with _key_lock:
        best_key = None
        best_idle = -1
        for i in range(len(keys)):
            key = keys[(start_idx + i) % len(keys)]
            idle = now - _key_last_fail.get(key, 0)
            if idle > best_idle:
                best_idle = idle
                best_key = key
        return best_key


def _mark_key_failed(key: str):
    with _key_lock:
        _key_last_fail[key] = _real_monotonic()


def set_gemini_worker_key(key: str):
    """Bind the current thread to a specific Gemini key (thread-safe)."""
    _gemini_thread_local.worker_key = key


def _get_thread_key() -> Optional[str]:
    return getattr(_gemini_thread_local, 'worker_key', None)


def _next_gemini_key() -> str:
    """Round-robin across Gemini API keys (thread-safe)."""
    global _gemini_key_idx
    with _gemini_key_lock:
        key = GEMINI_API_KEYS[_gemini_key_idx % len(GEMINI_API_KEYS)]
        _gemini_key_idx += 1
        return key


def _gemini_call(prompt: str, max_tokens: int, temperature: float, max_retries: int,
                 model: Optional[str] = None, keys: Optional[List[str]] = None) -> str:
    """Call Gemini, rotating keys until a non-empty response or the wall-time budget
    (GEMINI_MAX_WALL_TIME seconds) runs out."""
    model = model or GEMINI_MODEL
    _require_keys(keys or GEMINI_API_KEYS, "GEMINI_API_KEYS")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    start_time = _real_monotonic()
    max_wall_time = float(os.environ.get("GEMINI_MAX_WALL_TIME", "10"))
    # Keep each HTTP request bounded so max_wall_time is actually respected.
    max_request_timeout = float(os.environ.get("GEMINI_REQUEST_TIMEOUT", "8"))
    attempt = 0

    def sleep_remaining(seconds: float) -> None:
        remaining = max_wall_time - (_real_monotonic() - start_time)
        if remaining > 0:
            time.sleep(max(0.0, min(seconds, remaining)))

    while _real_monotonic() - start_time < max_wall_time and (max_retries <= 0 or attempt < max_retries):
        # First attempt: thread-bound key; after that: rotate
        if keys:
            key = keys[(os.getpid() + attempt) % len(keys)]
        elif attempt == 0:
            key = _get_thread_key() or _next_gemini_key()
        else:
            key = _next_gemini_key()
        url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={key}"
        try:
            remaining = max_wall_time - (_real_monotonic() - start_time)
            if remaining <= 0:
                break
            request_timeout = max(2.0, min(max_request_timeout, remaining + 1.0))
            resp = requests.post(url, json=payload, timeout=request_timeout)
            if resp.status_code == 429:
                sleep_remaining(min(3 * (attempt + 1), 15))
                attempt += 1
                continue
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text:
                return text
            sleep_remaining(2)
            attempt += 1
        except Exception as e:
            elapsed = _real_monotonic() - start_time
            if elapsed > max_wall_time * 0.8:
                print(f"[Gemini] {attempt+1} attempts, {elapsed:.0f}s elapsed, last error: {e}")
            sleep_remaining(min(3 * (attempt + 1), 15))
            attempt += 1
    print(f"[Gemini] Gave up after {attempt} attempts / {_real_monotonic()-start_time:.0f}s")
    return ""


def _openai_compatible_call(prompt: str, max_tokens: int, temperature: float,
                            keys: List[str], base_url: str, model: str,
                            max_retries: int, tag: str) -> str:
    """Call an OpenAI-compatible chat endpoint, rotating through keys on failure."""
    import openai

    worker_key = os.environ.get('_EXPHARNESS_WORKER_KEY')
    start_idx = keys.index(worker_key) if worker_key in keys else os.getpid() % len(keys)

    attempt = 0
    while True:
        if max_retries > 0 and attempt >= max_retries:
            print(f"[{tag}] Failed after {attempt} retries, returning empty response.")
            return ""

        key = _pick_best_key(keys, start_idx + attempt)
        try:
            client = openai.OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=60)
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
            if content and content.strip():
                return content.strip()
            _mark_key_failed(key)
            attempt += 1
        except Exception as e:
            _mark_key_failed(key)
            attempt += 1
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(1)
            if attempt % 15 == 0:
                print(f"[{tag}] {attempt} consecutive failures, last: {e}")


def _claude_call(prompt: str, max_tokens: int, temperature: float, max_retries: int = 8,
                 model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Call the Anthropic API."""
    import anthropic
    api_key = api_key or CLAUDE_API_KEY
    _require_keys([api_key] if api_key else [], "ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model or CLAUDE_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.content[0].text if resp.content else ""
            if text and text.strip():
                return text.strip()
        except Exception as e:
            wait = min(2 ** attempt, 30)
            if attempt < max_retries - 1:
                print(f"[Claude] attempt {attempt+1} failed: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                print(f"[Claude] Failed after {max_retries} retries: {e}")
    return ""


def get_model_name() -> str:
    """Name of the executor model for the active backend (for logging)."""
    return {
        "nvidia": NVIDIA_MODEL,
        "openai": OPENAI_MODEL,
        "claude": CLAUDE_MODEL,
    }.get(AGENT_API, GEMINI_MODEL)


def llm_call(prompt: str, max_tokens: int = 100, temperature: float = 0.0, max_retries: int = 8) -> str:
    """Single-turn LLM call; the backend is selected by AGENT_API."""
    if AGENT_API == "nvidia":
        _require_keys(NVIDIA_API_KEYS, "NVIDIA_API_KEYS")
        return _openai_compatible_call(prompt, max_tokens, temperature, NVIDIA_API_KEYS,
                                       NVIDIA_BASE_URL, NVIDIA_MODEL, NVIDIA_MAX_RETRIES, "NVIDIA")
    if AGENT_API == "openai":
        return _openai_compatible_call(prompt, max_tokens, temperature, OPENAI_API_KEYS,
                                       OPENAI_BASE_URL, OPENAI_MODEL, NVIDIA_MAX_RETRIES, "OpenAI")
    if AGENT_API == "claude":
        return _claude_call(prompt, max_tokens, temperature, max_retries)
    return _gemini_call(prompt, max_tokens, temperature, max_retries)


# --------------------------------------------------------------------------
# Experience summarizer (may use a different model than the executor)
# --------------------------------------------------------------------------

SUMMARIZER_API = os.environ.get("SUMMARIZER_API", "") or AGENT_API
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "")
SUMMARIZER_BASE_URL = os.environ.get("SUMMARIZER_BASE_URL", "")
SUMMARIZER_API_KEYS = _keys_from_env("SUMMARIZER_API_KEYS")


def summarizer_call(prompt: str, max_tokens: int = 300, temperature: float = 0.0, max_retries: int = 8) -> str:
    """LLM call used to summarize trajectories into experiences (skills / lessons)."""
    api, model, keys = SUMMARIZER_API, SUMMARIZER_MODEL, SUMMARIZER_API_KEYS
    if api == "nvidia":
        keys = keys or NVIDIA_API_KEYS
        _require_keys(keys, "NVIDIA_API_KEYS")
        return _openai_compatible_call(prompt, max_tokens, temperature, keys, NVIDIA_BASE_URL,
                                       model or NVIDIA_MODEL, NVIDIA_MAX_RETRIES, "Summarizer")
    if api == "openai":
        return _openai_compatible_call(prompt, max_tokens, temperature, keys or OPENAI_API_KEYS,
                                       SUMMARIZER_BASE_URL or OPENAI_BASE_URL, model or OPENAI_MODEL,
                                       NVIDIA_MAX_RETRIES, "Summarizer")
    if api == "claude":
        return _claude_call(prompt, max_tokens, temperature, max_retries,
                            model=model or None, api_key=keys[0] if keys else None)
    return _gemini_call(prompt, max_tokens, temperature, max_retries, model=model or None, keys=keys or None)
