"""One local or hosted chat model, shared by the words reader, the judge, and the assistant.

OpenAI is used when OPENAI_API_KEY is set. Otherwise Ollama on this machine is used.
If neither is available, callers fall back to the written rules.
"""

from __future__ import annotations

import json
import os
import urllib.request


def parse_json(text: str) -> dict | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _ollama_base() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def ollama_model() -> str | None:
    chosen = os.environ.get("OLLAMA_MODEL", "").strip()
    if chosen:
        return chosen
    try:
        with urllib.request.urlopen(_ollama_base() + "/api/tags", timeout=2) as response:
            body = json.loads(response.read().decode())
    except Exception:
        return None
    names = [item.get("name", "") for item in body.get("models", []) if item.get("name")]
    if not names:
        return None
    for preferred in ("llama3.2:3b", "llama3.2:latest", "llama3.2"):
        if preferred in names:
            return preferred
    return names[0]


def available() -> str:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if ollama_model():
        return "ollama"
    return ""


def complete(messages: list[dict], *, json_mode: bool = False, temperature: float = 0) -> str | None:
    kind = available()
    if kind == "openai":
        return _openai(messages, json_mode=json_mode, temperature=temperature)
    if kind == "ollama":
        return _ollama(messages, json_mode=json_mode, temperature=temperature)
    return None


def _openai(messages: list[dict], *, json_mode: bool, temperature: float) -> str | None:
    payload: dict = {
        "model": _openai_model(),
        "temperature": temperature,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY'].strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            body = json.loads(response.read().decode())
        return str(body["choices"][0]["message"]["content"])
    except Exception:
        return None


def _ollama(messages: list[dict], *, json_mode: bool, temperature: float) -> str | None:
    model = ollama_model()
    if not model:
        return None
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    request = urllib.request.Request(
        _ollama_base() + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode())
        return str(body.get("message", {}).get("content", ""))
    except Exception:
        return None
