"""Ollama API helpers."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .assistant_config import Message, OLLAMA_CHAT_URL, OLLAMA_MODEL


def call_ollama_chat(
    messages: List[Message],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_CHAT_URL,
    timeout: int = 120,
    format_json: bool = False,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Call Ollama chat API and return parsed JSON response."""
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if format_json:
        payload["format"] = "json"
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ask_ollama_chat(
    messages: List[Message],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_CHAT_URL,
) -> str:
    """Return assistant text from Ollama chat API."""
    try:
        result = call_ollama_chat(messages=messages, model=model, url=url, timeout=120)
        msg = result.get("message", {})
        return (msg.get("content") or "").strip()
    except urllib.error.URLError as e:
        raise SystemExit(
            "❌ Can't connect to Ollama at http://localhost:11434.\n"
            "Make sure Ollama is running (open the Ollama app, or run `ollama serve`).\n"
            f"Details: {e}"
        )
