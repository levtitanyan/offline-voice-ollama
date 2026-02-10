"""History helpers for chat persistence and trimming."""

import json
import os
from typing import List

from .assistant_config import HISTORY_FILE, MAX_TURNS_TO_KEEP, Message, SAVE_HISTORY


def load_history() -> List[Message]:
    """Load chat history from disk."""
    if not SAVE_HISTORY:
        return []
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict) and "role" in m and "content" in m]
    except Exception:
        pass
    return []


def save_history(history: List[Message]) -> None:
    """Save chat history to disk."""
    if not SAVE_HISTORY:
        return
    history_dir = os.path.dirname(HISTORY_FILE)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def ensure_system(history: List[Message], system_text: str) -> List[Message]:
    """Ensure history starts with a system message."""
    if not history or history[0].get("role") != "system":
        return [{"role": "system", "content": system_text}] + history
    return history


def trim_history(history: List[Message]) -> List[Message]:
    """Keep one system prompt and recent turns."""
    system_msgs = [m for m in history if m["role"] == "system"]
    other = [m for m in history if m["role"] != "system"]
    keep = other[-2 * MAX_TURNS_TO_KEEP :]
    return system_msgs[:1] + keep
