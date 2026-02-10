"""NLP command parsing and command-mode routing."""

import json
import re
import urllib.error
from typing import Any, Dict, List, Optional

from .assistant_config import (
    CANONICAL_COMMANDS,
    COMMANDS_CONFIG_FILE,
    COMMAND_PARSER_SYSTEM_PROMPT,
    DEFAULT_COMMAND_CONFIG,
    Message,
)
from .assistant_history import save_history, trim_history
from .assistant_ollama import call_ollama_chat


def _build_device_context(devices: List[Dict[str, Any]]) -> str:
    """Build a compact device context block for parser prompts."""
    lines: List[str] = []
    for device in devices:
        aliases = ", ".join(device.get("aliases", []))
        supported = ", ".join(device.get("supported_commands", []))
        lines.append(f'- name: "{device.get("name", "")}", aliases: [{aliases}], commands: [{supported}]')
    return "\n".join(lines)


def load_command_config() -> Dict[str, Any]:
    """Load device config from JSON or fallback defaults."""
    try:
        with open(COMMANDS_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        if isinstance(config, dict):
            return config
    except Exception:
        pass
    return DEFAULT_COMMAND_CONFIG


def _clean_target(raw_target: str) -> str:
    """Normalize a device target phrase."""
    target = raw_target.strip(" \t\n\r.,!?").lower()
    target = " ".join(target.split())
    for article in ("the ", "a ", "an "):
        if target.startswith(article):
            target = target[len(article) :]
            break
    return target


def _load_devices(command_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate and normalize device definitions."""
    fallback = DEFAULT_COMMAND_CONFIG["devices"]
    raw_devices = command_config.get("devices", fallback)
    if not isinstance(raw_devices, list):
        return fallback

    devices: List[Dict[str, Any]] = []
    for item in raw_devices:
        if not isinstance(item, dict):
            continue
        device_id = item.get("id")
        device_name = item.get("name")
        aliases = item.get("aliases", [])
        commands = item.get("supported_commands", [])
        if not isinstance(device_id, str) or not device_id.strip():
            continue
        if not isinstance(device_name, str) or not device_name.strip():
            continue
        if not isinstance(aliases, list):
            aliases = []
        if not isinstance(commands, list):
            commands = []

        normalized_aliases = {_clean_target(device_name)}
        for alias in aliases:
            if not isinstance(alias, str):
                continue
            normalized_alias = _clean_target(alias)
            if normalized_alias:
                normalized_aliases.add(normalized_alias)

        supported_commands = []
        for command_name in commands:
            if isinstance(command_name, str) and command_name.strip():
                supported_commands.append(command_name.strip().lower())
        if not supported_commands:
            continue

        devices.append(
            {
                "id": device_id.strip(),
                "name": " ".join(device_name.strip().split()),
                "aliases": sorted(normalized_aliases),
                "supported_commands": sorted(set(supported_commands)),
            }
        )

    return devices or fallback


def _find_device(target: str, devices: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the best device match for a target phrase."""
    normalized_target = _clean_target(target)
    if not normalized_target:
        return None

    best_device: Optional[Dict[str, Any]] = None
    best_score = -1

    for device in devices:
        for alias in device["aliases"]:
            if normalized_target == alias:
                score = 100 + len(alias)
            elif alias in normalized_target:
                score = 50 + len(alias)
            elif normalized_target in alias:
                score = 10 + len(normalized_target)
            else:
                continue

            if score > best_score:
                best_score = score
                best_device = device

    return best_device


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from text."""
    content = text.strip()
    if not content:
        return None
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _canonicalize_command_nlp(user_text: str, raw_command: str) -> Optional[str]:
    """Map a free-form action verb to one canonical command."""
    messages: List[Message] = [
        {
            "role": "system",
            "content": (
                "Map a smart-home action to one canonical command.\n"
                "Return only JSON with schema: "
                '{"command": string|null}\n'
                f'Allowed command values: {", ".join(CANONICAL_COMMANDS)}\n'
                "Use null if there is no clear mapping."
            ),
        },
        {
            "role": "user",
            "content": (
                f'User input: "{user_text}"\n'
                f'Raw action candidate: "{raw_command}"\n'
                "Return only JSON."
            ),
        },
    ]

    try:
        result = call_ollama_chat(
            messages=messages,
            timeout=30,
            format_json=True,
            temperature=0,
        )
    except urllib.error.URLError:
        return None

    content = ((result.get("message") or {}).get("content") or "").strip()
    parsed = _extract_json_object(content)
    if parsed is None:
        return None

    command = parsed.get("command")
    if not isinstance(command, str):
        return None
    normalized = command.strip().lower()
    if normalized in CANONICAL_COMMANDS:
        return normalized
    return None


def _parse_command_intent(parsed: Dict[str, Any], user_text: str) -> Optional[Dict[str, Any]]:
    """Validate and normalize raw parser output."""
    is_command = parsed.get("is_command")
    command = parsed.get("command")
    target = parsed.get("target")

    if not isinstance(is_command, bool):
        return None
    if not is_command:
        return {"is_command": False}
    if not isinstance(command, str):
        return None

    normalized_command = command.strip().lower()
    if normalized_command not in CANONICAL_COMMANDS:
        normalized_command = _canonicalize_command_nlp(user_text, normalized_command) or ""
    if normalized_command not in CANONICAL_COMMANDS:
        return None

    normalized_target = target.strip() if isinstance(target, str) else ""
    return {"is_command": True, "command": normalized_command, "target": normalized_target}


def extract_command_intent_nlp(user_text: str, devices: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Classify command intent from text via Ollama."""
    text = user_text.strip()
    if not text:
        return None

    device_context = _build_device_context(devices)
    messages: List[Message] = [
        {"role": "system", "content": COMMAND_PARSER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f'User input: "{text}"\n'
                "Known devices:\n"
                f"{device_context}\n"
                "Prefer canonical device names from this list when possible.\n"
                "Return only JSON."
            ),
        },
    ]

    for use_json_format in (True, False):
        try:
            result = call_ollama_chat(
                messages=messages,
                timeout=60,
                format_json=use_json_format,
                temperature=0,
            )
        except urllib.error.URLError:
            return None

        content = ((result.get("message") or {}).get("content") or "").strip()
        parsed = _extract_json_object(content)
        if parsed is None:
            continue

        intent = _parse_command_intent(parsed, user_text=text)
        if intent is not None:
            return intent

    return None


def extract_command_payload(user_text: str, command_config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Convert command-like input into a payload JSON."""
    devices = _load_devices(command_config)
    intent = extract_command_intent_nlp(user_text, devices)
    if intent is None or not intent.get("is_command"):
        return None

    command_name = intent["command"]
    raw_target = intent.get("target") or ""

    matched_device = None
    if raw_target:
        matched_device = _find_device(raw_target, devices)
    if matched_device is None:
        matched_device = _find_device(user_text, devices)

    if matched_device is not None:
        if command_name not in matched_device.get("supported_commands", []):
            return None
        device_name = matched_device["name"]
    else:
        device_name = _clean_target(raw_target)

    if not device_name:
        return None
    return {"device": device_name, "command": command_name}


def maybe_handle_command(
    history: List[Message], user_text: str, command_config: Dict[str, Any]
) -> Optional[List[Message]]:
    """Handle command input and return updated history."""
    payload = extract_command_payload(user_text, command_config)
    if payload is None:
        return None

    reply = json.dumps(payload, ensure_ascii=False)

    history.append({"role": "user", "content": user_text})
    history = trim_history(history)
    history.append({"role": "assistant", "content": reply})
    history = trim_history(history)
    save_history(history)

    print(f"\n{reply}\n")
    return history
