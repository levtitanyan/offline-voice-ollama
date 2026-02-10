"""Shared configuration and types for the assistant."""

from pathlib import Path
from typing import Any, Dict


Message = Dict[str, str]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "json"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:1b"
WHISPER_MODEL = "small"

SAMPLE_RATE = 16000
RECORD_SECONDS = 6
SPEAK_BACK = False

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Be very concise, practical, and ask clarifying questions only when necessary."
    "Answer short and keep answers serious. Avoid unnecessary chit-chat. If you don't know, say you don't know."
)

SAVE_HISTORY = True
HISTORY_FILE = str(DATA_DIR / "chat_history.json")
MAX_TURNS_TO_KEEP = 20

COMMANDS_CONFIG_FILE = str(DATA_DIR / "commands.json")
CANONICAL_COMMANDS = ["turn_on", "turn_off", "open", "close", "lock", "unlock", "start", "stop"]
COMMAND_PARSER_SYSTEM_PROMPT = (
    "You are an NLP command parser for smart-home style requests.\n"
    "Return only JSON with this exact schema:\n"
    '{"is_command": boolean, "command": string|null, "target": string|null}\n'
    f'Allowed command values: {", ".join(CANONICAL_COMMANDS)}\n'
    "Decide is_command=true only when the user is asking to control a device/action.\n"
    "For normal questions, explanations, or chitchat, set is_command=false and use null values.\n"
    "Map paraphrases to canonical commands using meaning (for example, enable/power up -> turn_on).\n"
    "target should be the controlled object phrase only (for example: living room lights).\n"
    "Do not include extra keys or any text outside JSON."
)

DEFAULT_COMMAND_CONFIG: Dict[str, Any] = {
    "devices": [
        {
            "id": "front_door",
            "name": "front door",
            "aliases": ["front door", "main door", "door"],
            "supported_commands": ["open", "close", "lock", "unlock"],
        },
        {
            "id": "garage_door",
            "name": "garage door",
            "aliases": ["garage door", "garage"],
            "supported_commands": ["open", "close", "stop"],
        },
        {
            "id": "living_room_lights",
            "name": "living room lights",
            "aliases": ["living room lights", "lights", "living lights"],
            "supported_commands": ["turn_on", "turn_off"],
        },
        {
            "id": "robot_vacuum",
            "name": "robot vacuum",
            "aliases": ["robot vacuum", "vacuum", "cleaner"],
            "supported_commands": ["start", "stop"],
        },
        {
            "id": "coffee_machine",
            "name": "coffee machine",
            "aliases": ["coffee machine", "coffee maker", "espresso machine"],
            "supported_commands": ["turn_on", "turn_off", "start", "stop"],
        },
    ],
}
