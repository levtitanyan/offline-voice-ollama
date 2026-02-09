"""
Offline Voice/Text Assistant:
- Voice input (r) -> Whisper -> Ollama chat
- Text input (t)  -> Ollama chat
- Quit (q)

- Persistent memory (chat_history.json)

- System instructions management:
    /system <text>        -> REPLACE system prompt
    /addsystem <text>     -> APPEND to system prompt (keeps default + adds facts)

- Reset commands:
    /clear                -> clear conversation messages, keep current system prompt
    /reset                -> reset system prompt to default and clear conversation
"""


import json
import os
import tempfile
import wave
import urllib.request
import urllib.error
import subprocess
from typing import List, Dict, Optional, Any

import sounddevice as sd
from faster_whisper import WhisperModel

# ------------------ SETTINGS ------------------
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:1b"          # change to any local model you have
WHISPER_MODEL = "small"             # "base" = faster, "small" = better accuracy

SAMPLE_RATE = 16000
RECORD_SECONDS = 6

SPEAK_BACK = False                  # macOS: uses `say` (offline). Set True if you want TTS.

# Default assistant behavior
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Be very concise, practical, and ask clarifying questions only when necessary."
    "Answer short and keep answers serious. Avoid unnecessary chit-chat. If you don't know, say you don't know."
)

# Persistent history (kept on disk)
SAVE_HISTORY = True
HISTORY_FILE = "chat_history.json"

# Keep only the most recent turns to avoid huge prompts
MAX_TURNS_TO_KEEP = 20  # last N user+assistant turns

# Command rules config (loaded from JSON folder)
COMMANDS_CONFIG_FILE = os.path.join("json", "commands.json")
DEFAULT_COMMAND_CONFIG: Dict[str, Any] = {
    "leading_phrases": ["please ", "can you ", "could you ", "would you "],
    "trailing_fillers": [" now", " please", " for me"],
    "commands": [
        {"prefix": "turn on ", "name": "turn_on"},
        {"prefix": "turn off ", "name": "turn_off"},
        {"prefix": "switch on ", "name": "turn_on"},
        {"prefix": "switch off ", "name": "turn_off"},
        {"prefix": "open ", "name": "open"},
        {"prefix": "close ", "name": "close"},
        {"prefix": "lock ", "name": "lock"},
        {"prefix": "unlock ", "name": "unlock"},
        {"prefix": "start ", "name": "start"},
        {"prefix": "stop ", "name": "stop"},
    ],
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
# ----------------------------------------------


Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


def load_command_config() -> Dict[str, Any]:
    """Load command config from json/commands.json, fallback to defaults."""
    try:
        with open(COMMANDS_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        if isinstance(config, dict):
            return config
    except Exception:
        pass
    return DEFAULT_COMMAND_CONFIG


def _clean_target(raw_target: str) -> str:
    """Normalize target phrase for command payloads."""
    target = raw_target.strip(" \t\n\r.,!?").lower()
    target = " ".join(target.split())
    for article in ("the ", "a ", "an "):
        if target.startswith(article):
            target = target[len(article) :]
            break
    return target


def _load_phrase_list(command_config: Dict[str, Any], key: str) -> List[str]:
    """Return normalized phrase list from command config or defaults."""
    fallback = DEFAULT_COMMAND_CONFIG[key]
    values = command_config.get(key, fallback)
    if not isinstance(values, list):
        return fallback
    cleaned = [v.lower() for v in values if isinstance(v, str) and v.strip()]
    return cleaned or fallback


def _load_patterns(command_config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return validated command patterns from config or defaults."""
    patterns = command_config.get("commands", DEFAULT_COMMAND_CONFIG["commands"])
    if not isinstance(patterns, list):
        return DEFAULT_COMMAND_CONFIG["commands"]
    valid_patterns: List[Dict[str, str]] = []
    for item in patterns:
        if not isinstance(item, dict):
            continue
        prefix = item.get("prefix")
        name = item.get("name")
        if not isinstance(prefix, str) or not isinstance(name, str):
            continue
        if not prefix.strip() or not name.strip():
            continue
        valid_patterns.append({"prefix": prefix.lower(), "name": name.strip().lower()})
    return valid_patterns or DEFAULT_COMMAND_CONFIG["commands"]


def _load_devices(command_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return validated device knowledge base from config or defaults."""
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
    """Find best matching device by normalized target aliases."""
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


def extract_command_payload(user_text: str, command_config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Detect command intents and convert them into a simple JSON payload."""
    text = user_text.strip()
    if not text:
        return None

    lowered = text.lower().strip(" \t\n\r.,!?")

    leading_phrases = _load_phrase_list(command_config, "leading_phrases")
    for phrase in leading_phrases:
        if lowered.startswith(phrase):
            lowered = lowered[len(phrase) :].strip()
            break

    trailing_fillers = _load_phrase_list(command_config, "trailing_fillers")
    changed = True
    while changed:
        changed = False
        for suffix in trailing_fillers:
            if lowered.endswith(suffix):
                lowered = lowered[: -len(suffix)].strip()
                changed = True

    if not lowered:
        return None

    detected_command: Optional[str] = None
    detected_target: Optional[str] = None

    for rule in _load_patterns(command_config):
        prefix = rule["prefix"]
        command_name = rule["name"]
        if lowered.startswith(prefix):
            raw_target = lowered[len(prefix) :].strip(" \t\n\r.,!?")
            target = _clean_target(raw_target)
            if target:
                detected_command = command_name
                detected_target = target
                break

    # Support natural phrasing like "turn lights off" / "turn the lights on".
    if detected_command is None and lowered.startswith("turn "):
        if lowered.endswith(" off"):
            raw_target = lowered[len("turn ") : -len(" off")].strip(" \t\n\r.,!?")
            target = _clean_target(raw_target)
            if target:
                detected_command = "turn_off"
                detected_target = target
        elif lowered.endswith(" on"):
            raw_target = lowered[len("turn ") : -len(" on")].strip(" \t\n\r.,!?")
            target = _clean_target(raw_target)
            if target:
                detected_command = "turn_on"
                detected_target = target

    # Support natural phrasing like "switch lights off" / "switch lights on".
    if detected_command is None and lowered.startswith("switch "):
        if lowered.endswith(" off"):
            raw_target = lowered[len("switch ") : -len(" off")].strip(" \t\n\r.,!?")
            target = _clean_target(raw_target)
            if target:
                detected_command = "turn_off"
                detected_target = target
        elif lowered.endswith(" on"):
            raw_target = lowered[len("switch ") : -len(" on")].strip(" \t\n\r.,!?")
            target = _clean_target(raw_target)
            if target:
                detected_command = "turn_on"
                detected_target = target

    if detected_command is None or detected_target is None:
        return None

    devices = _load_devices(command_config)
    matched_device = _find_device(detected_target, devices)
    device_name = matched_device["name"] if matched_device is not None else detected_target

    return {"device": device_name, "command": detected_command}


def maybe_handle_command(
    history: List[Message], user_text: str, command_config: Dict[str, Any]
) -> Optional[List[Message]]:
    """Handle command-style input and return updated history when matched."""
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


def load_history() -> List[Message]:
    """Load saved chat history from disk if persistence is enabled."""
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
    """Persist current chat history to disk."""
    if not SAVE_HISTORY:
        return
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def ensure_system(history: List[Message], system_text: str) -> List[Message]:
    """Ensure there is a system message at history[0]."""
    if not history or history[0].get("role") != "system":
        return [{"role": "system", "content": system_text}] + history
    return history


def trim_history(history: List[Message]) -> List[Message]:
    """Keep the system message and only the most recent conversation turns."""
    system_msgs = [m for m in history if m["role"] == "system"]
    other = [m for m in history if m["role"] != "system"]

    keep = other[-2 * MAX_TURNS_TO_KEEP :]  # last N user+assistant pairs
    return system_msgs[:1] + keep           # keep only one system message


def ask_ollama_chat(messages: List[Message]) -> str:
    """Send chat messages to Ollama and return the assistant response."""
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            msg = result.get("message", {})
            return (msg.get("content") or "").strip()
    except urllib.error.URLError as e:
        raise SystemExit(
            "❌ Can't connect to Ollama at http://localhost:11434.\n"
            "Make sure Ollama is running (open the Ollama app, or run `ollama serve`).\n"
            f"Details: {e}"
        )


def record_wav(path: str, seconds: int = RECORD_SECONDS) -> None:
    """Record microphone audio and save it as a WAV file."""
    print(f"🎙️ Recording for {seconds} seconds... Speak now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print("✅ Recording finished.")

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def transcribe(path: str, whisper: WhisperModel) -> str:
    """Transcribe speech from a WAV file using Whisper."""
    segments, _info = whisper.transcribe(path, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


def speak_macos(text: str) -> None:
    """Speak text aloud on macOS using the built-in `say` command."""
    if not SPEAK_BACK:
        return
    try:
        subprocess.run(["say", text], check=False)
    except FileNotFoundError:
        pass


def chat_turn(history: List[Message], user_text: str, command_config: Dict[str, Any]) -> List[Message]:
    """Run one chat turn: add user message, get assistant reply, save/trim history."""
    command_history = maybe_handle_command(history, user_text, command_config)
    if command_history is not None:
        return command_history

    history.append({"role": "user", "content": user_text})
    history = trim_history(history)

    answer = ask_ollama_chat(history)

    history.append({"role": "assistant", "content": answer})
    history = trim_history(history)
    save_history(history)

    print(f"\n🤖 LLM: {answer}\n")
    speak_macos(answer)
    return history


def main() -> None:
    """Run the assistant loop (voice + text) with memory and system controls."""
    print("--------------------------------------------------------------------------")
    print("Voice/Text → Whisper (offline for voice) → Ollama Chat (local, with memory)")
    print("Commands: [r]=record | [t]=text | [q]=quit | /reset | /clear | /system <text> | /addsystem <text>\n")
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print(f"Using Whisper model: {WHISPER_MODEL}\n")

    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    history = load_history()
    history = ensure_system(history, SYSTEM_PROMPT)
    command_config = load_command_config()

    # If you want the default system prompt to overwrite old ones each run, uncomment:
    # history[0]["content"] = SYSTEM_PROMPT

    history = trim_history(history)
    save_history(history)

    while True:
        cmd = input(" Commands: r=record | t=text | q=quit: ").strip()

        if cmd.lower() == "q":
            print("Conversation over. See you soon!")
            break

        # --------- system controls ----------
        if cmd.startswith("/reset"):
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            save_history(history)
            print("🏭 Factory reset: system prompt restored + history cleared.\n")
            continue

        if cmd.startswith("/clear"):
            history = ensure_system(history, SYSTEM_PROMPT)
            history = [history[0]]  # keep CURRENT system prompt, clear conversation
            save_history(history)
            print("🧼 Clear: conversation cleared, system kept.\n")
            continue

        if cmd.startswith("/system "):
            new_sys = cmd[len("/system ") :].strip()
            if new_sys:
                # Replace system prompt completely (keeps conversation)
                history = [{"role": "system", "content": new_sys}] + [m for m in history if m["role"] != "system"]
                history = trim_history(history)
                save_history(history)
                print("✅ System instructions replaced.\n")
            continue

        if cmd.startswith("/addsystem "):
            extra = cmd[len("/addsystem ") :].strip()
            if extra:
                # Append to existing system prompt (keeps default + adds facts)
                history = ensure_system(history, SYSTEM_PROMPT)
                history[0]["content"] = history[0]["content"].rstrip() + "\n" + extra
                history = trim_history(history)
                save_history(history)
                print("✅ System instructions appended.\n")
            continue

        # --------- text mode ----------
        if cmd.lower() == "t":
            user_text = input("Your Input: ").strip()
            if user_text:
                history = chat_turn(history, user_text, command_config)
            continue

        # --------- voice mode ----------
        if cmd.lower() != "r":
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            record_wav(tmp.name, seconds=RECORD_SECONDS)
            user_text = transcribe(tmp.name, whisper)

        if not user_text:
            print("I didn't catch that. Try again.\n")
            continue

        print(f"\n User: {user_text}")
        history = chat_turn(history, user_text, command_config)


if __name__ == "__main__":
    main()
