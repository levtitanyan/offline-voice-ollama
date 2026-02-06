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
from typing import List, Dict

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
# ----------------------------------------------


Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


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


def chat_turn(history: List[Message], user_text: str) -> List[Message]:
    """Run one chat turn: add user message, get assistant reply, save/trim history."""
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
                history = chat_turn(history, user_text)
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
        history = chat_turn(history, user_text)


if __name__ == "__main__":
    main()
