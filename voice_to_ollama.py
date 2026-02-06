import json
import tempfile
import wave
import urllib.request
import urllib.error
import subprocess

import sounddevice as sd
from faster_whisper import WhisperModel

# ------------------ SETTINGS ------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"      # change to any local model you have
WHISPER_MODEL = "small"         # "base" = faster, "small" = better accuracy
SAMPLE_RATE = 16000
RECORD_SECONDS = 6

SPEAK_BACK = True               # macOS: uses `say` (offline). Set False if you don't want TTS.

# ----------------------------------------------


def ask_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        raise SystemExit(
            "❌ Can't connect to Ollama at http://localhost:11434.\n"
            "Make sure Ollama is running (open the Ollama app, or run `ollama serve`).\n"
            f"Details: {e}"
        )


def record_wav(path: str, seconds: int = RECORD_SECONDS) -> None:
    print(f"🎙️ Recording for {seconds} seconds... Speak now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print("✅ Recording finished.")

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def transcribe(path: str, whisper: WhisperModel) -> str:
    segments, _info = whisper.transcribe(path, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def speak_macos(text: str) -> None:
    if not SPEAK_BACK:
        return
    try:
        subprocess.run(["say", text], check=False)
    except FileNotFoundError:
        # `say` exists on macOS; if not available just ignore
        pass


def main() -> None:
    print("Voice → Whisper (offline) → Ollama (local)")
    print("Commands: [r] record | [q] quit\n")
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print(f"Using Whisper model: {WHISPER_MODEL}\n")

    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    while True:
        cmd = input("Press [r] to record, [q] to quit: ").strip().lower()
        if cmd == "q":
            print("Bye 👋")
            break
        if cmd != "r":
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            record_wav(tmp.name, seconds=RECORD_SECONDS)
            user_text = transcribe(tmp.name, whisper)

        if not user_text:
            print("🤷 I didn't catch that. Try again.\n")
            continue

        print(f"\n🗣️ You said: {user_text}")
        answer = ask_ollama(user_text)
        print(f"\n🤖 LLM: {answer}\n")

        speak_macos(answer)


if __name__ == "__main__":
    main()
