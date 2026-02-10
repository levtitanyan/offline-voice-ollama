"""Audio input/output helpers."""

import subprocess
import wave

import sounddevice as sd
from faster_whisper import WhisperModel

from .assistant_config import RECORD_SECONDS, SAMPLE_RATE, SPEAK_BACK


def record_wav(path: str, seconds: int = RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> None:
    """Record microphone input to a WAV file."""
    print(f"Recording for {seconds} seconds... Speak now.")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("✅ Recording finished.")

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def transcribe(path: str, whisper: WhisperModel) -> str:
    """Transcribe WAV audio with Whisper."""
    segments, _info = whisper.transcribe(path, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


def speak_macos(text: str, enabled: bool = SPEAK_BACK) -> None:
    """Speak text on macOS when enabled."""
    if not enabled:
        return
    try:
        subprocess.run(["say", text], check=False)
    except FileNotFoundError:
        pass
