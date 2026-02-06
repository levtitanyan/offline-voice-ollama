# Offline Voice Assistant (Whisper + Ollama)

A local/offline voice assistant:
- Voice → text using Whisper (offline STT)
- Text → response using Ollama (local LLM)
- Optional voice reply on macOS using `say`

## Requirements
- macOS + Homebrew
- Ollama installed and at least one model pulled (e.g. `gemma3:1b`)

## Setup

### System (one-time)
```bash
brew install ffmpeg portaudio

