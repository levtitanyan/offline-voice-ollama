# Offline Voice Assistant (Whisper + Ollama)

Local voice/text assistant with:
- Offline speech-to-text via `faster-whisper`
- Local LLM responses via Ollama
- NLP command parsing that outputs structured JSON for device-control requests
- Persistent chat memory in `json/chat_history.json`

## Production-Oriented Notes
- No cloud API is required for normal operation.
- Command parsing is constrained to canonical actions (`turn_on`, `turn_off`, `open`, `close`, `lock`, `unlock`, `start`, `stop`).
- Device control targets are validated against `json/commands.json` aliases and allowed commands.
- Command parsing uses Ollama locally (it asks the model to classify/normalize commands), then the code matches targets to your device catalog.

## Project Layout
```text
.
├── app/
│   ├── __init__.py
│   ├── assistant_audio.py
│   ├── assistant_commands.py
│   ├── assistant_config.py
│   ├── assistant_history.py
│   ├── assistant_ollama.py
│   └── voice_to_ollama.py
├── json/
│   ├── commands.json
│   └── chat_history.json   # created/updated at runtime
├── requirements.txt
└── README.md
```

## Prerequisites
- macOS (recommended for this setup)
- Python 3.10+
- Ollama installed
- A local Ollama model pulled (default: `gemma3:1b`)

System dependencies (macOS):
```bash
brew install ffmpeg portaudio
```

## Setup
```bash
git clone <https://github.com/levtitanyan/offline-voice-ollama>
cd offline-voice-ollama
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull the default model:
```bash
ollama pull gemma3:1b
```

Start Ollama service if needed:
```bash
ollama serve
```

## Run
From repo root:
```bash
python -m app.voice_to_ollama
```

Direct script run (also supported):
```bash
python app/voice_to_ollama.py
```


## Example Command Output
If you type a device-control request, the assistant prints JSON instead of a normal chat answer, for example:
```json
{"device":"living room lights","command":"turn_off"}
```

## Runtime Controls
- `r`: record voice input
- `t`: text input
- `q`: quit (clears the conversation)
- `/clear`: clear conversation, keep current system prompt
- `/reset`: reset system prompt and clear conversation
- `/system <text>`: replace system instructions
- `/addsystem <text>`: append to system instructions

## Configuration
Main settings are in `app/assistant_config.py`:
- `OLLAMA_MODEL`
- `WHISPER_MODEL`
- `RECORD_SECONDS`
- `SPEAK_BACK`
- `SYSTEM_PROMPT`

Device catalog is in `json/commands.json`:
- `id`
- `name`
- `aliases`
- `supported_commands`

## NLP Command Pipeline (Simple)
1. User input arrives (voice transcription or text).
2. NLP parser asks Ollama if input is a command.
3. If command: normalize to canonical command + target.
4. Target is matched to known devices via aliases.
5. Output JSON is returned (instead of normal assistant answer).
6. If not a command: normal chat response path is used.

## Data Files
- `json/commands.json`: tracked in git (device definitions)
- `json/chat_history.json`: runtime data, ignored by git

## Troubleshooting
- If Ollama is unreachable:
  - verify service is up: `ollama serve`
  - verify endpoint: `http://localhost:11434`
- If microphone input fails:
  - check macOS microphone permission for Terminal/IDE
  - verify `portaudio` is installed
- If command detection misses:
  - expand aliases in `json/commands.json`
  - keep phrasing explicit (action + device)

## Deployment Suggestions
- Run inside a dedicated virtual environment.
- Pin model names and versions in `app/assistant_config.py`.
- Supervise process with `launchd`/`systemd` if running continuously.
- Keep `json/commands.json` under version control for reproducible behavior.
