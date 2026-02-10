"""
Offline Voice/Text Assistant:
- Voice input (r) -> Whisper -> Ollama chat
- Text input (t)  -> Ollama chat
- Quit (q)

- Persistent memory (json/chat_history.json)
- NLP command parsing to structured JSON when input is a device-control command

- System instructions management:
    /system <text>        -> REPLACE system prompt
    /addsystem <text>     -> APPEND to system prompt (keeps default + adds facts)

- Reset commands:
    /clear                -> clear conversation messages, keep current system prompt
    /reset                -> reset system prompt to default and clear conversation
"""

import tempfile
from typing import List

from faster_whisper import WhisperModel

if __package__:
    from .assistant_audio import record_wav, speak_macos, transcribe
    from .assistant_commands import load_command_config, maybe_handle_command
    from .assistant_config import Message, OLLAMA_MODEL, RECORD_SECONDS, SYSTEM_PROMPT, WHISPER_MODEL
    from .assistant_history import ensure_system, load_history, save_history, trim_history
    from .assistant_ollama import ask_ollama_chat
else:
    # Allow direct execution: `python app/voice_to_ollama.py`
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from app.assistant_audio import record_wav, speak_macos, transcribe
    from app.assistant_commands import load_command_config, maybe_handle_command
    from app.assistant_config import Message, OLLAMA_MODEL, RECORD_SECONDS, SYSTEM_PROMPT, WHISPER_MODEL
    from app.assistant_history import ensure_system, load_history, save_history, trim_history
    from app.assistant_ollama import ask_ollama_chat


def chat_turn(history: List[Message], user_text: str, command_config: dict) -> List[Message]:
    """Process one user turn and update history."""
    command_history = maybe_handle_command(history, user_text, command_config)
    if command_history is not None:
        return command_history

    history.append({"role": "user", "content": user_text})
    history = trim_history(history)

    answer = ask_ollama_chat(history)

    history.append({"role": "assistant", "content": answer})
    history = trim_history(history)
    save_history(history)

    print(f"\nAI {answer}\n")
    speak_macos(answer)
    return history


def main() -> None:
    """Run the interactive assistant loop."""
    print("--------------------------------------------------------------------------")
    print("Commands: [r]=record | [t]=text | [q]=quit | /reset | /clear | /system <text> | /addsystem <text>\n")
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print(f"Using Whisper model: {WHISPER_MODEL}\n")
    print("--------------------------------------------------------------------------")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    history = load_history()
    history = ensure_system(history, SYSTEM_PROMPT)
    command_config = load_command_config()

    history = trim_history(history)
    save_history(history)

    while True:
        cmd = input(" Commands: r=record | t=text | q=quit: ").strip()

        if cmd.lower() == "q":
            history = ensure_system(history, SYSTEM_PROMPT)
            history = [history[0]]
            save_history(history)
            print("Conversation over. See you soon!")
            break

        if cmd.startswith("/reset"):
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            save_history(history)
            print("Factory reset: system prompt restored + history cleared.\n")
            continue

        if cmd.startswith("/clear"):
            history = ensure_system(history, SYSTEM_PROMPT)
            history = [history[0]]
            save_history(history)
            print("Clear: conversation cleared, system kept.\n")
            continue

        if cmd.startswith("/system "):
            new_sys = cmd[len("/system ") :].strip()
            if new_sys:
                history = [{"role": "system", "content": new_sys}] + [
                    m for m in history if m["role"] != "system"
                ]
                history = trim_history(history)
                save_history(history)
                print("System instructions replaced.\n")
            continue

        if cmd.startswith("/addsystem "):
            extra = cmd[len("/addsystem ") :].strip()
            if extra:
                history = ensure_system(history, SYSTEM_PROMPT)
                history[0]["content"] = history[0]["content"].rstrip() + "\n" + extra
                history = trim_history(history)
                save_history(history)
                print("System instructions appended.\n")
            continue

        if cmd.lower() == "t":
            user_text = input("Your Input: ").strip()
            if user_text:
                history = chat_turn(history, user_text, command_config)
            continue

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

