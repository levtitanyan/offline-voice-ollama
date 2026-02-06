import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:1b"   # change to any model you have, e.g. "deepseek-r1:8b"

def ask(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        raise SystemExit(
            "❌ Cannot connect to Ollama at http://localhost:11434.\n"
            "Make sure Ollama is running (open the Ollama app, or run `ollama serve`).\n"
            f"Details: {e}"
        )

if __name__ == "__main__":
    print(f"Using model: {MODEL}")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        answer = ask(user)
        print("\nLLM:", answer.strip())
