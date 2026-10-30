"""Terminale locale per Gemma 3 quantizzata (senza modificare Lumina).

Uso: .venv\\Scripts\\python.exe chat_gemma.py
Il runner usa il file GGUF già presente, più leggero del checkpoint PyTorch.
"""
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL = Path("models/gemma-3-4b-it-clean/gemma-3-4b-it-Q4_K_M.gguf")


def main() -> int:
    ap = argparse.ArgumentParser(description="Chat interattiva locale Gemma")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--ctx", type=int, default=2048, help="contesto massimo (basso per non appesantire il PC)")
    ap.add_argument("--gpu-layers", type=int, default=20, help="layer GPU; usa 0 per CPU")
    args = ap.parse_args()
    model = args.model.resolve()
    if not model.exists():
        raise SystemExit(f"Modello non trovato: {model}")
    try:
        from llama_cpp import Llama
    except ImportError:
        raise SystemExit(
            "Manca llama-cpp-python. Installa la dipendenza nell'ambiente locale "
            "e rilancia questo script; Lumina non viene modificata."
        )
    llm = Llama(model_path=str(model), n_ctx=args.ctx, n_gpu_layers=args.gpu_layers, verbose=False)
    print("Gemma pronto. Scrivi /esci per terminare, /reset per azzerare la conversazione.")
    history: list[dict[str, str]] = []
    while True:
        try:
            prompt = input("Tu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"/esci", "/exit", "/quit"}:
            break
        if prompt.lower() == "/reset":
            history.clear()
            print("Memoria conversazione azzerata.")
            continue
        history.append({"role": "user", "content": prompt})
        out = llm.create_chat_completion(messages=history, max_tokens=256, temperature=0.2)
        answer = out["choices"][0]["message"]["content"].strip()
        print(f"Gemma> {answer}\n")
        history.append({"role": "assistant", "content": answer})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
