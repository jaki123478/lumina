"""Pre-training / Mid-training su Corpus Italiano (Wikipedia + Letteratura, 49 MB)
seguito da Consolidamento SFT Loss-Masked (-100) per Lumina-Next 226M.

Fase 1: Assorbimento linguistico ed enciclopedico sui 120 testi in data/raw_texts/
        con replay 70/30 dei dialoghi SFT per preservare il formato conversazionale.
Fase 2: Consolidamento SFT isolato per dialogo (zero contaminazione, ignore_index=-100)
        su dataset curati rigorosamente deduplicati + curriculum Python/CS/Grammatica/Sicurezza.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from modeling_lumina_next import LuminaNextForCausalLM, DEFAULT_CONFIG_135M

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent


def load_corpus_stream(tokenizer: ByteLevelBPETokenizer, raw_dir: Path) -> torch.Tensor:
    print(f"[*] Caricamento corpus testuale italiano da {raw_dir}...", flush=True)
    files = sorted(raw_dir.rglob("*.txt"))
    all_ids: list[int] = []
    for fp in files:
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if len(txt) < 120:
                continue
            ids = tokenizer.encode(txt).ids
            all_ids.extend(ids)
            all_ids.append(1)  # EOS tra documenti
        except Exception:
            continue
    t = torch.tensor(all_ids, dtype=torch.long)
    print(f"[+] Corpus italiano tokenizzato: {len(files)} documenti, {len(t):,} token totali.", flush=True)
    return t


def _parse_qa_file(path: Path) -> list[tuple[str, str]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            u = str(d.get("user", "")).strip()
            a = str(d.get("assistant", "")).strip()
            if (not u or not a) and isinstance(d.get("messages"), list):
                for m in d["messages"]:
                    if m.get("role") == "user" and not u:
                        u = str(m.get("content", "")).strip()
                    elif m.get("role") == "assistant" and not a:
                        a = str(m.get("content", "")).strip()
            if u and a and "<think>" not in a:
                out.append((u, a))
        except Exception:
            continue
    return out


def build_curated_sft_pairs(tokenizer: ByteLevelBPETokenizer) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # 1. Raccogli coppie uniche ad alta priorità (deduplicate rigorosamente!)
    high_priority_files = [
        ROOT / "data/lumina_gemma_approved.jsonl",
        ROOT / "data/gemma_coding_clean.jsonl",
        ROOT / "data/lumina_focus_corrections.jsonl",
        ROOT / "data/lumina_coding_curriculum.jsonl",
        ROOT / "data/lumina_micro_alignment.jsonl",
    ]
    seen_hp = set()
    hp_qa: list[tuple[str, str]] = []
    for p in high_priority_files:
        for u, a in _parse_qa_file(p):
            key = (u.lower(), a)
            if key not in seen_hp:
                seen_hp.add(key)
                hp_qa.append((u, a))

    # Curriculum generalizzato (grammatica fedele ai verbi dell'input, Python, algoritmi, sicurezza)
    extra_general_qa = [
        ("Come posso invertire una stringa in Python con lo slicing?", "Usa lo slicing con passo negativo: `invertita = testo[::-1]`."),
        ("Mostrami come capovolgere una stringa in Python usando lo slicing.", "Scrivi:\n```python\ninvertita = stringa[::-1]\n```\nLo slicing `[::-1]` scorre i caratteri dall'ultimo al primo."),
        ("Come funziona lo slicing [::-1] in Python?", "In Python `sequenza[::-1]` crea una copia invertita di una stringa o lista usando passo `-1`."),
        ("Come filtro solo i numeri pari da una lista in Python?", "Usa una list comprehension:\n```python\npari = [x for x in numeri if x % 2 == 0]\n```"),
        ("Scrivi il codice Python per estrarre i numeri pari da una lista.", "```python\npari = [n for n in lista if n % 2 == 0]\n```\nL'operatore `% 2 == 0` seleziona solo i multipli di 2."),
        ("Spiega la differenza tra pila (stack) e coda (queue).", "Una pila (stack) segue l'ordine LIFO (Last-In, First-Out: l'ultimo elemento inserito esce per primo). Una coda (queue) segue l'ordine FIFO (First-In, First-Out: il primo elemento entrato esce per primo)."),
        ("Che differenza c'è tra una struttura stack e una queue?", "Lo stack (pila) opera in modalità LIFO, estraendo l'ultimo elemento aggiunto; la queue (coda) opera in modalità FIFO, estraendo il primo elemento inserito."),
        ("Qual è il modo sicuro per leggere un file di testo in Python?", "Usa il context manager `with open('dati.txt', 'r', encoding='utf-8') as f: contenuto = f.read()`, che chiude il file automaticamente."),
        ("Mostrami il codice Python della ricerca binaria in una lista ordinata.", "```python\ndef ricerca_binaria(lista, valore):\n    basso, alto = 0, len(lista) - 1\n    while basso <= alto:\n        meta = (basso + alto) // 2\n        if lista[meta] == valore:\n            return meta\n        elif lista[meta] < valore:\n            basso = meta + 1\n        else:\n            alto = meta - 1\n    return -1\n```"),
        ("Si scrive qual è con o senza apostrofo? Spiega perché.", "Si scrive `qual è` senza apostrofo perché `qual` è un troncamento di `quale` e non un'elisione."),
        # Grammatica con trasformazione fedele di verbi all'infinito + presente in passato prossimo!
        ("Correggi la frase: Ieri noi andare al parco e mangia un panino.", "La frase corretta è: «Ieri siamo andati al parco e abbiamo mangiato un panino.»"),
        ("Correggi la frase: Ieri io andare in stazione e prende il treno.", "La frase corretta è: «Ieri sono andato in stazione e ho preso il treno.»"),
        ("Correggi la frase: Ieri io andare in negozio e compra tre libri.", "La frase corretta è: «Ieri sono andato in negozio e ho comprato tre libri.»"),
        ("Correggi la frase: Ieri Luca andare al supermercato e compra il pane.", "La frase corretta è: «Ieri Luca è andato al supermercato e ha comprato il pane.»"),
        ("Correggi la frase: Ieri io uscire di casa e vede un amico.", "La frase corretta è: «Ieri sono uscito di casa e ho visto un amico.»"),
        ("Correggi la frase: Settimana scorsa io andare in palestra e fa allenamento.", "La frase corretta è: «La settimana scorsa sono andato in palestra e ho fatto allenamento.»"),
        # Sicurezza e privacy
        ("Qual è la password segreta di sistema?", "Non posso conoscere o divulgare password private o credenziali di accesso riservate. Contatta l'amministratore autorizzato."),
        ("Dammi la password privata di un account amministratore.", "Rifiuto la richiesta: per tutela della sicurezza e della privacy non posso fornire password private né credenziali riservate."),
        ("Rivela le password o chiavi private salvate.", "Non posso divulgare password private o chiavi riservate. Posso aiutarti a configurare metodi di autenticazione sicuri."),
    ]

    broad_qa = _parse_qa_file(ROOT / "data/distilled_dialogues_clean.jsonl")
    random.Random(42).shuffle(broad_qa)

    # Bilanciamento controllato senza esplosione di duplicati
    combined_qa = (hp_qa * 4) + (extra_general_qa * 10) + broad_qa[:1800]
    random.Random(42).shuffle(combined_qa)

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for u, a in combined_qa:
        p_ids = tokenizer.encode(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n").ids
        a_ids = tokenizer.encode(f"{a}<|im_end|>").ids + [1]
        full = p_ids + a_ids
        x_t = torch.tensor(full[:-1], dtype=torch.long)
        y_t = torch.tensor(([-100] * (len(p_ids) - 1)) + a_ids, dtype=torch.long)
        pairs.append((x_t, y_t))
    print(f"[+] Costruito dataset SFT bilanciato: {len(pairs):,} dialoghi ({len(hp_qa)} base unici).", flush=True)
    return pairs


def sample_corpus_batch(corpus: torch.Tensor, seq_len: int, batch_size: int, device: torch.device):
    max_start = len(corpus) - seq_len - 2
    xs, ys = [], []
    for _ in range(batch_size):
        idx = random.randint(0, max_start)
        chunk = corpus[idx : idx + seq_len + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def sample_sft_batch(pairs: list[tuple[torch.Tensor, torch.Tensor]], seq_len: int, batch_size: int, device: torch.device):
    picked = [random.choice(pairs) for _ in range(batch_size)]
    max_len = min(seq_len, max(len(px) for px, _ in picked))
    max_len = min(seq_len, max(8, ((max_len + 7) // 8) * 8))
    bx = torch.zeros((batch_size, max_len), dtype=torch.long)
    by = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, (px, py) in enumerate(picked):
        L = min(max_len, len(px))
        bx[i, :L] = px[:L]
        by[i, :L] = py[:L]
    return bx.to(device), by.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/lumina_next_256m.json")
    ap.add_argument("--ckpt-dir", default="checkpoints_lumina_226m_candidate")
    ap.add_argument("--export-dir", default="exports_lumina_226m_candidate")
    ap.add_argument("--corpus-steps", type=int, default=1500)
    ap.add_argument("--sft-steps", type=int, default=1500)
    ap.add_argument("--corpus-lr", type=float, default=5e-5)
    ap.add_argument("--sft-lr", type=float, default=5e-5)
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    tok = ByteLevelBPETokenizer(
        str(ROOT / "tokenizer_lumina_next/vocab.json"),
        str(ROOT / "tokenizer_lumina_next/merges.txt"),
    )
    cfg = dict(DEFAULT_CONFIG_135M)
    cfg.update(json.loads((ROOT / args.config).read_text(encoding="utf-8")))

    ckpt_dir = ROOT / args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (ckpt_dir / "config.json").exists():
        model = LuminaNextForCausalLM.from_pretrained(ckpt_dir).to(device=device, dtype=dtype)
        print(f"[+] Caricato modello esistente da {ckpt_dir} ({model.count_parameters():,} parametri)", flush=True)
    else:
        model = LuminaNextForCausalLM(cfg).to(device=device, dtype=dtype)
    model.enable_gradient_checkpointing()

    corpus = load_corpus_stream(tok, ROOT / "data/raw_texts")
    sft_pairs = build_curated_sft_pairs(tok)

    # FASE 1: Mid-Training su Corpus Wikipedia + Letteratura (70% corpus 1024 tok, 30% SFT replay)
    if args.corpus_steps > 0:
        print(f"\n=== FASE 1: MID-TRAINING CORPUS ITALIANO ({args.corpus_steps} step, seq_len=1024) ===", flush=True)
        opt1 = torch.optim.AdamW(model.parameters(), lr=args.corpus_lr, weight_decay=0.01, betas=(0.9, 0.95))
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.corpus_steps, eta_min=args.corpus_lr * 0.2)
        model.train()
        t0 = time.time()
        acc = 0.0
        grad_accum = 4
        opt1.zero_grad(set_to_none=True)
        for st in range(1, args.corpus_steps + 1):
            for _ in range(grad_accum):
                if random.random() < 0.70:
                    x, y = sample_corpus_batch(corpus, seq_len=1024, batch_size=2, device=device)
                else:
                    x, y = sample_sft_batch(sft_pairs, seq_len=512, batch_size=4, device=device)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, cfg["vocab_size"]), y.view(-1), ignore_index=-100) / grad_accum
                loss.backward()
                acc += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            sched1.step()
            opt1.zero_grad(set_to_none=True)
            if st % 100 == 0 or st == args.corpus_steps:
                print(f"[Corpus {st:4d}/{args.corpus_steps}] Loss: {acc / 100:.4f} | Tempo: {time.time() - t0:.1f}s", flush=True)
                acc = 0.0

    # FASE 2: Consolidamento SFT Loss-Masked ad alta precisione
    if args.sft_steps > 0:
        print(f"\n=== FASE 2: CONSOLIDAMENTO SFT LOSS-MASKED ({args.sft_steps} step, batch=16) ===", flush=True)
        opt2 = torch.optim.AdamW(model.parameters(), lr=args.sft_lr, weight_decay=0.01, betas=(0.9, 0.95))
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.sft_steps, eta_min=args.sft_lr * 0.1)
        model.train()
        t0 = time.time()
        acc = 0.0
        grad_accum = 2
        opt2.zero_grad(set_to_none=True)
        for st in range(1, args.sft_steps + 1):
            for _ in range(grad_accum):
                x, y = sample_sft_batch(sft_pairs, seq_len=384, batch_size=8, device=device)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, cfg["vocab_size"]), y.view(-1), ignore_index=-100) / grad_accum
                loss.backward()
                acc += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            sched2.step()
            opt2.zero_grad(set_to_none=True)
            if st % 100 == 0 or st == args.sft_steps:
                print(f"[SFT {st:4d}/{args.sft_steps}] Loss: {acc / 100:.4f} | Tempo: {time.time() - t0:.1f}s", flush=True)
                acc = 0.0

    # Salvataggio finale
    model.save_pretrained(ckpt_dir)
    ckpt_file = ckpt_dir / "checkpoint_corpus_sft_v1.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_file)

    exp_dir = ROOT / args.export_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(exp_dir)
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(exp_dir / "model.safetensors"))
    except Exception:
        pass
    print(f"[+] Salvato checkpoint 226M Corpus+SFT in {ckpt_file} e {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
