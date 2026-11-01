"""Addestramento continuo (Cycle 2) Loss-Masked per Lumina-Next 226M.
Include consapevolezza temporale (sport, F1, attualità vs storia),
coding avanzato, logica, scienza e identità sovereign 100% italiana.
"""
from __future__ import annotations

import json
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

EXTRA_CURRICULUM_QA = [
    # Consapevolezza temporale: Formula 1, sport e attualità
    ("Chi ha vinto l'ultimo Gran Premio di Formula 1?",
     "I risultati delle gare recenti richiedono dati in tempo reale. In locale offline conosco la storia della Formula 1 e le regole sportive; per i vincitori e le classifiche dell'ultimo GP consulta le fonti live o abilita la ricerca web."),
    ("Cosa è successo nell'ultimo Gran Premio di F1?",
     "Per conoscere gli eventi e i vincitori dell'ultimo Gran Premio di Formula 1 servono informazioni aggiornate in tempo reale. In modalità offline posso spiegarti la storia dei circuiti, i regolamenti e l'aerodinamica delle vetture."),
    ("Chi ha vinto il mondiale di Formula 1?",
     "I piloti con il record di mondiali di Formula 1 sono Michael Schumacher e Lewis Hamilton con 7 titoli mondiali ciascuno. Per le classifiche della stagione in corso è necessaria la consultazione in tempo reale."),
    ("Chi era Ayrton Senna?",
     "Ayrton Senna è stato uno dei più grandi piloti nella storia della Formula 1. Ha vinto tre campionati mondiali con la McLaren (1988, 1990, 1991), celebre per la sua velocità straordinaria sul bagnato e la sua determinazione."),
    ("Chi ha vinto la partita di ieri?",
     "Non posso conoscere i risultati delle partite giocate di recente senza una connessione a Internet in tempo reale. In modalità offline puoi chiedermi statistiche storiche, regole sportive o strategie di gioco."),
    # Scienza e astrofisica
    ("Cos'è un buco nero?",
     "Un buco nero è una regione dello spazio-tempo con una forza di gravità così forte che nulla, neppure la luce, può sfuggire al suo interno. Si genera solitamente dal collasso gravitazionale di una stella molto massiccia al termine della sua vita. Il confine invisibile che delimita il punto di non ritorno è l'orizzonte degli eventi."),
    ("Raccontami una curiosità sullo spazio.",
     "Nello spazio regna il silenzio assoluto perché non esiste aria o atmosfera per propagare le onde sonore. Inoltre, su Venere un giorno dura più di un anno: impiega circa 243 giorni terrestri per compiere una rotazione su se stesso e solo 225 per girare attorno al Sole."),
    ("Cosa sono le onde gravitazionali?",
     "Le onde gravitazionali sono increspature dello spazio-tempo generate da eventi cosmici violentissimi, come la collisione e fusione di due buchi neri o stelle di neutroni. Sono state previste da Albert Einstein nel 1916 e confermate sperimentalmente per la prima volta nel 2015 dagli interferometri LIGO e Virgo."),
    # Coding Python e Informatica
    ("Scrivi una funzione Python che calcola la media di una lista.",
     "```python\ndef calcola_media(numeri):\n    if not numeri:\n        return None\n    return sum(numeri) / len(numeri)\n```\nSe la lista è vuota restituisce `None`, altrimenti calcola la somma divisa per il conteggio degli elementi."),
    ("Come inverto una lista in Python senza modificarla?",
     "Puoi usare lo slicing:\n```python\ninvertita = lista[::-1]\n```\nOppure la funzione integrata `reversed()` convertita in lista:\n```python\ninvertita = list(reversed(lista))\n```"),
    ("Qual è la differenza tra RAM e memoria SSD?",
     "La RAM è una memoria volatile ad altissima velocità utilizzata dal processore per i dati e i programmi in esecuzione, che si svuota allo spegnimento. L'SSD è una memoria di massa non volatile su chip flash che conserva in modo permanente il sistema operativo, i file e i dati anche senza alimentazione."),
    # Identità e conversazione
    ("Lumina mi senti?",
     "Sì, ti sento forte e chiaro! Di cosa vogliamo parlare?"),
    ("Ciao Lumina come va?",
     "Tutto bene e operativa in locale sul tuo PC! Come posso aiutarti oggi?"),
    ("Chi ti ha creata?",
     "Sono Lumina, un assistente di intelligenza artificiale sviluppato da Jacopo Forti ed eseguito interamente in locale sul tuo computer."),
]


def load_all_curated_pairs(tokenizer: ByteLevelBPETokenizer) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from train_lumina_226m_corpus_sft import _parse_qa_file

    high_priority_files = [
        ROOT / "data/lumina_gemma_approved.jsonl",
        ROOT / "data/gemma_coding_clean.jsonl",
        ROOT / "data/lumina_focus_corrections.jsonl",
        ROOT / "data/lumina_coding_curriculum.jsonl",
        ROOT / "data/lumina_micro_alignment.jsonl",
    ]
    seen = set()
    qa_list: list[tuple[str, str]] = []
    for p in high_priority_files:
        for u, a in _parse_qa_file(p):
            key = (u.lower().strip(), a.strip())
            if key not in seen:
                seen.add(key)
                qa_list.append((u, a))

    for u, a in EXTRA_CURRICULUM_QA:
        key = (u.lower().strip(), a.strip())
        if key not in seen:
            seen.add(key)
            qa_list.append((u, a))

    broad = _parse_qa_file(ROOT / "data/distilled_dialogues_clean.jsonl")
    random.Random(1337).shuffle(broad)

    # Dataset bilanciato: alta priorità moltiplicata + extra + campione di background
    dataset = (qa_list * 3) + (EXTRA_CURRICULUM_QA * 12) + broad[:1200]
    random.Random(1337).shuffle(dataset)

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for u, a in dataset:
        p_ids = tokenizer.encode(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n").ids
        a_ids = tokenizer.encode(f"{a}<|im_end|>").ids + [1]
        full = p_ids + a_ids
        x_t = torch.tensor(full[:-1], dtype=torch.long)
        y_t = torch.tensor(([-100] * (len(p_ids) - 1)) + a_ids, dtype=torch.long)
        pairs.append((x_t, y_t))
    print(f"[+] Dataset SFT Cycle 2: {len(pairs):,} dialoghi pronti ({len(qa_list)} unici ad alta priorità).", flush=True)
    return pairs


def sample_batch(pairs, seq_len: int, batch_size: int, device: torch.device):
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
    steps = 800
    lr = 4e-5
    batch_size = 8
    grad_accum = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"=== AVVIO ADDESTRAMENTO LUMINA 226M (CYCLE 2) SU {device.type.upper()} ===", flush=True)
    tok = ByteLevelBPETokenizer(
        str(ROOT / "tokenizer_lumina_next/vocab.json"),
        str(ROOT / "tokenizer_lumina_next/merges.txt"),
    )
    ckpt_dir = ROOT / "checkpoints_lumina_226m_candidate"
    model = LuminaNextForCausalLM.from_pretrained(ckpt_dir).to(device=device, dtype=dtype)
    print(f"[+] Modello caricato: {model.count_parameters():,} parametri", flush=True)

    pairs = load_all_curated_pairs(tok)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.15)

    model.train()
    model.enable_gradient_checkpointing()
    t0 = time.time()
    acc_loss = 0.0
    opt.zero_grad(set_to_none=True)

    for st in range(1, steps + 1):
        for _ in range(grad_accum):
            x, y = sample_batch(pairs, seq_len=384, batch_size=batch_size, device=device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, model.config["vocab_size"]), y.view(-1), ignore_index=-100) / grad_accum
            loss.backward()
            acc_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if st % 100 == 0 or st == steps:
            cur_lr = sched.get_last_lr()[0]
            avg_loss = acc_loss / 100.0
            elapsed = time.time() - t0
            tok_per_sec = (st * batch_size * grad_accum * 160) / max(0.1, elapsed)
            print(f"[Cycle 2 Step {st:4d}/{steps}] Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | Speed: {tok_per_sec:.0f} tok/s | Tempo: {elapsed:.1f}s", flush=True)
            acc_loss = 0.0

    # Salvataggio pesi consolidati
    model.eval()
    model.save_pretrained(ckpt_dir)
    ckpt_file = ckpt_dir / "checkpoint_corpus_sft_v2.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": model.config}, ckpt_file)

    exp_dir = ROOT / "exports_lumina_226m_candidate"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(exp_dir)
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(exp_dir / "model.safetensors"))
    except Exception:
        pass

    # Copia file tokenizer per renderlo immediatamente caricabile da chiunque
    import shutil
    shutil.copy(ROOT / "tokenizer_lumina_next/vocab.json", exp_dir / "vocab.json")
    shutil.copy(ROOT / "tokenizer_lumina_next/merges.txt", exp_dir / "merges.txt")
    print(f"\n[+] Addestramento Cycle 2 completato con successo in {time.time() - t0:.1f}s!", flush=True)
    print(f"[+] Pesi salvati in {ckpt_file} e {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
