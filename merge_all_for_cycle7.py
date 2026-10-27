"""
merge_all_for_cycle7.py - Unisce dataset v2, curriculum certificato AMD/Intel e il nuovo pacchetto CoT Scratchpad.
Priorità di insegnamento:
1. CoT Scratchpad (<think>...</think>) caricato per primo, sostituendo le vecchie risposte secche.
2. Cluster AMD / Intel certificati.
3. Dataset base v2.
Produce: data/lumina_train_dataset_v3_cot.jsonl
"""

import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

out_file = Path("data/lumina_train_dataset_v3_cot.jsonl")
seen_records = set()
cot_prompts = set()
merged_records = []

def add_record(u: str, a: str, cat: str, force: bool = False):
    u = u.strip()
    a = a.strip()
    if not u or not a or len(a) < 5:
        return
    
    # Se il prompt ha già una versione CoT ed è un record normale v2, non sovrascrivere il CoT
    if not force and u.lower() in cot_prompts:
        return

    key = (u.lower(), a[:40].lower())
    if key in seen_records:
        return
    seen_records.add(key)
    
    merged_records.append({
        "user": u,
        "assistant": a,
        "category": cat,
        "split": "train"
    })

print("=" * 65)
print("  🚀 FUSIONE DATASET CICLO 7: CoT SCRATCHPAD + AMD/INTEL + V2 BASE")
print("=" * 65)

# 1. Carica CoT Scratchpad con priorità assoluta
cot_file = Path("data/lumina_cot_scratchpad_1k.jsonl")
if cot_file.exists():
    with open(cot_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    u = d.get("user", "")
                    a = d.get("assistant", "")
                    if u and a:
                        cot_prompts.add(u.strip().lower())
                        add_record(u, a, d.get("category", "cot_scratchpad"), force=True)
                except Exception:
                    pass
    print(f"[+] 1/3 Pacchetto CoT Scratchpad caricato con priorità: {len(merged_records):,} campioni")

# 2. Cluster AMD / Intel
intel_curated = Path("C:/Users/Jaki1/Desktop/test/intel_outbox/curriculum_amd_curated_by_intel.jsonl")
if intel_curated.exists():
    count_before = len(merged_records)
    with open(intel_curated, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    u = d.get("prompt", "") or d.get("user", "") or d.get("instruction", "")
                    a = d.get("response", "") or d.get("assistant", "") or d.get("output", "")
                    add_record(u, a, d.get("category", "amd_intel_curated"))
                except Exception:
                    pass
    print(f"[+] 2/3 Campioni certificati AMD/Intel: +{len(merged_records) - count_before:,} nuovi unici")

# 3. Dataset Base v2 (solo campioni che non confliggono con il nuovo CoT)
v2_file = Path("data/lumina_train_dataset_v2.jsonl")
if v2_file.exists():
    count_before = len(merged_records)
    with open(v2_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    add_record(d.get("user", ""), d.get("assistant", ""), d.get("category", "v2_curated"))
                except Exception:
                    pass
    print(f"[+] 3/3 Dataset v2 base caricato: +{len(merged_records) - count_before:,} campioni")

# Salva dataset v3
with open(out_file, "w", encoding="utf-8") as f:
    for r in merged_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("-" * 65)
print(f"[SUCCESS] Dataset v3 CoT salvato in: {out_file}")
print(f"          Totale campioni verificati unici: {len(merged_records):,}")
print("=" * 65)
