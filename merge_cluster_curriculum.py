"""
merge_cluster_curriculum.py - Integra il curriculum distillato dal PC AMD e certificato dal PC Intel
nel dataset principale Lumina-Next (data/lumina_train_dataset_v2.jsonl).
"""

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

main_ds_path = Path("data/lumina_train_dataset_v2.jsonl")
amd_intel_path = Path("C:/Users/Jaki1/Desktop/test/intel_outbox/curriculum_amd_curated_by_intel.jsonl")
amd_raw_path = Path("C:/Users/Jaki1/Desktop/test/amd_results/qwen_generated_curriculum_amd.jsonl")

# Carica dataset esistente
existing_prompts = set()
with open(main_ds_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            existing_prompts.add(d.get("user", "").strip().lower())
        except Exception:
            pass

print(f"[*] Campioni esistenti nel dataset: {len(existing_prompts):,}")

# Carica e valida campioni AMD/Intel
new_records = []
src_files = [amd_intel_path, amd_raw_path]

for src in src_files:
    if not src.exists():
        continue
    print(f"[*] Esame sorgente: {src}")
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                u = item.get("prompt", "") or item.get("user", "") or item.get("instruction", "")
                a = item.get("response", "") or item.get("assistant", "") or item.get("output", "")
                u = u.strip()
                a = a.strip()
                if not u or not a or len(a) < 10:
                    continue
                norm_u = u.lower()
                if norm_u not in existing_prompts:
                    existing_prompts.add(norm_u)
                    new_records.append({
                        "user": u,
                        "assistant": a,
                        "category": item.get("category", "cluster_amd_intel_distilled"),
                        "split": "train",
                        "node_source": item.get("node", "PC_AMD_INTEL")
                    })
            except Exception:
                continue

print(f"[+] Nuovi campioni unici da aggiungere: {len(new_records):,}")

if new_records:
    with open(main_ds_path, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] Dataset aggiornato salvato con successo: {main_ds_path}")
else:
    print("[*] Nessun nuovo record da aggiungere.")
