"""
evaluate_amd_benchmark_suite.py - Valutazione di Lumina-Next sulle 20 domande create dal PC AMD.
Esegue l'inferenza per ciascuna domanda della suite 'domande_benchmark_v2.json' e produce
un rapporto dettagliato salvato in local data/ e sincronizzato in test/lenovo_status/.
"""

import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from modeling_lumina_next import LuminaNextForCausalLM

ROOT = Path(__file__).resolve().parent
AMD_BENCHMARK_FILE = Path(r"C:\Users\Jaki1\Desktop\test\amd_results\domande_benchmark_v2.json")
OUTPUT_REPORT_JSON = ROOT / "data" / "evaluation_amd_suite_v2.json"
OUTPUT_REPORT_MD = Path(r"C:\Users\Jaki1\Desktop\test\lenovo_status\VALUTAZIONE_BENCHMARK_AMD_V2.md")


def run_evaluation():
    print("=" * 75)
    print(" 🧪 VALUTAZIONE LUMINA-NEXT SULLA TEST SUITE V2 DEL PC AMD")
    print("=" * 75)

    if not AMD_BENCHMARK_FILE.exists():
        print(f"[!] File benchmark non trovato: {AMD_BENCHMARK_FILE}")
        return

    suite_data = json.loads(AMD_BENCHMARK_FILE.read_text(encoding="utf-8"))
    questions = suite_data.get("questions", [])
    print(f"[*] Domande da valutare: {len(questions)} (Suite: {suite_data.get('suite_name')})")

    # Carica Tokenizer e Modello Lumina-Next
    tok_dir = ROOT / "tokenizer_lumina_next"
    tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    ckpt_dir = ROOT / "checkpoints_lumina_next"
    print(f"[*] Caricamento pesi Lumina-Next da '{ckpt_dir}' su {device}...")
    t0 = time.time()
    model = LuminaNextForCausalLM.from_pretrained(ckpt_dir).to(device=device, dtype=dtype)
    model.eval()
    print(f"[+] Modello caricato in {time.time()-t0:.2f}s (VRAM: {torch.cuda.memory_allocated()/(1024**2):.1f} MB).\n")

    results = []
    print("-" * 75)

    for idx, q in enumerate(questions, 1):
        q_id = q.get("id", f"Q{idx}")
        cat = q.get("category", "")
        prompt = q.get("prompt", "")
        expected = q.get("risposta_attesa_riferimento", "")

        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor([tokenizer.encode(chat_prompt).ids], device=device)

        t_gen_start = time.time()
        with torch.no_grad():
            out_ids = model.generate(input_ids, max_new_tokens=65, temperature=0.3, top_p=0.9)
        gen_time = time.time() - t_gen_start

        full_text = tokenizer.decode(out_ids[0].tolist())
        if "<|im_start|>assistant\n" in full_text:
            ans = full_text.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in ans:
                ans = ans.split("<|im_end|>")[0]
        else:
            ans = full_text
        ans = ans.strip()

        gen_tokens = len(out_ids[0]) - len(input_ids[0])
        tok_s = gen_tokens / gen_time if gen_time > 0 else 0

        print(f"[{idx:2d}/20] {q_id} ({cat})")
        print(f"  D: {prompt[:70]}...")
        print(f"  R: {ans[:80]}...")
        print(f"  -> {gen_tokens} token generati in {gen_time:.2f}s ({tok_s:.1f} tok/s)\n")

        results.append({
            "id": q_id,
            "category": cat,
            "prompt": prompt,
            "expected_reference": expected,
            "lumina_answer": ans,
            "tokens_generated": gen_tokens,
            "generation_time_s": round(gen_time, 2),
            "tok_per_sec": round(tok_s, 1)
        })

    # Salva JSON locale
    report_data = {
        "timestamp": time.time(),
        "checkpoint": "checkpoints_lumina_next (Step 2500)",
        "suite_evaluated": suite_data.get("suite_name"),
        "total_questions": len(results),
        "results": results
    }
    OUTPUT_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_JSON.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Salva Markdown per Syncthing (letto da AMD e Intel)
    md_lines = [
        "# 📊 Valutazione Lumina-Next sulla Test Suite v2 (Fornita dal PC AMD)\n\n",
        f"- **Data Valutazione**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"- **Modello Valutato**: Lumina-Next 135M (Checkpoint Step 2.500 | Loss: `1.5491`)\n",
        f"- **Hardware Esecuzione**: PC Principale Lenovo (NVIDIA GeForce RTX 5060 Laptop GPU)\n",
        f"- **Suite Test**: `{suite_data.get('suite_name')}` ({len(results)} domande)\n\n",
        "---\n\n",
        "## Risposte Fornite da Lumina-Next\n\n"
    ]

    for r in results:
        md_lines.append(f"### [{r['id']}] {r['category']}\n")
        md_lines.append(f"**Domanda**: {r['prompt']}\n\n")
        md_lines.append(f"**Risposta Lumina**: *\"{r['lumina_answer']}\"*\n\n")
        md_lines.append(f"> **Riferimento Atteso**: {r['expected_reference']}\n\n")
        md_lines.append(f"*Throughput: {r['tok_per_sec']} tok/s ({r['tokens_generated']} token in {r['generation_time_s']}s)*\n\n---\n\n")

    OUTPUT_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_MD.write_text("".join(md_lines), encoding="utf-8")

    print("=" * 75)
    print(" 🎉 VALUTAZIONE DELLA TEST SUITE AMD COMPLETATA!")
    print(f" Report locale salvato in      : {OUTPUT_REPORT_JSON}")
    print(f" Report sincronizzato Syncthing: {OUTPUT_REPORT_MD}")
    print("=" * 75)

    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_evaluation()
