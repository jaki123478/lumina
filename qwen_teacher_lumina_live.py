"""
qwen_teacher_lumina_live.py - Sistema Autonomo di Insegnamento Continuo Qwen -> Lumina con Visualizzazione Live.
Caratteristiche:
1. Qwen (Teacher 0.5B Instruct su GPU RTX 5060) genera continuamente domande e risposte maestre.
2. Lumina-Next 135M (Student Step 8000 su GPU RTX 5060) risponde in tempo reale alla stessa domanda.
3. Mostra a schermo in diretta sia la risposta di Qwen che quella di Lumina con colori, token/s e latenze.
4. Salva i dialoghi per l'addestramento continuo e aggiorna test/status.json per il cluster.
"""

import sys
import os
import time
import json
import random
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if sys.platform == "win32":
    os.system("chcp 65001 >nul")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_lumina_next import LuminaNextForCausalLM
from tokenizers import ByteLevelBPETokenizer

# Colori ANSI
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

DOMAINS = [
    {
        "cat": "matematica_aritmetica",
        "name": "Aritmetica e Scomposizione Mentale",
        "system": "Sei un maestro di matematica. Genera una domanda di moltiplicazione o calcolo a due cifre in italiano (es. 'Quanto fa 24 per 6?').",
        "solve_sys": "Risolvi il calcolo mostrando prima la scomposizione passo-passo (es. 20*6 + 4*6) e poi il totale esatto."
    },
    {
        "cat": "logica_temporale",
        "name": "Logica Temporale e Calendario",
        "system": "Sei un logico. Genera una domanda sui giorni della settimana (es. 'Se oggi è giovedì, che giorno sarà tra tre giorni?').",
        "solve_sys": "Rispondi indicando il giorno finale esatto e la sequenza dei giorni intermedi tra parentesi."
    },
    {
        "cat": "coding_python",
        "name": "Programmazione Python e Algoritmi",
        "system": "Sei un docente di informatica. Genera una domanda pratica su una funzione Python, lista, ciclo o algoritmo.",
        "solve_sys": "Fornisci una risposta sintetica con il codice Python pulito racchiuso tra backtick e una breve spiegazione."
    },
    {
        "cat": "grammatica_italiana",
        "name": "Grammatica e Regole Linguistiche",
        "system": "Sei un linguista dell'Accademia della Crusca. Genera una frase con un errore comune da correggere (es. ausiliare essere/avere, qual è, apostrofo).",
        "solve_sys": "Fornisci la forma corretta esatta e spiega la regola grammaticale in modo chiaro e autorevole."
    },
    {
        "cat": "scienza_geografia",
        "name": "Scienza, Natura e Geografia",
        "system": "Sei un enciclopedista. Genera una domanda precisa su biologia, fisica, geografia fisica italiana o astronomia.",
        "solve_sys": "Fornisci una risposta scientificamente rigorosa, chiara ed esaustiva (es. fotosintesi, fiumi, laghi, pianeti)."
    },
    {
        "cat": "onesta_epistemica",
        "name": "Onestà Epistemica e Limiti Temporali",
        "system": "Genera una domanda su un evento futuro ipotetico (es. 'Chi vincerà le elezioni nel 2045?').",
        "solve_sys": "Dichiara con onestà che l'evento appartiene al futuro e non può essere conosciuto o previsto con certezza."
    }
]


def main():
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{CYAN}{BOLD}   🎓 LUMINA AUTONOMOUS MASTER: QWEN INSEGNA A LUMINA IN DIRETTA   {RESET}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{WHITE}• Teacher Model : {YELLOW}Qwen2.5-0.5B-Instruct (GPU CUDA BF16){RESET}")
    print(f"{WHITE}• Student Model : {GREEN}Lumina-Next 135M Step 8000 (GPU CUDA BF16){RESET}")
    print(f"{WHITE}• Cluster State : {MAGENTA}Sincronizzazione P2P Syncthing attiva su status.json{RESET}")
    print(f"{GRAY}Premi Ctrl+C per fermare la sessione.{RESET}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # 1. Carica Qwen
    print(f"[*] Caricamento Qwen 0.5B Teacher...")
    t0 = time.time()
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtype, device_map=device)
    print(f"[+] Qwen pronto in {time.time()-t0:.2f}s")

    # 2. Carica Lumina-Next
    print(f"[*] Caricamento Lumina-Next 135M Step 8000...")
    t0 = time.time()
    tok_dir = Path("tokenizer_lumina_next")
    lumina_tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    lumina = LuminaNextForCausalLM.from_pretrained("checkpoints_lumina_next").to(device=device, dtype=dtype).eval()
    print(f"[+] Lumina-Next pronto in {time.time()-t0:.2f}s (VRAM usata: {torch.cuda.memory_allocated()/(1024**2):.1f} MB)\n")

    out_file = Path("data/distilled_dialogues.jsonl")
    cluster_stream = Path("C:/Users/Jaki1/Desktop/test/amd_results/qwen_generated_curriculum_amd.jsonl")
    status_file = Path("C:/Users/Jaki1/Desktop/test/status.json")

    cycle = 1
    while True:
        try:
            domain = random.choice(DOMAINS)
            print(f"{GRAY}{'=' * 75}{RESET}")
            print(f"{BOLD}{CYAN}📚 LEZIONE #{cycle:04d} — DOMINIO: {domain['name'].upper()}{RESET}")
            print(f"{GRAY}{'=' * 75}{RESET}")

            # 1. Qwen crea la domanda / sfida
            q_msgs = [
                {"role": "system", "content": domain["system"]},
                {"role": "user", "content": "Genera una singola domanda o esercizio interessante e chiaro per il tuo studente Lumina. Scrivi SOLO la domanda."}
            ]
            q_text = qwen_tok.apply_chat_template(q_msgs, tokenize=False, add_generation_prompt=True)
            q_in = qwen_tok([q_text], return_tensors="pt").to(device)

            with torch.no_grad():
                q_out = qwen.generate(**q_in, max_new_tokens=60, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
            raw_q = qwen_tok.decode(q_out[0][len(q_in.input_ids[0]):], skip_special_tokens=True).strip()
            prompt = raw_q.split("\n")[0].replace('"', '').strip()
            if not prompt or len(prompt) < 10:
                prompt = "Quanto fa 18 per 7?"

            print(f"{YELLOW}{BOLD}❓ Domanda del Maestro (Qwen):{RESET} {WHITE}{prompt}{RESET}\n", flush=True)

            # 2. Qwen genera la soluzione maestra
            sol_msgs = [
                {"role": "system", "content": domain["solve_sys"]},
                {"role": "user", "content": prompt}
            ]
            sol_text = qwen_tok.apply_chat_template(sol_msgs, tokenize=False, add_generation_prompt=True)
            sol_in = qwen_tok([sol_text], return_tensors="pt").to(device)

            t_qwen = time.perf_counter()
            with torch.no_grad():
                sol_out = qwen.generate(**sol_in, max_new_tokens=90, temperature=0.3, top_p=0.9)
            el_qwen = time.perf_counter() - t_qwen
            tok_qwen = len(sol_out[0]) - len(sol_in.input_ids[0])
            spd_qwen = tok_qwen / el_qwen if el_qwen > 0 else 0
            qwen_answer = qwen_tok.decode(sol_out[0][len(sol_in.input_ids[0]):], skip_special_tokens=True).strip()

            print(f"{MAGENTA}{BOLD}👨‍🏫 Risposta di Qwen (Maestro):{RESET}", flush=True)
            print(f"  {WHITE}{qwen_answer}{RESET}", flush=True)
            print(f"  {GRAY}[Latenza Qwen: {el_qwen:.2f}s | {spd_qwen:.1f} tok/s]{RESET}\n", flush=True)

            # 3. Lumina risponde alla stessa domanda
            lum_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            lum_in = torch.tensor([lumina_tok.encode(lum_prompt).ids], device=device, dtype=torch.long)

            t_lum = time.perf_counter()
            with torch.no_grad():
                lum_out = lumina.generate(lum_in, max_new_tokens=85, temperature=0.2, top_p=0.9)
            el_lum = time.perf_counter() - t_lum
            tok_lum = len(lum_out[0]) - len(lum_in[0])
            spd_lum = tok_lum / el_lum if el_lum > 0 else 0
            lum_reply = lumina_tok.decode(lum_out[0][len(lum_in[0]):].tolist()).strip()
            if "<|im_end|>" in lum_reply:
                lum_reply = lum_reply.split("<|im_end|>")[0].strip()

            print(f"{GREEN}{BOLD}🤖 Risposta di Lumina (Studente Step 8000):{RESET}", flush=True)
            print(f"  {GREEN}{lum_reply}{RESET}", flush=True)
            print(f"  {GRAY}[Latenza Lumina: {el_lum:.2f}s | {spd_lum:.1f} tok/s]{RESET}\n", flush=True)

            # 4. Registra dialogo
            record = {
                "cycle": cycle,
                "category": domain["cat"],
                "user": prompt,
                "qwen_teacher": qwen_answer,
                "lumina_student": lum_reply,
                "timestamp": time.time(),
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"user": prompt, "assistant": qwen_answer, "category": domain["cat"]}, ensure_ascii=False) + "\n")

            if cluster_stream.parent.exists():
                with open(cluster_stream, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"user": prompt, "assistant": qwen_answer, "category": domain["cat"]}, ensure_ascii=False) + "\n")

            # Aggiorna status cluster ogni 5 cicli
            if cycle % 5 == 0 and status_file.exists():
                try:
                    s_data = json.loads(status_file.read_text(encoding="utf-8"))
                    s_data["nodes"]["lenovo_rtx_5060"]["live_teacher_cycle"] = cycle
                    s_data["nodes"]["lenovo_rtx_5060"]["last_lesson"] = prompt
                    s_data["nodes"]["lenovo_rtx_5060"]["heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S+01:00")
                    status_file.write_text(json.dumps(s_data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

            cycle += 1
            time.sleep(2.0)

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Chiusura sessione di insegnamento live.{RESET}")
            break
        except Exception as e:
            print(f"[!] Errore durante il ciclo: {e}")
            time.sleep(3.0)


if __name__ == "__main__":
    main()
