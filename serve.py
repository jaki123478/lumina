"""
serve.py - Web Server & OpenAI-Compatible REST API Server per Lumina.

Supporta:
- Web App UI (dist/)
- API Monitor di addestramento (/api/auto_status, /api/recent_dialogues)
- API Protezione Sentinel (/api/check_safety)
- API Ufficiale OpenAI v1 (/v1/models, /v1/chat/completions)
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import re
import sys
import time
import uuid
import webbrowser
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

from lumina_network_policy import get_network_policy

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

# Cache per i modelli caricati
_MODELS = {}
_SENTINEL = None
_CODER_MODEL = None
_CODER_TOKENIZER = None
_MEMORY = None


def get_memory():
    global _MEMORY
    if _MEMORY is None:
        try:
            from lumina_memory import LuminaMemory
            _MEMORY = LuminaMemory()
        except Exception as e:
            print(f"[API] Errore inizializzazione Lumina Memory: {e}")
            _MEMORY = False
    return _MEMORY if _MEMORY is not False else None


def get_coder_model():
    global _CODER_MODEL, _CODER_TOKENIZER
    if _CODER_MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[API] Inizializzazione Lumina Coder Engine ({model_id}) su {device}...")
        _CODER_TOKENIZER = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        _CODER_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            local_files_only=True
        )
        print(f"[API] Lumina Coder Engine pronto per programmare su {device}!")
    return _CODER_MODEL, _CODER_TOKENIZER


def get_sentinel():
    global _SENTINEL
    if _SENTINEL is None:
        try:
            from lumina_guardrails import LuminaSentinel
            _SENTINEL = LuminaSentinel()
        except Exception as e:
            print(f"[API] Impossibile inizializzare Lumina Sentinel: {e}")
            _SENTINEL = False
    return _SENTINEL if _SENTINEL is not False else None


_NEXT_MODEL = None
_NEXT_TOKENIZER = None

def get_next_model():
    global _NEXT_MODEL, _NEXT_TOKENIZER
    if _NEXT_MODEL is None:
        import torch
        from modeling_lumina_next import LuminaNextForCausalLM
        from tokenizers import ByteLevelBPETokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        from lumina_next_runtime import active_model_dir
        folder = active_model_dir()
        print(f"[API] Caricamento Lumina Next (135M RoPE/YaRN) su {device}...")
        _NEXT_TOKENIZER = ByteLevelBPETokenizer(str(folder / "vocab.json"), str(folder / "merges.txt"))
        _NEXT_MODEL = LuminaNextForCausalLM.from_pretrained(folder).to(device=device, dtype=dtype)
        _NEXT_MODEL.eval()
        print(f"[API] Lumina Next (135M) pronto per inferenza long-context!")
    return _NEXT_MODEL, _NEXT_TOKENIZER


def get_lumina_model(model_name: str = "lumina-2.75m"):
    global _MODELS
    import torch
    from modeling_lumina import LuminaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = model_name.lower()

    if "14m" in name or "xlarge" in name:
        key = "14m"
        folder = ROOT / "hf_export_lumina_xlarge"
    else:
        key = "2.75m"
        folder = ROOT / "hf_export_lumina_large"

    if key not in _MODELS:
        print(f"[API] Caricamento modello Lumina ({key}) su {device}...")
        try:
            m = LuminaForCausalLM.from_pretrained(folder)
            m.to(device)
            _MODELS[key] = (m, device)
            print(f"[API] Modello Lumina ({key}) pronto!")
        except Exception as e:
            print(f"[API ERRORE] Impossibile caricare {folder}: {e}")
            return None, device

    return _MODELS.get(key)


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Permissions-Policy", "microphone=(self)")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # 1. API Monitor Status
        if self.path == "/api/auto_status":
            status_file = ROOT / "data" / "auto_train_status.json"
            content = status_file.read_text(encoding="utf-8") if status_file.exists() else "{}"
            body = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 2. API Dialoghi Recenti
        if self.path == "/api/recent_dialogues":
            distilled_file = ROOT / "data" / "distilled_dialogues.jsonl"
            lines = []
            if distilled_file.exists():
                with open(distilled_file, "r", encoding="utf-8") as f:
                    all_lines = [l.strip() for l in f if l.strip()]
                    for l in all_lines[-15:]:
                        try:
                            lines.append(json.loads(l))
                        except Exception:
                            pass
            body = json.dumps(list(reversed(lines)), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 2b. API Memoria a Lungo Termine
        if self.path == "/api/memory":
            mem = get_memory()
            profile = mem.get_all_profile() if mem else {}
            facts = mem.recall_facts(limit=10) if mem else []
            body = json.dumps({"profile": profile, "recent_facts": facts}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 2c. API Politica di Rete ed Esplicito Consenso (/api/network_policy)
        if self.path in ("/api/network_policy", "/api/v1/network_policy"):
            pol = get_network_policy()
            from lumina_web_sources import enabled as web_sources_enabled
            body = json.dumps({**pol.policy,'web_sources_enabled':web_sources_enabled()}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 3. API Ufficiale OpenAI: Lista Modelli (/v1/models)
        if self.path in ("/v1/models", "/api/v1/models"):
            models_payload = {
                "object": "list",
                "data": [
                    {"id":"lumina-assisted","object":"model","owned_by":"lumina",
                     "backend":"Qwen/Qwen2.5-Coder-1.5B-Instruct locale e calcolatrice locale"},
                    {
                        "id": "lumina-2.75m",
                        "object": "model",
                        "created": 1793000000,
                        "owned_by": "lumina",
                        "permission": [],
                        "root": "lumina-2.75m",
                        "parent": None
                    },
                    {
                        "id": "lumina-14m",
                        "object": "model",
                        "created": 1793000000,
                        "owned_by": "lumina",
                        "permission": [],
                        "root": "lumina-14m",
                        "parent": None
                    },
                    {
                        "id": "lumina-coder",
                        "object": "model",
                        "created": 1793000000,
                        "owned_by": "lumina",
                        "permission": [],
                        "root": "lumina-coder",
                        "parent": None
                    },
                    {
                        "id": "lumina-next",
                        "object": "model",
                        "created": 1793000000,
                        "owned_by": "lumina",
                        "permission": [],
                        "root": "lumina-next",
                        "parent": None
                    }
                ]
            }
            body = json.dumps(models_payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        super().do_GET()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"

        if self.path == '/api/tts':
            try:
                from lumina_voice_neural import synthesize_bytes
                text = str(json.loads(raw_body).get('text', ''))
                if not text.strip() or len(text) > 4000:
                    raise ValueError('Inserisci un testo da 1 a 4000 caratteri')
                body = synthesize_bytes(text)
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
            except Exception as e:
                body = json.dumps({'error': str(e)}, ensure_ascii=False).encode('utf-8')
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 1. API Sicurezza Sentinel
        if self.path == "/api/check_safety":
            try:
                data = json.loads(raw_body)
                prompt = data.get("prompt", "")
                sentinel = get_sentinel()
                if sentinel:
                    res = sentinel.evaluate(prompt)
                    resp = {
                        "is_safe": res.is_safe,
                        "action": res.action,
                        "category": res.category,
                        "confidence": res.confidence,
                        "sanitized_prompt": res.sanitized_prompt,
                        "refusal_response": res.refusal_response,
                        "trigger": res.trigger
                    }
                else:
                    resp = {"is_safe": True, "action": "ALLOW", "sanitized_prompt": prompt}
            except Exception as e:
                resp = {"is_safe": True, "action": "ALLOW", "error": str(e)}

            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 1a. API Politica di Rete: Aggiorna Consenso (/api/network_policy)
        if self.path in ("/api/network_policy", "/api/v1/network_policy"):
            try:
                data = json.loads(raw_body)
                pol = get_network_policy()
                if data.get("consent"):
                    pol.grant_consent(scope=data.get("scope", "session"))
                else:
                    pol.revoke_consent()
                from lumina_web_sources import set_enabled
                set_enabled(bool(data.get('consent')))
                resp = {"success": True, "policy": pol.policy}
            except Exception as e:
                resp = {"success": False, "error": str(e)}

            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 1b. API RAG: Parsing Documento (PDF, TXT, MD, Code)
        if self.path == "/api/parse_doc":
            try:
                data = json.loads(raw_body)
                filename = data.get("filename", "documento.txt")
                content_b64 = data.get("content_b64", "")
                text_content = data.get("text", "")

                if content_b64:
                    import base64
                    import io
                    raw_bytes = base64.b64decode(content_b64)
                    if filename.lower().endswith(".pdf"):
                        import pypdf
                        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                        text_content = "\n".join([page.extract_text() or "" for page in reader.pages])
                    else:
                        text_content = raw_bytes.decode("utf-8", errors="replace")

                text_content = re.sub(r"\s+", " ", text_content).strip()
                words = text_content.split()

                chunks = []
                cur = []
                cur_len = 0
                for w in words:
                    cur.append(w)
                    cur_len += len(w) + 1
                    if cur_len >= 250:
                        chunks.append(" ".join(cur))
                        cur = cur[-15:]
                        cur_len = sum(len(x) + 1 for x in cur)
                if cur:
                    chunks.append(" ".join(cur))

                resp = {
                    "status": "ok",
                    "filename": filename,
                    "chunks_count": len(chunks),
                    "total_words": len(words),
                    "preview": text_content[:300] + ("..." if len(text_content) > 300 else ""),
                    "chunks": chunks[:100]
                }
            except Exception as e:
                resp = {"status": "error", "message": str(e)}

            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 1c. API RAG: Query Intelligente su Documento
        if self.path == "/api/rag_chat":
            try:
                data = json.loads(raw_body)
                question = data.get("question", "")
                chunks = data.get("chunks", [])
                filename = data.get("filename", "documento")

                from collections import Counter

                def tok(t):
                    return re.findall(r"\b\w{2,}\b", t.lower())

                q_toks = tok(question)
                q_counter = Counter(q_toks)
                best_chunk = chunks[0] if chunks else ""
                best_score = -1.0

                for c in chunks:
                    c_counter = Counter(tok(c))
                    score = sum(c_counter[term] * q_counter[term] for term in q_counter if term in c_counter)
                    if score > best_score:
                        best_score = score
                        best_chunk = c

                coder_m, coder_tok = get_coder_model()
                import torch
                dev = next(coder_m.parameters()).device
                prompt_messages = [
                    {"role": "system", "content": f"Sei Lumina RAG. Rispondi alla domanda dell'utente in italiano usando come fonte rigorosa l'estratto del documento '{filename}'. Se l'informazione manca, dillo con chiarezza.\n\nESTRATTO:\n{best_chunk}"},
                    {"role": "user", "content": question}
                ]
                prompt_str = coder_tok.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                inputs = coder_tok([prompt_str], return_tensors="pt").to(dev)
                with torch.no_grad():
                    out = coder_m.generate(**inputs, max_new_tokens=220, do_sample=False)
                answer_text = coder_tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

                resp = {
                    "status": "ok",
                    "answer": answer_text,
                    "source_chunk": best_chunk[:250] + ("..." if len(best_chunk) > 250 else ""),
                    "score": best_score,
                    "document": filename
                }
            except Exception as e:
                resp = {"status": "error", "message": str(e)}

            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 2. API Ufficiale OpenAI: Chat Completions (/v1/chat/completions)
        if self.path in ("/v1/chat/completions", "/api/v1/chat/completions"):
            try:
                data = json.loads(raw_body)
                requested_model = data.get("model", "lumina-assisted")
                backend = None
                messages = data.get("messages", [])
                temperature = max(0.0, min(1.5, float(data.get("temperature", 0.0))))
                max_tokens = max(1, min(512, int(data.get("max_tokens", 100))))

                # Estrai l'ultimo messaggio utente
                user_msg = ""
                for m in reversed(messages):
                    if m.get("role") == "user":
                        user_msg = str(m.get("content", "")).strip()
                        break

                if not user_msg:
                    user_msg = "Ciao!"

                # Step 0a: Controllo Consenso Rete Esterna (Lumina Network Policy)
                reply_text = ""
                net_policy = get_network_policy()
                user_msg_lower = user_msg.lower()
                web_requested=bool(data.get('web') or re.search(r'\b(attualità|aggiornamenti|ultime notizie|cerca su internet|cerca sul web|notizie aggiornate)\b',user_msg_lower))
                public_web_requested=bool(re.search(r'^\s*(?:cerca\b|ricerca\b)',user_msg_lower) or re.search(r'https?://',user_msg_lower) or data.get('web_search'))
                research_requested=bool(data.get('research') or re.search(r'\b(ricerca approfondita|fai una ricerca)\b',user_msg_lower))
                weather_requested=bool(re.search(r'\b(meteo|tempo|previsioni|piove|temperatura)\b', user_msg_lower))
                web_requested=web_requested or public_web_requested or research_requested or weather_requested
                from lumina_web_sources import enabled as web_enabled, set_enabled as set_web_enabled

                # Gestione comandi di consenso esplicito via chat/terminale/Visual Studio
                if re.search(r"\b(consento|autorizzo|attiva|abilita)\s+(l'accesso\s+a\s+)?internet\b", user_msg_lower):
                    net_policy.grant_consent(scope="session")
                    set_web_enabled(True)
                    reply_text = (
                        "🛡️ [Lumina Sentinel]\n"
                        "Consenso esplicito accordato: l'accesso a Internet è ora ABILITATO per questa sessione di lavoro."
                    )
                elif re.search(r"\b(revoca|disattiva|blocca|disabilita)\s+(l'accesso\s+a\s+)?internet\b", user_msg_lower):
                    net_policy.revoke_consent()
                    set_web_enabled(False)
                    reply_text = (
                        "🛡️ [Lumina Sentinel]\n"
                        "Consenso revocato: l'accesso a Internet è ora DISABILITATO. Lumina è tornata in modalità 100% offline."
                    )
                else:
                    requires_internet = bool(
                        data.get("tools") or 
                        data.get("allow_internet") or
                        re.search(r"\b(cerca su internet|cerca sul web|scarica da internet|naviga su|fetch url|connettiti a internet)\b", user_msg_lower)
                    )
                    if requires_internet and not net_policy.is_internet_allowed() and not (web_requested and web_enabled()):
                        reply_text = (
                            "🛡️ [Lumina Sentinel — Richiesta di Consenso]\n"
                            "L'accesso a Internet esterno è attualmente DISATTIVATO per impostazione predefinita su Visual Studio e piattaforme collegate.\n\n"
                            "Lumina opera interamente in locale e offline a tutela della tua privacy e sicurezza.\n"
                            "Per autorizzare l'accesso a Internet, scrivi semplicemente: `Autorizzo l'accesso a internet` oppure imposta `set LUMINA_ALLOW_INTERNET=1` nel terminale."
                        )
                    if web_requested and not web_enabled():
                        reply_text='I feed Internet sono disattivati. Scrivi "autorizzo internet" per abilitarli. Modello e voce restano sul computer.'

                # Step 0b: Memoria a Lungo Termine
                mem = get_memory()
                if mem and not reply_text:
                    mem.auto_extract_memories(user_msg)

                # Step A: Controllo Sicurezza con Lumina Sentinel
                sentinel = get_sentinel()
                if sentinel and not reply_text:
                    sec = sentinel.evaluate(user_msg)
                    if not sec.is_safe:
                        reply_text = sec.refusal_response
                    else:
                        user_msg = sec.sanitized_prompt

                # Step B: Generazione con Lumina se sicuro
                if not reply_text:
                    if web_requested:
                        if weather_requested:
                            from lumina_weather import answer
                        elif research_requested:
                            from lumina_research import research as answer
                        elif public_web_requested:
                            from lumina_public_web import answer
                        else:
                            from lumina_web_sources import answer
                        reply_text,backend=answer(user_msg)
                    elif requested_model.lower() in ('lumina-assisted','lumina-general','auto','lumina'):
                        from lumina_general import generate, SYSTEM
                        context_messages=[dict(m) for m in messages if m.get('role') in ('user','assistant')]
                        if context_messages and context_messages[-1]['role']=='user':
                            context_messages[-1]['content']=user_msg
                        else:context_messages.append({'role':'user','content':user_msg})
                        teacher_prompt=None
                        if data.get('purpose')=='teacher_lesson':
                            from qwen_precision import precision_instruction
                            teacher_prompt=SYSTEM+'\n'+precision_instruction()
                        reply_text,backend=generate(context_messages,max_tokens=max_tokens,system=teacher_prompt)
                    elif "coder" in requested_model.lower():
                        try:
                            import torch
                            coder_m, coder_tok = get_coder_model()
                            coder_messages = [
                                {"role": "system", "content": "Sei Lumina Coder, un assistente esperto in programmazione e sviluppo software. Rispondi con codice Python pulito, funzionante, ben formattato e spiegazioni chiare ed essenziali in italiano."},
                            ] + [m for m in messages if m.get('role') in ('user', 'assistant')]
                            coder_messages[-1] = {"role": "user", "content": user_msg}
                            prompt_str = coder_tok.apply_chat_template(coder_messages, tokenize=False, add_generation_prompt=True)
                            dev = next(coder_m.parameters()).device
                            inputs = coder_tok([prompt_str], return_tensors="pt").to(dev)
                            with torch.no_grad():
                                out = coder_m.generate(**inputs, max_new_tokens=max_tokens or 250, do_sample=False)
                            reply_text = coder_tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
                        except Exception as e:
                            reply_text = f"Errore Lumina Coder: {e}"
                    elif "next" in requested_model.lower():
                        try:
                            import torch
                            next_m, next_tok = get_next_model()
                            dev = next(next_m.parameters()).device
                            from lumina_next_runtime import prompt_ids, stop_sequences, response_text
                            context_messages = [dict(m) for m in messages if m.get('role') in ('system', 'user', 'assistant')]
                            if context_messages and context_messages[-1].get('role') == 'user':
                                context_messages[-1] = {"role": "user", "content": user_msg}
                            else:
                                context_messages.append({"role": "user", "content": user_msg})
                            ids = prompt_ids(next_tok, context_messages,
                                             budget=min(2048, next_m.config['max_position_embeddings'] - max_tokens))
                            inputs = torch.tensor([ids], device=dev, dtype=torch.long)
                            with torch.no_grad():
                                out = next_m.generate(inputs, max_new_tokens=max_tokens, temperature=temperature,
                                                      eos_token_id=1, stop_token_sequences=stop_sequences(next_tok))
                            reply_text = response_text(next_tok, out[0][len(ids):].tolist())
                        except Exception as e:
                            reply_text = f"Errore Lumina Next: {e}"
                    else:
                        model, device = get_lumina_model(requested_model)
                        if model:
                            reply_text = model.generate(user_msg, max_new_tokens=max_tokens, temperature=temperature)
                        else:
                            reply_text = "Lumina: Modello non disponibile sul server locale."

                # Costruzione Risposta Specifica Standard OpenAI
                cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                openai_response = {
                    "id": cmpl_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": requested_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": reply_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": max(1, len(user_msg.split())),
                        "completion_tokens": max(1, len(reply_text.split())),
                        "total_tokens": max(2, len(user_msg.split()) + len(reply_text.split()))
                    }
                }
                if backend is not None:
                    openai_response['backend'] = backend
                body = json.dumps(openai_response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            except Exception as e:
                err_resp = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "param": None,
                        "code": 500
                    }
                }
                body = json.dumps(err_resp, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Avvia la web app e server API OpenAI Lumina")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    root = ROOT / "dist"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(root)))
    url = f"http://localhost:{args.port}"
    print("=" * 65)
    print(f" 🌟 LUMINA WEB & OPENAI-COMPATIBLE API SERVER")
    print("=" * 65)
    print(f" [Web App]: {url}")
    print(f" [Monitor]: {url}/monitor.html")
    print(f" [OpenAI API Models]: {url}/v1/models")
    print(f" [OpenAI API Chat]  : {url}/v1/chat/completions")
    print("=" * 65)
    print("Premi Ctrl+C per arrestare il server.\n", flush=True)

    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
