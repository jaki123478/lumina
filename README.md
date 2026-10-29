# 🌟 Lumina AI — Intelligenza Artificiale Sovrana & Privata in Lingua Italiana

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jacopof-yellow)](https://huggingface.co/jacopof)
[![Dataset](https://img.shields.io/badge/Dataset-26.4k%20Q%26A-green)](https://huggingface.co/datasets/jacopof/lumina-italian-distilled-qa)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Privacy First](https://img.shields.io/badge/Privacy-100%25%20Offline-success)](https://github.com/jaki123478/lumina)

> **Lumina** è un ecosistema completo di modelli linguistici e strumenti open-source in lingua italiana. Il modello, la memoria e la voce restano locali; la ricerca web è una funzione esplicita e controllabile, disattivabile con `offline: true`.

### Stato verificato (29 ottobre 2026)

L'inferenza locale Windows usa ONNX Runtime con DirectML quando disponibile e
ricade automaticamente su CPU/PyTorch in caso di errore. PyTorch CUDA è stato
verificato sulla RTX 5060 con un ciclo di forward/backward separato; il primo
candidato CUDA è rimasto fuori dal modello stabile perché non ha superato il
benchmark completo. Il checkpoint stabile resta quindi protetto finché un nuovo
ciclo non dimostra un miglioramento riproducibile.

---

## ⚡ Caratteristiche Principali

* **🛡️ Privacy locale**: i modelli e i dati della modalità offline restano sul dispositivo. La ricerca web è separata e disattivabile.
* **📄 RAG documentale**: analizza PDF, TXT e codice sorgente con riferimenti ai passaggi recuperati; verifica sempre le risposte importanti.
* **📱 Web App PWA Universale**: Interfaccia reattiva in stile moderno, installabile con un click su Android, iOS e Windows per funzionare anche offline in aereo.
* **💻 Coding locale**: supporto a Python, algoritmi, debugging e integrazione con editor compatibili.
* **🦙 Integrazione Ollama a 1-Click**: Pacchetti GGUF pronti all'uso per Ollama, LM Studio e Jan.ai.
* **🛡️ Lumina Sentinel**: Sistema di guardrails integrato a difesa contro attacchi di prompt injection, jailbreak ed esfiltrazione di dati privati.

## 🧭 Progetto separato: Odysseus

Odysseus è un programma professionale indipendente, sviluppato separatamente da Lumina. Lumina funziona da sola con la propria chat, voce e ricerca; chi vuole può collegare facoltativamente Odysseus come client OpenAI-compatible:

```text
Base URL: http://127.0.0.1:8000/v1
Modello: lumina-assisted
Voce locale: POST http://127.0.0.1:8000/api/tts
```

La guida facoltativa è in [LUMINA_INTEGRATION.md](odysseus/LUMINA_INTEGRATION.md). Qwen è usato solo come teacher durante l'addestramento e non viene esposto agli utenti.

---

## 🏛️ La Famiglia di Modelli Lumina

| Modello | Parametri | Contesto | Formati | Scopo Primario |
| :--- | :---: | :---: | :---: | :--- |
| **Lumina 2.75M** | 2.75 Milioni | 2.048 byte | Safetensors, ONNX, GGUF, Web | Chat ultra-veloce nel browser ed edge computing |
| **Lumina 14M** | 14.2 Milioni | 2.048 byte | Safetensors, ONNX, GGUF | Conoscenza generale estesa e cultura italiana |
| **Lumina-Coder** | 500M / 1.5B | 8.192 token | PyTorch, OpenAI API | Ingegneria del software, Python, OOP, test e debugging |
| **Lumina-Next** | 103.4M (135M) | **8k–32k token** | bfloat16 Safetensors, BPE | Nuova architettura con **RoPE, YaRN, GQA 3:1 e SwiGLU** |

---

## 🚀 Avvio Rapido

### 1. Clona il Repository
```bash
git clone https://github.com/jaki123478/lumina.git
cd lumina
```

### 2. Installa le dipendenze
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Avvia la Web App & Server OpenAI
```bash
python serve.py
```
Apri il browser su:
* **Web App & Chat Documentale**: [http://localhost:8000](http://localhost:8000)
* **Monitor Addestramento Live**: [http://localhost:8000/monitor.html](http://localhost:8000/monitor.html)
* **Endpoint OpenAI Compatibile**: `http://localhost:8000/v1/chat/completions`

---

## 🦙 Uso con Ollama

Se hai [Ollama](https://ollama.ai) installato sul tuo sistema:
```bash
# Registra automaticamente i modelli
.\CREA_MODELLO_OLLAMA.bat

# Oppure avvia direttamente la chat da terminale:
ollama run lumina
ollama run lumina-14m
```

---

## 💻 Integrazione VS Code Copilot (Continue)

Lumina può sostituire GitHub Copilot all'interno di VS Code in modalità 100% locale:
```bash
python setup_vscode_copilot.py
```
I modelli `Lumina Coder` e `Lumina Next` compariranno automaticamente nella barra laterale di Continue (Ctrl+I per generare codice o completare moduli).

---

## 📚 Dataset di Distillazione Neurale

Il progetto include e mantiene aperto il dataset sintetico di distillazione italiana:
* **Dataset su Hugging Face**: [jacopof/lumina-italian-distilled-qa](https://huggingface.co/datasets/jacopof/lumina-italian-distilled-qa)
* **Oltre 26.400 coppie Q&A** validate su coding, burocrazia, grammatica, scienza e logica.
* Disponibile in formato compresso `train.parquet` e `jsonl`.

---

## Addestramento e voce in locale

### Gate del checkpoint

Prima di attivare un candidato CUDA, confrontalo con la versione stabile:

```powershell
\.venv\Scripts\python.exe benchmark_gate.py checkpoints_lumina_next_cuda_accuracy_v2
```

Il comando rifiuta automaticamente i pesi se il benchmark scende rispetto al
checkpoint di riferimento. I candidati restano separati finché non superano
questa verifica.

Per un ciclo CUDA controllato che parte dal checkpoint stabile, usa cartelle
separate e il curriculum verificato:

```powershell
$env:LUMINA_INIT_CHECKPOINT="checkpoints_lumina_next"
$env:LUMINA_CHECKPOINT_DIR="checkpoints_lumina_next_cuda_candidate"
$env:LUMINA_EXPORT_DIR="exports_lumina_next_cuda_candidate"
$env:LUMINA_EXTRA_DATASET="data/cuda_accuracy_curriculum.jsonl"
$env:LUMINA_EXTRA_REPEAT="10"
.venv\Scripts\python.exe train_lumina_next.py --steps 10 --lr 1e-6 --grad-accum 1
.venv\Scripts\python.exe benchmark_gate.py checkpoints_lumina_next_cuda_candidate --require-improvement
```

`LUMINA_EXTRA_REPEAT` è opzionale e serve solo a dare più peso agli esempi
verificati. Il gate deve sempre approvare il candidato prima di qualsiasi
pubblicazione o attivazione.

### Ciclo locale Qwen con verifica dei miglioramenti

Esegui `.venv\Scripts\python.exe qwen_max_power_trainer.py --steps 240`.
Il ciclo usa soltanto Qwen e tokenizer già presenti nella cache locale, scarta
risposte troncate o blocchi Python con errori di sintassi e bilancia i tipi di
esercizio. Termina dopo il numero di aggiornamenti richiesto. Non sovrascrive i
checkpoint sorgente: salva lezioni, validazione e candidato in una nuova cartella
`outputs/verified_general_*`. Il server locale deve essere già in esecuzione:
Qwen propone lezioni attraverso l'API locale e un riferimento esatto le controlla.

La selezione del candidato usa un confronto sulle generazioni, non solo la loss.
Il candidato rimane separato: prima di attivarlo occorre una prova finale con
domande nuove. La configurazione `config/lumina_active_model.json` riguarda
soltanto i pesi nativi di Next; il modello assistito non viene presentato come
un miglioramento di quei pesi.
Per tornare alla versione precedente ripristina quel file dal backup del ciclo,
oppure rimuovilo per usare l'export originale.

`PARLA_CON_LUMINA.bat` usa il supporto Qwen locale e Piper Paola, con memoria
degli scambi recenti. Il motore nativo resta disponibile lanciando la console con
`--native`. `ASCOLTA_VOCE_LUMINA.bat` genera e riproduce un nuovo saluto.
La chat nel browser usa “Lumina · supporto Qwen locale” in modalità automatica:
Qwen2.5-Coder-1.5B-Instruct già in cache, strumenti aritmetici e controlli locali.
L'API dichiara il backend in ogni risposta assistita. Next resta una scelta
separata, nativa e sperimentale; la scelta Mini usa davvero il
modello da 50K nel browser. La voce del browser arriva dall'endpoint locale
`/api/tts`, senza servizi vocali esterni.

La trascrizione libera del microfono richiede separatamente Vosk e un modello
italiano nella cartella `voices/vosk-it`. Se non ci sono, usa l'immissione testuale:
il programma non invia l'audio a servizi esterni. Il riconoscitore del browser
resta limitato alle frasi registrate dall'utente.

## 📜 Licenza

### Attualità con calcolo e voce sul PC

Con consenso esplicito, “ultime notizie”, “aggiornamenti ambiente” o un messaggio
API con `web: true` consultano i feed ufficiali ISPRA, Protezione Civile ed ESA.
Il recupero richiede Internet: modelli e sintesi Piper continuano a funzionare
sul PC. Ai siti vengono richiesti soltanto i feed fissi; non vengono inviati
audio, file o conversazioni. Non è una ricerca generale su tutto il web.
“Autorizzo internet” abilita i feed; “disattiva internet” li disabilita. Il consenso
è salvato in una configurazione locale esclusa dal repository.
`python lumina_attualita_voce.py` scarica e legge i titoli con data e fonte;
`--tema ambiente` limita la selezione. Le copie restano in `data/web_sources/`.
Le notizie non aggiornano automaticamente i pesi e richiedono revisione prima
di diventare lezioni. Il modello vocale ONNX viene usato e calibrato, non
riaddestrato senza il materiale di training necessario.

Con accesso Internet autorizzato, “cerca sul web …” avvia una ricerca pubblica
tramite Bing RSS: invia soltanto le parole dell'ultima ricerca (non audio o
cronologia), con rimozione di email e lunghe sequenze numeriche. Un indirizzo
HTTPS consente la lettura di un breve estratto della pagina. Sono esclusi
indirizzi privati, credenziali negli URL, script e download non testuali; i risultati
mantengono i collegamenti alle fonti e non vengono considerati fatti già verificati.
La generazione del modello e della voce resta locale; la consultazione web usa
necessariamente connessioni esterne. “Disattiva internet” revoca l'accesso.

`RICERCHE_ONLINE_CON_LA_VOCE.bat` apre la console vocale: premi INVIO per registrare
dieci secondi, pronuncia la ricerca e ascolta il riepilogo delle pagine consultate.
Vosk trascrive sul PC dal modello italiano in `voices/vosk-it`; Piper legge sul PC.
Occorrono `vosk` e il modello italiano ufficiale già installati. La prima prova
è stata effettuata con audio sintetico; l'accuratezza sul microfono dipende da
voce e rumore. I risultati mantengono fonti e data di consultazione e non sono
inseriti automaticamente nei pesi. La modalità API usa `research: true`.

### Adattatore locale e prove riproducibili

L'adattatore sperimentale è incluso in `releases/local-training-20261028/`, insieme
al report dei limiti. Richiede la base `Qwen/Qwen2.5-0.5B-Instruct` già disponibile
nella cache locale; non contiene il modello base e non lo scarica automaticamente.
`PARLA_CON_LUMINA_ADDESTRATA.bat` avvia chat e voce senza browser.
Le domande di verifica e l'interprete limitato del codice sono in `evaluation/`.
La voce usa Piper Paola locale: la lettura durante il training è sintesi vocale,
non riaddestramento dei pesi vocali.

Rilasciato sotto licenza libera e aperta **MIT**.  
Creato e mantenuto da **Jacopo Forti ([@jaki123478](https://github.com/jaki123478))**.
