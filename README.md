# 🌟 Lumina AI — Intelligenza Artificiale Sovrana & Privata in Lingua Italiana

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jacopof-yellow)](https://huggingface.co/jacopof)
[![Dataset](https://img.shields.io/badge/Dataset-26.4k%20Q%26A-green)](https://huggingface.co/datasets/jacopof/lumina-italian-distilled-qa)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Privacy First](https://img.shields.io/badge/Privacy-100%25%20Offline-success)](https://github.com/jaki123478/lumina)

> **Lumina** è un ecosistema completo di modelli linguistici e strumenti open-source in lingua italiana, progettato per garantire **totale privacy, sovranità dei dati e funzionamento 100% offline** su qualsiasi dispositivo (da laptop consumer a vecchi PC e smartphone), senza abbonamenti e senza inviare informazioni all'esterno.

---

## ⚡ Caratteristiche Principali

* **🛡️ Privacy Assoluta (Zero Cloud)**: Tutti i modelli e i dati restano sul dispositivo locale. Ideale per aziende, professionisti, medici e avvocati nel rispetto del GDPR.
* **📄 RAG Documentale Integrato ("Parla con i tuoi PDF")**: Trascina un file PDF, TXT o codice sorgente per fare domande ed estrarre informazioni con **zero allucinazioni** e citazione automatica dei passaggi.
* **📱 Web App PWA Universale**: Interfaccia reattiva in stile moderno, installabile con un click su Android, iOS e Windows per funzionare anche offline in aereo.
* **💻 Specializzazione nel Coding**: Assistente Python esperto in grado di generare funzioni, spiegare algoritmi, debuggare errori e operare dentro **VS Code** via Copilot/Continue.
* **🦙 Integrazione Ollama a 1-Click**: Pacchetti GGUF pronti all'uso per Ollama, LM Studio e Jan.ai.
* **🛡️ Lumina Sentinel**: Sistema di guardrails integrato a difesa contro attacchi di prompt injection, jailbreak ed esfiltrazione di dati privati.

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
