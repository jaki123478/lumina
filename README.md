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

## 📜 Licenza

Rilasciato sotto licenza libera e aperta **MIT**.  
Creato e mantenuto da **Jacopo Forti ([@jaki123478](https://github.com/jaki123478))**.
