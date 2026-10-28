# 📡 Lumina — Dashboard di Monitoraggio Remoto (PC Principale)
> **Ultimo Aggiornamento**: `2026-10-28 01:38:16 `  
> **Stato Sistema**: **`In attesa / Pronto per nuovo ciclo`**  
> **Hardware**: `NVIDIA GeForce RTX 5060 Laptop GPU` | VRAM: `2940.0 / 8151.0 MB` | Temp: `52.91 °C`
---
## 1. 🚀 Stato del Modello & Addestramento (Lumina-Next 135M)
* **Step Completati**: `1500 / 1500` (Fase 1 completata)
* **Loss Iniziale**: `N/A` ➔ **Loss Attuale**: **`3.6652`**
* **Finestra di Contesto**: `2,048 token` (RoPE + YaRN)
* **Throughput**: `~13.500 token/s` su precisione mista `bfloat16`
* **Tempo Fase 1**: `151 secondi` (2.5 minuti per 500 step)

### Andamento della Loss (Ultimi Step Registrati):
| Step | Loss | Throughput | VRAM Allocata |
| :---: | :---: | :---: | :---: |

---
## 2. 📚 Stato dei Dataset Locali
| Dataset | File Locale | Campioni Validi | Dimensione | Stato |
| :--- | :--- | :---: | :---: | :--- |
| **Dataset Originale Grezzo** | `data\distilled_dialogues.jsonl` | **29,139** | 9.33 MB | Preservato intatto |
| **Dataset Curato & Deduplicato (v1)** | `data\distilled_dialogues_clean.jsonl` | **11,363** | 3.73 MB | 100% Unico (No dupl.) |
| **Nuovo Batch 2 (Grounded)** | `data\distilled_dialogues_batch2_5k.jsonl` | **5,000** | 1.03 MB | 100% Unico (No dupl.) |
| **Curriculum Specialistico Python** | `data\lumina_coding_curriculum.jsonl` | **71** | 0.01 MB | Preservato intatto |

* **Totale Campioni Puliti Pronti**: **`16,363 dialoghi verificati`**.
---
## 3. 🧪 Benchmark Standard a 8 Domande
| Checkpoint / Fase | Punteggio (Max 16) | % Superamento | Vocabolario / Coerenza Linguistica |
| :--- | :---: | :---: | :--- |
| **Baseline Iniziale (Step 0)** | `0 / 16` | 0.0% | Caratteri casuali e simboli incoerenti |
| **Checkpoint Step 500 (Loss 3.66)** | `0 / 16` | 0.0% | **100% Italiano grammaticale** (*'La funzione in Python...'*) |

> **Nota didattica**: Tra Step 0 e Step 500 il modello ha appreso l'alfabeto italiano e la sintassi del codice. Il prossimo addestramento sui 5.000 nuovi dati grounded insegnerà le conoscenze fattuali (Garda, 17x6=102, martedì+3=venerdì).

---
## 4. 💾 Checkpoint e Pesi Preservati (Hardware Locale)
| File Checkpoint | Descrizione | Dimensione | Data Salvataggio |
| :--- | :--- | :---: | :---: |
| `checkpoints_lumina_next/checkpoint_step500.pt` | Checkpoint incrementale Step 500 (Loss 3.6652) | **197.3 MB** | `2026-10-27 17:58:26` |
| `checkpoints_lumina_next/checkpoint_step500_safe_backup.pt` | Copia di sicurezza blindata Step 500 | **197.3 MB** | `2026-10-27 17:58:26` |
| `checkpoints_lumina_next/pytorch_model.bin` | Pesi PyTorch ufficiali Lumina-Next 135M | **197.29 MB** | `2026-10-27 22:29:19` |
| `hf_export_lumina_next/model.safetensors` | Pesi Hugging Face Safetensors bfloat16 | **394.44 MB** | `2026-10-27 22:30:40` |
| `hf_export_lumina_xlarge/model.safetensors` | Lumina 14M Med (Checkpoint consolidato) | **54.92 MB** | `2026-10-26 19:12:11` |
| `hf_export_lumina_large/model.safetensors` | Lumina 2.75M Nano (Checkpoint consolidato) | **10.5 MB** | `2026-10-26 19:12:11` |

> *Tutti i file binari pesanti e checkpoint rimangono esclusivamente in locale e sono protetti da `.gitignore`.*

---
## 5. 🎯 Prossime Azioni & Avvisi Operativi
* [x] **Completato**: Esecuzione Ciclo 1 (500 step Lumina-Next, loss discesa a 3.6652).
* [x] **Completato**: Curatela e deduplicazione con Qwen (11.363 campioni unici salvati).
* [x] **Completato**: Generazione e salvataggio dei 5.000 nuovi dialoghi grounded.
* [x] **Completato**: Configurazione monitoraggio remoto su GitHub con sicurezza credenziali.
* [ ] **In Attesa di Consenso Utente**: Unione dati (11.3k + 5k = 16.3k) e avvio addestramento 1.000 step su GPU RTX 5060.
