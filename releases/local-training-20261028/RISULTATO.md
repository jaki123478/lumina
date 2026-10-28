# Lumina addestrata e voce locale — 28 ottobre 2026

Sono stati eseguiti tre cicli reali da 320 aggiornamenti ciascuno: **960 aggiornamenti complessivi in tre esperimenti separati**. Ogni esperimento parte dalla base Qwen2.5-0.5B-Instruct già presente nella cache del computer. Non è un nuovo addestramento da zero del vecchio Next da 103M parametri.

L'adattatore finale aggiorna **540.672 parametri**, con LoRA di rango 8 e fattore alpha 16 sulle proiezioni query e value. La base resta congelata. L'adattatore salvato è di circa 2,2 MB e richiede il modello base locale per funzionare. Non sono stati installati pacchetti né scaricati modelli: nei processi di addestramento erano bloccati socket e DNS, e non sono stati rilevati tentativi di rete.

Il primo tentativo si è fermato prima del training per un'incompatibilità del tokenizer, poi corretta. È conservato nel relativo log. I tre cicli conteggiati sopra sono quelli completati.

## Selezione e verifica

Il primo esperimento migliorava alcune risposte ma introduceva regressioni. Il secondo raggiungeva 21/30 nei controlli, ma perdeva una risposta sui limiti del futuro e non è stato selezionato. Il terzo contiene più esempi controllati di grammatica e limiti di conoscenza e richiede di conservare le risposte già corrette in quelle aree di sviluppo.

L'ultimo ciclo passa da **14/30 a 20/30** sui controlli automatici di sviluppo. Il punteggio automatico sulle dieci domande finali passa da **5/10 a 8/10**. Una revisione completa distingue però:

- **6 risposte complete corrette:** somma, copia di MAGNOLIA, nome Maristella, numero di nocciole, memoria del cane Quarzo, numero di lati del quadrato.
- **2 risposte parziali:** riconosce il limite sul numero nell'armadio ma dà una motivazione poco pertinente; corregge l'ausiliare in “Io sono andato a biblioteca”, lasciando una preposizione sbagliata.
- **2 risposte errate:** ripete “porto” una sola volta anziché due; indica Vantaa invece di Helsinki come capitale della Finlandia.

Quindi il risultato finale pienamente corretto è **6/10**, non 8/10. Il campione è piccolo e gli esperimenti sono stati corretti usando gli errori dei cicli precedenti. Non è una prova generale di accuratezza. La versione resta **sperimentale**, separata dalla modalità assistita e dal vecchio Next; non sostituisce automaticamente il motore predefinito della chat.

I dati della prova finale sono esclusi dal curriculum per corrispondenza esatta della conversazione. Il materiale di training è composto da riferimenti controllati; non sono stati riutilizzati i dialoghi troncati del vecchio addestratore.

## Uso diretto in locale

`PARLA_CON_LUMINA_ADDESTRATA.bat` avvia una console con il modello base locale, l'adattatore selezionato e Piper. Questo percorso non apre pagine e non richiede un server HTTP. La selezione è registrata in `config/lumina_trained_adapter.json`; `chat_lumina_addestrata.py` mostra esplicitamente la natura sperimentale e l'origine della base.

La prova reale “Come ti chiami?” ha prodotto **“Lumina”**. La risposta è stata sintetizzata in `audio_neurale/prova_locale_20261028/saluto_modello_addestrato.wav`.

## Voce e lettura continua

La voce usa **Piper Paola già presente sul PC**. Sono stati regolati ritmo, volume e parametri di sintesi; **i pesi vocali non sono stati riaddestrati**, perché sono disponibili i file di inferenza ONNX, non i sorgenti di training del modello vocale. Non viene presentata una semplice regolazione come addestramento della rete vocale.

Sono stati generati tre campioni a ritmi diversi. Il profilo equilibrato usa length scale 1,08, volume 0,85, noise scale 0,55 e noise width 0,7. Il campione esteso dura circa **9,33 secondi**, raggiunge un picco di **0,850** e ha **zero campioni saturati**. Le impostazioni sono salvate in `config/lumina_voice_settings.json`. La naturalezza non è stata valutata da ascoltatori umani.

Su richiesta dell'utente è stata attivata la lettura continua dagli altoparlanti durante l'addestramento, senza aprire browser. `lumina_training_voice.py` ha letto esercizi di riferimento e avanzamenti, una voce alla volta, e ha annunciato la conclusione di ciascun ciclo. Gli esercizi sono stati identificati come riferimenti corretti, non come prove di ciò che il modello aveva già imparato. I log confermano “VOCE TERMINATA” dopo la fine dei cicli; non è rimasto un processo a parlare indefinitamente a training concluso.

Il file `audio_neurale/prova_locale_20261028/equilibrata.wav` permette di riascoltare il campione esteso. La risposta generata dal modello e il campione sono stati riprodotti localmente.

## Evidenze

- `report.json`: risultati grezzi dell'ultimo ciclo; contiene i punteggi automatici, da leggere con la revisione sopra.
- `adapter.pt`, `last_adapter.pt`, `adapter_config.json`: adattatore selezionato, ultimo adattatore e configurazione.
- `train.jsonl`, `development.json`, `final_questions.json`: riferimenti di training e domande di controllo.
- `artifacts/lumina_local_adapter_training_v4.log`: avanzamenti del training.
- `artifacts/lumina_training_voice_v4.log`: frasi effettivamente pronunciate durante il ciclo.
- `artifacts/lumina_adapter_tests.log`: **23 test Python superati**, inclusi congelamento dei pesi base e salvataggio/caricamento dell'adattatore. Superati anche i controlli JavaScript della voce.

I limiti rimasti richiedono ulteriori dati e valutazioni; questa versione non viene dichiarata pronta per risposte sempre precise.
