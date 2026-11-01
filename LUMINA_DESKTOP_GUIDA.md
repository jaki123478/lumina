# Lumina Desktop

Applicazione locale per usare Lumina come assistente completo, senza browser.

## Avvio

Apri `AVVIA_LUMINA_EXE.bat` dalla cartella principale. L'avvio usa il server
locale e apre la finestra desktop; se la build grafica non è presente, usa
automaticamente l'app Python locale.

## Funzioni disponibili

- chat con cronologia salvata sul PC;
- modalità Generale, Lumina Next, Coding, Immagini, Genera immagine, Documenti, Ricerca web, Traduttore e Riassunto;
- allegati locali: PDF, DOCX, TXT, Markdown, CSV, JSON, Python, HTML, CSS, JavaScript e immagini;
- voce locale con microfono Vosk e sintesi neurale italiana;
- feedback `Utile` e `Da rivedere`, salvato nel dataset locale;
- pulsante `Rigenera` per ripetere l'ultima richiesta;
- `Ctrl+N` per aprire rapidamente una nuova conversazione;
- modalità Immagini automatica per allegati grafici, con limiti dichiarati dal motore testuale;
- fallback automatico tra API locale e modello direttamente installato.

La modalità `Genera immagine` è opzionale e resta completamente locale: richiede
un modello Diffusers già presente e la variabile `DIFFUSION_MODEL_PATH`. Non
scarica modelli automaticamente.

## Privacy

La modalità offline non invia dati online. La ricerca web è separata e va
selezionata esplicitamente. I documenti vengono letti localmente e il testo
estratto viene passato al modello sul PC.

## Verifica rapida

Usa una nuova chat e prova, nell'ordine:

1. una domanda generale;
2. un file TXT o PDF allegato in modalità Documenti;
3. una richiesta di codice;
4. il pulsante della voce live.

Se un motore non è disponibile, l'app mostra l'errore e tenta il percorso
locale di riserva invece di lasciare la chat bloccata.
