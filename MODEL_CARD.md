# Lumina 50K

## Stato del progetto

Il modello 50K è un modello sperimentale leggero per esecuzione locale. Le
funzioni voce, ricerca e DirectML appartengono all'applicazione e non aumentano
il numero di parametri del checkpoint. I candidati prodotti dall'addestramento
GPU vengono valutati separatamente e non sostituiscono il checkpoint stabile se
introducono regressioni su aritmetica, coding, grammatica o onestà epistemica.

## Aggiornamento tecnico

L'app desktop può usare DirectML per l'inferenza GPU su Windows, con fallback
automatico a CPU/PyTorch. CUDA è stato verificato separatamente sulla RTX 5060
con forward e backward reali. Il training GPU viene eseguito in cartelle di
prova e accettato solo dopo il benchmark; il candidato breve che ha perso
accuratezza aritmetica non è stato pubblicato.

Modello sperimentale di generazione del testo, costruito e addestrato da zero.
Non importa pesi preaddestrati, non chiama servizi di generazione e non usa Ollama.
È un Transformer minuscolo, non un modello con capacità paragonabili ai chatbot
commerciali. Le risposte sono prodotte autoregressivamente dai pesi, senza una
tabella di risposte o un instradamento per parole chiave.

## Architettura

Decoder causale pre-LayerNorm, due blocchi, quattro teste di attenzione,
dimensione interna 40, feed-forward 120, GELU con approssimazione tanh.
Il tokenizer usa 256 valori di byte UTF-8 e tre token speciali (inizio, fine,
riempimento). Il contesto massimo è di 165 token, corrispondenti a byte e token
speciali, non a 165 parole. L'uscita condivide i pesi dell'embedding dei token.

| Componente | Parametri |
| --- | ---: |
| Embedding dei 259 token × 40 | 10.360 |
| Embedding delle 165 posizioni × 40 | 6.600 |
| Primo blocco Transformer | 16.480 |
| Secondo blocco Transformer | 16.480 |
| LayerNorm finale | 80 |
| **Totale** | **50.000** |

Ogni blocco comprende 6.560 parametri di attenzione, 9.760 di feed-forward e
160 nelle due LayerNorm. Non esistono tensori aggiunti solo per raggiungere il
numero richiesto. I 50.000 valori float32 occupano 200.000 byte prima di metadati
e codifica JSON. Il modello Python e quello JavaScript applicano le stesse
operazioni; l'esportazione non contiene codice eseguibile.

## Dati e addestramento

Il corpus originale contiene 910 esempi su 67 argomenti: 843 formulazioni di
addestramento e 67 formulazioni di validazione. Comprende saluti, descrizioni del
modello, piccole nozioni di programmazione e brevi risposte didattiche.
Le varianti di maiuscole e punteggiatura sono esempi di addestramento, non regole
usate per scegliere le risposte. Le domande di validazione non coincidono con
quelle di addestramento neanche normalizzando maiuscole e punteggiatura.
Gli argomenti e le risposte desiderate invece coincidono: la validazione misura
soltanto la capacità di riconoscere formulazioni simili, non conoscenza generale.

L'inizializzazione è casuale; il seed predefinito è 42. Si usa AdamW e cross
entropy sul testo delle risposte. La loss è calcolata in nat per token di
risposta. Il codice non utilizza Internet durante l'addestramento.
I numeri effettivi dell'ultima esecuzione, il numero di passi e l'hash SHA-256
del corpus si trovano in `training` dentro `dist/model/weights.json`.

## Riprodurre o continuare

```powershell
python data/build_dataset.py
python train.py --steps 8000 --threads 4
python tests/test_model.py -v
python model.py "Come funzioni?"
```

`train.py` crea `model.pt` e `dist/model/weights.json`. Per continuare dai pesi
esistenti: `python train.py --resume --steps 2000`. `--resume` riparte dai pesi,
crea un nuovo ottimizzatore e aggiunge i passi ai metadati. Per usare un corpus
diverso: `--data percorso/dialoghi.jsonl`, mantenendo righe JSON con `user`,
`assistant` e `split` (`train` oppure `validation`). Le risposte devono essere
brevi per entrare nel contesto. Dopo una nuova esportazione ricaricare l'app.

## Limiti

Il modello tende a memorizzare i pochi dialoghi. Domande diverse dagli esempi
possono produrre risposte sbagliate, ripetizioni, frasi senza senso e byte UTF-8
incompleti. Non è un calcolatore generale, non cerca sul web, non conosce le
notizie e non è adatto a decisioni importanti. Una cronologia mostrata
dall'interfaccia non equivale a una memoria appresa dal modello. Il formato di
addestramento è un singolo turno: `BOS + U: domanda + \nL: risposta + EOS`.

Il numero 50.000 descrive esclusivamente il modello di testo. Il riconoscimento
e la sintesi vocale dell'app sono componenti separati: non sono inclusi in quel
conteggio e non trasformano il modello in un sistema multimodale generale.
