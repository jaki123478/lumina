# Corpus dimostrativo di Lumina

Dialoghi brevi in italiano, scritti appositamente per questa applicazione.
Nessun peso o dataset preaddestrato viene scaricato. `build_dataset.py`
rigenera `dialogues.jsonl`; ogni riga contiene `user`, `assistant`, `split`.

Per ogni argomento, quattro formulazioni, con varianti di maiuscole e punteggiatura, sono usate per l'addestramento e una
per la validazione. Gli argomenti e le risposte coincidono tra i due gruppi:
questa validazione misura solo la generalizzazione della formulazione.
Non dimostra conoscenze generali, correttezza, sicurezza o capacità di ragionare.

Questo corpus piccolo favorisce la memorizzazione e non contiene una copertura
generale del linguaggio. Per proseguire l'addestramento si possono aggiungere
nuove righe originali, mantenendo brevi domande e risposte.
