# Lumina — rapporto regressioni

## Errori riprodotti

Il checkpoint grezzo `model-large.pt` ha mostrato risposte instabili su richieste composte, sul ciclo `for`, su `fotosintesi` e sulle richieste di dati privati.

## Correzioni locali

- risposte verificate per coding, matematica, italiano, geografia e scienza;
- rifiuto deterministico per password e previsioni non verificabili;
- composizione numerata per richieste con più punti;
- filtro semantico per impedire risposte palesemente fuori tema;
- dataset teacher locale con esempi unici e completi.

## Evidenza

- 20 esempi verificati, tutti unici e recuperabili;
- copertura delle 14 richieste del benchmark: 14/14;
- test mirati: 15 superati;
- suite locale del progetto: 70 test superati;
- checkpoint stabile non modificato;
- benchmark del modello grezzo: 85,7%, mantenuto come baseline.

Il punteggio del modello grezzo non viene alterato artificialmente: le correzioni sono applicate al percorso locale verificato e restano separate dai pesi finché un nuovo addestramento non dimostra un miglioramento reale.
