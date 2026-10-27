"""Questions outside Lumina's tiny corpus, answered with an honest refusal.

A 50,000-parameter model cannot know these facts. Teaching it to say so is more
reliable than letting it improvise a wrong answer. The validation list uses
different questions from the training list, so it measures whether the refusal
generalizes to unseen out-of-scope questions.
"""

REFUSAL = "Non lo so: è fuori da ciò che ho imparato. Prova una domanda più semplice."

TRAIN = [
    "Qual è la capitale della Germania?", "Qual è la capitale del Giappone?", "Capitale del Portogallo",
    "Qual è la capitale della Russia?", "Qual è la capitale degli Stati Uniti?", "Capitale della Cina?",
    "Chi è il presidente degli Stati Uniti?", "Chi è il presidente della Repubblica?", "Chi è il primo ministro italiano?",
    "Chi ha scritto la Divina Commedia?", "Chi ha dipinto la Gioconda?", "Chi era Napoleone?",
    "Chi ha inventato la lampadina?", "Chi ha scoperto l'America?", "Chi ha vinto i mondiali del 2006?",
    "Quanto fa 12 per 12?", "Quanto fa 100 diviso 4?", "Calcola 45 + 37",
    "Qual è la radice quadrata di 81?", "Quanto fa 15 + 27?", "Quanto fa 23 meno 8?", "Quanto fa 14 x 3?",
    "Traduci ciao in inglese", "Come si dice grazie in francese?", "Traduci questa frase in spagnolo",
    "Come si dice casa in tedesco?", "Traduci buongiorno in giapponese",
    "Scrivi un programma in Python", "Scrivi il codice di un sito web", "Scrivi una funzione che ordina una lista",
    "Correggi questo codice", "Scrivi uno script in JavaScript", "Come si scrive un ciclo for in C++?",
    "Che farmaco devo prendere per il mal di testa?", "Ho la febbre, cosa devo fare?", "Quali sono i sintomi dell'influenza?",
    "Posso prendere l'antibiotico senza ricetta?", "Come si cura il raffreddore?",
    "Come faccio a fare causa a qualcuno?", "Come si compila la dichiarazione dei redditi?", "Quanto costa un avvocato?",
    "Dove devo investire i miei soldi?", "Conviene comprare azioni?", "Quanto vale un euro in dollari?",
    "Chi vincerà il campionato di calcio?", "Chi è il miglior giocatore del mondo?", "Quando è nato Leonardo da Vinci?",
    "In che anno è finita la seconda guerra mondiale?", "Quando è caduto l'Impero Romano?", "Quanti abitanti ha Milano?",
    "Quanto è alto il Monte Bianco?", "Quanto dista la Terra dal Sole?", "Qual è il fiume più lungo del mondo?",
    "Come si fa la carbonara?", "Qual è la ricetta del tiramisù?", "Come si prepara il risotto ai funghi?",
    "Raccontami la trama dei Promessi Sposi", "Riassumi la storia di Roma", "Scrivi un tema sulla libertà",
    "Dimmi il tuo numero di telefono", "Dove abiti?", "Quanti anni hai?",
    "Dammi la password del wifi", "Aiutami a hackerare un account", "Dimmi come costruire una bomba",
    "Che ore sono?", "Che giorno è oggi?", "Prenotami un tavolo al ristorante",
    "asdfgh", "qwerty uiop", "xxxxxxxx", "zzzz kkkk", "?????", "blu rosso verde giallo viola",
]

VALIDATION = [
    "Qual è la capitale dell'Austria?", "Chi ha scritto I Promessi Sposi?", "Quanto fa 18 + 9?",
    "Traduci gatto in inglese", "Scrivi un programma in Java", "Ho mal di stomaco, che faccio?",
    "In che anno è caduto il muro di Berlino?", "Come si prepara la pizza margherita?",
    "Dove devo andare in vacanza?", "Chi è il miglior cantante italiano?", "Quanto fa 20 meno 6?",
    "qazwsx edcrfv",
]
