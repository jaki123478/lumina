"""
Generatore del Curriculum di Coding per Lumina AI.
Crea un dataset ad altissima densità informativa (100% Coding in italiano):
Python, algoritmi, strutture dati, OOP, debug ed errori, SQL e Git.
Tutte le risposte sono concise, corrette e perfettamente adatte al budget di Lumina.
"""

import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CODING_SAMPLES = [
    # --- 1. FUNZIONI E SINTASSI PYTHON FONDAMENTALE ---
    {"user": "Come creo una funzione che somma due numeri in Python?", "assistant": "Scrivi: `def somma(a, b): return a + b`. La chiami con `somma(3, 5)` e restituisce 8."},
    {"user": "Come calcolo il quadrato di un numero in Python?", "assistant": "Scrivi: `def quadrato(x): return x ** 2`. Ad esempio `quadrato(4)` restituisce 16."},
    {"user": "Come creo una funzione lambda in Python?", "assistant": "Usa `lambda`: `raddoppia = lambda x: x * 2`. Chiamando `raddoppia(5)` ottieni 10."},
    {"user": "Come restituisco più valori da una funzione in Python?", "assistant": "Separa i valori con la virgola: `def coordinate(): return 10, 20`. Vengono restituiti come tupla `(10, 20)`."},
    {"user": "Come imposto un argomento predefinito in una funzione Python?", "assistant": "Assegna il valore nella firma: `def saluta(nome='Mondo'): return f'Ciao {nome}'`."},
    {"user": "Come calcolo il valore assoluto in Python?", "assistant": "Usa la funzione nativa `abs()`: `abs(-7)` restituisce 7."},
    {"user": "Come calcolo la lunghezza di una stringa o lista in Python?", "assistant": "Usa `len()`: `len('ciao')` restituisce 4, `len([1, 2, 3])` restituisce 3."},
    {"user": "Come converto una stringa in numero intero in Python?", "assistant": "Usa `int()`: `numero = int('42')` converte la stringa '42' nel numero intero 42."},
    {"user": "Come converto un numero in stringa in Python?", "assistant": "Usa `str()`: `testo = str(100)` converte il numero 100 nella stringa '100'."},
    {"user": "Come uso le f-string in Python?", "assistant": "Anteponi `f` alla stringa e metti le variabili tra parentesi graffe: `f'Ciao {nome}, hai {anni} anni'`."},

    # --- 2. LISTE, TUPLE E COMPREHENSION ---
    {"user": "Come filtro i numeri pari da una lista in Python?", "assistant": "Usa una list comprehension: `pari = [x for x in numeri if x % 2 == 0]`."},
    {"user": "Come filtro i numeri dispari da una lista in Python?", "assistant": "Scrivi: `dispari = [x for x in numeri if x % 2 != 0]`."},
    {"user": "Come calcolo il quadrato di ogni numero di una lista in Python?", "assistant": "Usa una list comprehension: `quadrati = [x**2 for x in numeri]`."},
    {"user": "Come aggiungo un elemento alla fine di una lista in Python?", "assistant": "Usa `.append()`: `lista.append('nuovo_elemento')` aggiunge l'elemento in coda alla lista."},
    {"user": "Come rimuovo l'ultimo elemento da una lista in Python?", "assistant": "Usa `.pop()`: `ultimo = lista.pop()` rimuove e restituisce l'ultimo elemento."},
    {"user": "Come ordino una lista in ordine crescente in Python?", "assistant": "Usa `lista.sort()` per ordinare sul posto, oppure `ordinata = sorted(lista)` per una nuova lista."},
    {"user": "Come inverto una lista in Python?", "assistant": "Usa lo slicing: `invertita = lista[::-1]` oppure il metodo sul posto `lista.reverse()`."},
    {"user": "Come trovo il valore massimo e minimo in una lista?", "assistant": "Usa le funzioni native `max(lista)` per il massimo e `min(lista)` per il minimo."},
    {"user": "Come calcolo la somma di tutti gli elementi di una lista?", "assistant": "Usa `sum()`: `totale = sum(numeri)` restituisce la somma degli elementi numerici."},
    {"user": "Come verifico se un elemento è presente in una lista?", "assistant": "Usa l'operatore `in`: `if elemento in lista: print('Trovato')`."},
    {"user": "Come unisco due liste in Python?", "assistant": "Usa l'operatore `+`: `unione = lista1 + lista2` oppure `lista1.extend(lista2)`."},
    {"user": "Cos'è la differenza tra lista e tupla in Python?", "assistant": "Le liste sono mutabili `[1, 2]`, mentre le tuple sono immutabili `(1, 2)` e non modificabili dopo la creazione."},

    # --- 3. DIZIONARI E SET ---
    {"user": "Come creo un dizionario in Python?", "assistant": "Usa le graffe: `dati = {'nome': 'Lumina', 'versione': 2.75}`."},
    {"user": "Come leggo un valore da un dizionario senza causare KeyError?", "assistant": "Usa `.get()`: `valore = dati.get('chiave', 'default')` che restituisce il fallback se la chiave manca."},
    {"user": "Come aggiungo o modifico una coppia chiave-valore in un dizionario?", "assistant": "Scrivi: `dizionario['chiave'] = 'nuovo_valore'` per aggiornare o inserire la chiave."},
    {"user": "Come itero su chiavi e valori di un dizionario?", "assistant": "Usa `.items()`: `for k, v in dati.items(): print(k, v)`."},
    {"user": "Come rimuovo i duplicati da una lista in Python?", "assistant": "Converti la lista in set: `senza_duplicati = list(set(lista))`."},
    {"user": "Cos'è un set in Python?", "assistant": "Un set `s = {1, 2, 3}` è una collezione non ordinata di elementi unici senza duplicati."},

    # --- 4. STRINGHE E MANIPOLAZIONE TESTO ---
    {"user": "Come divido una frase in singole parole in Python?", "assistant": "Usa `.split()`: `parole = frase.split()` divide la stringa sugli spazi."},
    {"user": "Come unisco una lista di parole in un'unica stringa in Python?", "assistant": "Usa `.join()`: `frase = ' '.join(lista_parole)` unisce gli elementi separandoli con spazio."},
    {"user": "Come rimuovo gli spazi bianchi all'inizio e alla fine di una stringa?", "assistant": "Usa `.strip()`: `pulita = testo.strip()` rimuove spazi e ritorni a capo ai bordi."},
    {"user": "Come converto una stringa tutta in minuscolo o maiuscolo?", "assistant": "Usa `.lower()` per il minuscolo `testo.lower()` e `.upper()` per il maiuscolo `testo.upper()`."},
    {"user": "Come sostituisco una parola dentro una stringa in Python?", "assistant": "Usa `.replace()`: `nuovo = testo.replace('vecchio', 'nuovo')`."},
    {"user": "Come verifico se una stringa inizia o finisce con un prefisso?", "assistant": "Usa `.startswith('prefisso')` o `.endswith('suffisso')` che restituiscono True o False."},

    # --- 5. GESTIONE ERRORI E DEBUGGING (ZERO ALLUCINAZIONI) ---
    {"user": "Cosa significa IndexError in Python e come si risolve?", "assistant": "IndexError indica l'accesso a un indice inesistente nella lista. Si risolve verificando `len(lista)` o con `try/except IndexError`."},
    {"user": "Cosa significa KeyError in Python?", "assistant": "KeyError si verifica quando cerchi di accedere a una chiave che non esiste nel dizionario. Risolvi usando `dict.get(chiave)`."},
    {"user": "Cosa significa TypeError in Python?", "assistant": "TypeError indica un'operazione su tipi incompatibili, ad esempio sommare una stringa con un intero `'testo' + 5`."},
    {"user": "Cosa significa ZeroDivisionError in Python?", "assistant": "Indica un tentativo di divisione per zero `x / 0`. Si previene controllando che il divisore sia diverso da zero."},
    {"user": "Come gestisco un'eccezione con try ed except in Python?", "assistant": "Scrivi: `try: ris = 10 / x` e poi `except ZeroDivisionError: ris = 0` per catturare l'errore senza bloccare il programma."},
    {"user": "A cosa serve il blocco finally in Python?", "assistant": "`finally` viene eseguito sempre, sia in caso di successo che di errore, tipicamente per chiudere file o connessioni."},
    {"user": "Come sollevo manualmente un errore in Python?", "assistant": "Usa `raise`: `raise ValueError('Valore non consentito')` per interrompere l'esecuzione con un messaggio chiaro."},

    # --- 6. FILE I/O E DATI ---
    {"user": "Come apro e leggo un file di testo in Python?", "assistant": "Scrivi: `with open('file.txt', 'r', encoding='utf-8') as f: testo = f.read()`. Chiude il file automaticamente."},
    {"user": "Come scrivo del testo in un file in Python?", "assistant": "Usa la modalità `'w'`: `with open('file.txt', 'w', encoding='utf-8') as f: f.write('Ciao')`."},
    {"user": "Come aggiungo testo in coda a un file esistente?", "assistant": "Usa la modalità append `'a'`: `with open('file.txt', 'a', encoding='utf-8') as f: f.write('Nuova riga\\n')`."},
    {"user": "Come leggo e scrivo un file JSON in Python?", "assistant": "Importa `json`: `dati = json.load(f)` per leggere e `json.dump(dati, f, indent=2)` per scrivere."},

    # --- 7. PROGRAMMAZIONE A OGGETTI (OOP) ---
    {"user": "Come definisco una classe con un costruttore in Python?", "assistant": "Scrivi: `class Cane: def __init__(self, nome): self.nome = nome`. Crei un'istanza con `c = Cane('Fido')`."},
    {"user": "Cosa rappresenta self nei metodi di una classe Python?", "assistant": "`self` è il riferimento all'istanza specifica dell'oggetto su cui viene invocato il metodo."},
    {"user": "Come creo una classe derivata che eredita da un'altra classe?", "assistant": "Passa la classe padre tra parentesi: `class Studente(Persona): pass` per ereditare tutti i metodi e attributi."},

    # --- 8. ALGORITMI CLASSICI ---
    {"user": "Cos'è la ricerca binaria in Python?", "assistant": "È un algoritmo efficiente per trovare un elemento in una lista ordinata dimezzando l'intervallo a ogni passo. Complessità: O(log n)."},
    {"user": "Come calcolo il fattoriale di un numero in Python?", "assistant": "In modo ricorsivo: `def fattoriale(n): return 1 if n <= 1 else n * fattoriale(n - 1)`."},
    {"user": "Come genero la sequenza di Fibonacci in Python?", "assistant": "Scrivi: `a, b = 0, 1` e poi in un ciclo `while a < limite: print(a); a, b = b, a + b`."},
    {"user": "Cos'è la complessità computazionale O(1) e O(n)?", "assistant": "O(1) indica tempo costante immediato (es. accesso a dict); O(n) indica tempo proporzionale al numero di elementi (es. scorrere una lista)."},

    # --- 9. SQL E DATABASE ---
    {"user": "Come seleziono tutti i record da una tabella in SQL?", "assistant": "Scrivi: `SELECT * FROM utenti;` per estrarre tutte le colonne e tutte le righe della tabella utenti."},
    {"user": "Come filtro i record con una condizione in SQL?", "assistant": "Usa WHERE: `SELECT nome, email FROM utenti WHERE eta >= 18;`."},
    {"user": "Come ordino i risultati di una query in SQL?", "assistant": "Usa ORDER BY: `SELECT * FROM prodotti ORDER BY prezzo DESC;` per ordinare dal più caro al più economico."},
    {"user": "Come inserisco una nuova riga in una tabella SQL?", "assistant": "Scrivi: `INSERT INTO utenti (nome, email) VALUES ('Mario', 'mario@email.it');`."},
    {"user": "Come aggiorno un record esistente in SQL?", "assistant": "Scrivi: `UPDATE utenti SET stato = 'attivo' WHERE id = 1;`."},

    # --- 10. GIT E CONTROLLO VERSIONE ---
    {"user": "Come creo un nuovo commit in Git?", "assistant": "Aggiungi i file con `git add .` e poi committa con `git commit -m 'Descrizione modifiche'`."},
    {"user": "Come creo e passo a un nuovo branch in Git?", "assistant": "Usa il comando `git checkout -b nome-branch` oppure `git switch -c nome-branch`."},
    {"user": "Come controllo lo stato dei file modificati in Git?", "assistant": "Esegui `git status` per visualizzare i file tracciati, non tracciati e modificati."},
    {"user": "Come scarico le ultime modifiche dal repository remoto in Git?", "assistant": "Esegui `git pull` per scaricare e fondere automaticamente i nuovi commit dal branch remoto."}
]


def build_curriculum():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "lumina_coding_curriculum.jsonl"

    print("=" * 60)
    print("  💻 CREAZIONE CURRICULUM CODING INTENSIVO PER LUMINA AI")
    print("=" * 60)

    # Genera esempi con split train e validation
    records = []
    for i, item in enumerate(CODING_SAMPLES):
        split = "validation" if i % 7 == 0 else "train"
        records.append({"user": item["user"], "assistant": item["assistant"], "category": "coding_curriculum", "split": split})

    # Aggiungi varianti dirette dei comandi (es: "filtro numeri pari", "quadrato python")
    variations = [
        ("Come faccio una somma in Python?", "Scrivi: `def somma(a, b): return a + b`."),
        ("Funzione quadrato Python", "Scrivi: `def quadrato(x): return x ** 2`."),
        ("List comprehension numeri pari", "Usa: `pari = [x for x in numeri if x % 2 == 0]`."),
        ("Come invertire una stringa in Python?", "Usa lo slicing: `s[::-1]`."),
        ("Cos'è un IndexError in Python?", "È un errore che indica l'accesso a un indice fuori dai limiti della lista."),
        ("Come apro un file di testo in Python?", "Scrivi: `with open('file.txt', 'r', encoding='utf-8') as f: testo = f.read()`."),
        ("Come creo un dizionario in Python?", "Usa le graffe: `d = {'chiave': 'valore'}`."),
        ("Come rimuovo duplicati da una lista?", "Converti in set: `list(set(lista))`."),
        ("Come fare una query SELECT in SQL?", "Scrivi: `SELECT * FROM tabella WHERE id = 1;`."),
        ("Comando per creare un branch in Git", "Usa: `git checkout -b nome-branch`.")
    ]
    for i, (u, a) in enumerate(variations):
        split = "validation" if i % 5 == 0 else "train"
        records.append({"user": u, "assistant": a, "category": "coding_curriculum_var", "split": split})

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Creato {out_file} con {len(records)} esempi puri di coding!")
    return out_file


if __name__ == "__main__":
    build_curriculum()
