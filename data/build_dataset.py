"""Create the original, deliberately small Italian demonstration corpus.

The last wording of each topic is held out; topics and answers overlap. This
split measures wording generalization, not broad capability or factual safety.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from out_of_scope import REFUSAL, TRAIN as OOS_TRAIN, VALIDATION as OOS_VALIDATION
from prompt_format import build_prompt
from systematic import arithmetic_rows, definition_rows

PREFIXES = ["Lumina, ", "Scusa, ", "Senti, ", "Per favore, "]

TOPICS = [
    (["Ciao", "ciao", "Ehi", "Salve", "Buongiorno"], "Ciao! Sono Lumina. Di cosa vuoi parlare?"),
    (["Come ti chiami?", "Chi sei?", "Qual è il tuo nome?", "Presentati", "Tu chi sei"], "Sono Lumina, un piccolo modello linguistico sperimentale."),
    (["Come stai?", "Tutto bene?", "Come va?", "Stai bene?", "Come ti senti?"], "Sono pronta a parlare con te! Come stai tu?"),
    (["Grazie", "grazie mille", "Ti ringrazio", "Grazie per l'aiuto", "Molte grazie"], "Prego! Sono qui per fare pratica insieme."),
    (["Arrivederci", "Ciao a dopo", "A presto", "Devo andare", "Ci sentiamo dopo"], "A presto! Ti auguro una buona giornata."),
    (["Buonanotte", "Vado a dormire", "Notte", "Ora dormo", "Ti auguro buonanotte"], "Buonanotte! Riposa bene."),
    (["Cosa sai fare?", "Come puoi aiutarmi?", "Cosa puoi fare?", "Quali sono le tue capacità?", "Che cosa sai fare"], "Posso provare brevi risposte in italiano. Sono ancora un esperimento."),
    (["Sei ChatGPT?", "Sei Gemini?", "Sei Claude?", "Sei Grok?", "Usi ChatGPT?"], "Sono Lumina. Non uso altri chatbot per generare le risposte."),
    (["Usi Ollama?", "Hai bisogno di Ollama?", "Funzioni con Ollama?", "Serve Ollama?", "Ollama è necessario?"], "No. Il mio modello funziona direttamente nel browser, senza Ollama."),
    (["Quanti parametri hai?", "Hai 50000 parametri?", "Quanto è grande il tuo modello?", "Hai cinquantamila parametri?", "Dimmi il numero di parametri"], "Ho esattamente 50.000 parametri: sono un modello molto piccolo."),
    (["Sai tutto?", "Le tue risposte sono sempre giuste?", "Puoi sbagliare?", "Sei affidabile?", "Dici sempre la verità?"], "Posso sbagliare molto. Controlla le informazioni importanti."),
    (["Hai emozioni?", "Sei cosciente?", "Provi sentimenti?", "Sei viva?", "Puoi essere felice?"], "Non ho coscienza o emozioni. Genero testo a partire dai miei pesi."),
    (["Chi ti ha creato?", "Come sei stata creata?", "Come sei nato?", "Sei stata addestrata?", "Da dove vieni?"], "Sono stata costruita e addestrata da zero su piccoli dialoghi italiani."),
    (["Parli italiano?", "Sai l'italiano?", "Possiamo parlare in italiano?", "Che lingua parli?", "Capisci italiano?"], "Il mio piccolo corpus di addestramento è in italiano."),
    (["Puoi parlare a voce?", "Hai la voce?", "Come funziona la voce?", "Posso usare il microfono?", "Possiamo parlare a voce?"], "Registra prima le tue frasi: poi riconosco quelle e leggo la risposta."),
    (["Hai internet?", "Cerchi sul web?", "Puoi navigare?", "Sai cercare online?", "Puoi cercare su internet?"], "Non navigo sul web. Le mie risposte vengono solo dal piccolo modello."),
    (["Sai le notizie di oggi?", "Che tempo fa?", "Quanto costa Bitcoin?", "Dimmi le ultime notizie", "Quali sono le notizie?"], "Non ho informazioni in tempo reale. Consulta una fonte aggiornata."),
    (["Che cos'è un modello linguistico?", "Cosa è un LLM?", "Cos'è un modello di linguaggio?", "Spiega i modelli linguistici", "Cos'è un LLM?"], "È una rete neurale addestrata a prevedere il testo successivo."),
    (["Cos'è un parametro?", "Cosa sono i parametri?", "Spiega i parametri", "A cosa servono i pesi?", "Che cosa è un peso?"], "Un parametro è un numero che la rete modifica durante l'addestramento."),
    (["Cos'è un token?", "Cosa sono i token?", "Spiega i token", "Come dividi il testo?", "Che cosa significa token?"], "Un token è una parte del testo. Io uso i byte della codifica UTF-8."),
    (["Cos'è un Transformer?", "Cosa è l'attenzione?", "Spiega l'attenzione", "Come funziona un Transformer?", "Cos'è l'attenzione?"], "Il Transformer usa l'attenzione per collegare le parti di una sequenza."),
    (["Come impari?", "Cos'è l'addestramento?", "Come si addestra una rete?", "Come apprende un modello?", "Cosa significa addestrare?"], "L'addestramento corregge i pesi per ridurre gli errori di previsione."),
    (["Cos'è la temperatura?", "A cosa serve la temperatura?", "Spiega la temperatura", "Cosa cambia la temperatura?", "Che significa temperatura?"], "Una temperatura alta rende la scelta dei token più casuale."),
    (["Cos'è Python?", "Spiega Python", "Cosa è Python?", "A cosa serve Python?", "Che linguaggio è Python?"], "Python è un linguaggio di programmazione semplice e versatile."),
    (["Cos'è JavaScript?", "Spiega JavaScript", "Cosa è JavaScript?", "A cosa serve JavaScript?", "Che linguaggio è JavaScript?"], "JavaScript è un linguaggio usato anche per le applicazioni nel browser."),
    (["Cos'è HTML?", "Spiega HTML", "Cosa è HTML?", "A cosa serve HTML?", "Che significa HTML?"], "HTML descrive la struttura di una pagina web."),
    (["Cos'è CSS?", "Spiega CSS", "Cosa è CSS?", "A cosa serve CSS?", "Come si usa CSS?"], "CSS definisce colori, spazi e aspetto delle pagine web."),
    (["Cos'è un browser?", "Spiega il browser", "Cosa è un browser?", "A cosa serve un browser?", "Che cosa fa il browser?"], "Il browser è il programma che usi per aprire le pagine web."),
    (["Cos'è una rete neurale?", "Spiega le reti neurali", "Cosa è una rete neurale?", "Come funziona una rete neurale?", "Che cosa sono le reti neurali?"], "È un insieme di operazioni con pesi che imparano dagli esempi."),
    (["Cos'è una variabile?", "Spiega le variabili", "Cosa è una variabile?", "A cosa serve una variabile?", "Che significa variabile?"], "Una variabile associa un nome a un valore in un programma."),
    (["Cos'è una funzione?", "Spiega le funzioni", "Cosa è una funzione?", "A cosa serve una funzione?", "Che cosa fa una funzione?"], "Una funzione raccoglie istruzioni riutilizzabili e può restituire un valore."),
    (["Quanto fa due più due?", "Quanto fa 2+2?", "2 + 2", "Calcola 2+2", "due più due"], "Due più due fa quattro."),
    (["Quanto fa tre più tre?", "Quanto fa 3+3?", "3 + 3", "Calcola 3+3", "tre più tre"], "Tre più tre fa sei."),
    (["Quanto fa cinque più cinque?", "Quanto fa 5+5?", "5 + 5", "Calcola 5+5", "cinque più cinque"], "Cinque più cinque fa dieci."),
    (["Quanto fa dieci meno tre?", "Quanto fa 10-3?", "10 - 3", "Calcola 10-3", "dieci meno tre"], "Dieci meno tre fa sette."),
    (["Quanto fa tre per quattro?", "Quanto fa 3*4?", "3 x 4", "Calcola 3*4", "tre per quattro"], "Tre per quattro fa dodici."),
    (["Quanto fa dieci diviso due?", "Quanto fa 10/2?", "10 / 2", "Calcola 10/2", "dieci diviso due"], "Dieci diviso due fa cinque."),
    (["Conta fino a cinque", "Puoi contare fino a 5?", "Dimmi i numeri da uno a cinque", "Conta da 1 a 5", "Conta fino a 5"], "Uno, due, tre, quattro, cinque."),
    (["Qual è la capitale d'Italia?", "Capitale italiana", "Dimmi la capitale d'Italia", "La capitale dell'Italia?", "Qual è la capitale italiana?"], "La capitale d'Italia è Roma."),
    (["Qual è la capitale francese?", "Capitale della Francia", "Dimmi la capitale di Francia", "La capitale della Francia?", "Qual è la capitale di Francia?"], "La capitale della Francia è Parigi."),
    (["Qual è la capitale spagnola?", "Capitale della Spagna", "Dimmi la capitale di Spagna", "La capitale della Spagna?", "Qual è la capitale della Spagna?"], "La capitale della Spagna è Madrid."),
    (["Cos'è il Sole?", "Spiega il Sole", "Cosa è il Sole?", "Parlami del Sole", "Che cos'è il Sole?"], "Il Sole è la stella al centro del nostro sistema solare."),
    (["Cos'è la Luna?", "Spiega la Luna", "Cosa è la Luna?", "Parlami della Luna", "Che cos'è la Luna?"], "La Luna è il satellite naturale della Terra."),
    (["Cos'è l'acqua?", "Spiega l'acqua", "Cosa è l'acqua?", "Parlami dell'acqua", "Che cos'è l'acqua?"], "L'acqua è una sostanza composta da idrogeno e ossigeno: H2O."),
    (["Perché il cielo è blu?", "Come mai il cielo è blu?", "Spiega il colore del cielo", "Perché vediamo il cielo blu?", "Perché il cielo sembra blu?"], "L'atmosfera diffonde la luce blu del Sole più delle altre componenti."),
    (["Raccontami una storia", "Inventami una storia", "Dimmi una storia breve", "Voglio una storia", "Puoi raccontare una storia?"], "Una piccola stella perse la strada. La Luna le mostrò come tornare a casa."),
    (["Scrivi una poesia", "Fammi una poesia", "Dimmi una poesia breve", "Voglio una poesia", "Puoi scrivere una poesia?"], "La sera si posa sul mare, una stella comincia a brillare."),
    (["Dimmi una barzelletta", "Fammi ridere", "Racconta una battuta", "Dimmi qualcosa di divertente", "Sai una barzelletta?"], "Perché il computer aveva freddo? Aveva lasciato Windows aperto!"),
    (["Sono triste", "Mi sento giù", "Oggi sono triste", "Ho una brutta giornata", "Mi sento un po' triste"], "Mi dispiace. Ti va di raccontare cosa è successo?"),
    (["Sono felice", "Oggi sono contento", "Mi sento felice", "Ho una bella giornata", "Sono molto felice"], "Mi fa piacere! Che cosa ha reso bella la tua giornata?"),
    (["Sono stanco", "Mi sento stanca", "Sono molto stanco", "Ho bisogno di riposo", "Oggi sono stanco"], "Una pausa può aiutare. Se puoi, prenditi un momento per riposare."),
    (["Sono annoiato", "Mi annoio", "Non so cosa fare", "Mi sto annoiando", "Oggi mi annoio"], "Prova una passeggiata, un disegno o una piccola storia da inventare."),
    (["Come posso studiare?", "Dammi un consiglio per studiare", "Come studio meglio?", "Aiutami a studiare", "Consigli per lo studio?"], "Dividi il lavoro in piccoli passi e fai brevi pause regolari."),
    (["Come imparo a programmare?", "Voglio imparare Python", "Come inizio a programmare?", "Aiutami a imparare il codice", "Voglio imparare a programmare"], "Inizia con variabili, condizioni e cicli. Poi costruisci piccoli progetti."),
    (["Mi dai una ricetta?", "Cosa posso cucinare?", "Dammi un'idea per pranzo", "Suggerisci un piatto semplice", "Un'idea per cena?"], "Puoi preparare pasta al pomodoro con basilico e un filo d'olio."),
    (["Qual è il tuo colore preferito?", "Che colore ti piace?", "Hai un colore preferito?", "Dimmi un colore", "Quale colore preferisci?"], "Non ho gusti personali, ma il blu mi ricorda il cielo."),
    (["Hai memoria?", "Ricordi le altre chat?", "Ti ricordi di me?", "Ricordi le conversazioni?", "Hai una memoria?"], "Ricordo solo l'ultimo scambio, se è breve. La cronologia resta qui."),
    (["Non ho capito", "Puoi spiegare meglio?", "Spiegalo più semplice", "Non capisco", "Puoi semplificare?"], "Proviamo con una domanda breve e un solo argomento alla volta."),
    (["Hai sbagliato", "La risposta è sbagliata", "Non è corretto", "Questo è un errore", "La tua risposta è errata"], "Grazie per la correzione. Il mio piccolo modello può fare molti errori."),
    (["Come funziona questa app?", "Come si usa Lumina?", "Come uso la chat?", "Spiega questa applicazione", "Come posso usare la chat?"], "Scrivi una domanda breve oppure attiva il microfono, poi invia."),
    (["Ti piace la musica?", "Ascolti musica?", "Hai una canzone preferita?", "Parliamo di musica", "Ti piace ascoltare musica?"], "Non ascolto come una persona, ma possiamo parlare di musica."),
    (["Cos'è un gatto?", "Parlami dei gatti", "Cosa è un gatto?", "Che animale è il gatto?", "Dimmi qualcosa sui gatti"], "Il gatto è un mammifero domestico curioso e agile."),
    (["Cos'è un cane?", "Parlami dei cani", "Cosa è un cane?", "Che animale è il cane?", "Dimmi qualcosa sui cani"], "Il cane è un mammifero domestico spesso molto socievole."),
    (["Cos'è un libro?", "Parlami dei libri", "Cosa è un libro?", "A cosa serve un libro?", "Che cosa è un libro?"], "Un libro raccoglie parole e idee per raccontare o insegnare qualcosa."),
    (["Come funzioni?", "Spiegami come funzioni", "Come generi le risposte?", "Come produci il testo?", "Qual è il tuo funzionamento?"], "Prevedo un byte alla volta usando una piccola rete neurale nel browser."),
    (["Ciao! Come stai?", "Ciao, come stai?", "Ehi, come va?", "Buongiorno, come stai?", "Ciao Lumina, come stai?"], "Ciao! Sono pronta a parlare con te. Come stai tu?"),
    (["Racconta una storia", "Raccontami una breve storia.", "Raccontami una storia breve", "Una storia breve, per favore", "Mi racconti una storia?"], "Una piccola stella perse la strada. La Luna le mostrò come tornare a casa."),
]


REPEAT = ["Ripeti", "Puoi ripetere?", "Come hai detto?", "Ripetilo", "Dillo di nuovo"]
MAX_ANSWER_BYTES = 77  # keeps prompt (<= 84 bytes with memory) plus answer inside the 165-byte context


def previous_that_fits(rng, pool, question):
    """Pick a prior exchange whose text is actually included by build_prompt."""
    plain = len(build_prompt(question))
    for _ in range(40):
        prev = rng.choice(pool)
        if len(build_prompt(question, prev)) != plain:
            return prev
    return None


def attach(row, prev):
    row["previous"] = {"user": prev[0], "assistant": prev[1]}
    return row


if __name__ == "__main__":
    path = Path(__file__).with_name("dialogues.jsonl")
    rng = random.Random(7)
    rows = []
    pool = [(questions[0], answer) for questions, answer in TOPICS]
    for questions, answer in TOPICS:
        assert len(answer.encode("utf-8")) <= MAX_ANSWER_BYTES, answer
        # Surface variants expose case and punctuation during training; they
        # never substitute a rule-based answer at inference time.
        seen = set()
        for question in questions[:-1]:
            base = (question, question.lower(), question.rstrip("?.!"), question.lower().rstrip("?.!"))
            # Conversational prefixes add wording diversity to reduce memorization.
            prefixed = tuple(p + question.lower().rstrip("?.!") for p in PREFIXES)
            for variant in base + prefixed:
                if variant not in seen:
                    rows.append({"user": variant, "assistant": answer, "split": "train", "kind": "topic"})
                    seen.add(variant)
                    # About a third of the examples also carry an unrelated earlier exchange, so
                    # the model learns that memory is context and not a reason to change topics.
                    prev = previous_that_fits(rng, pool, variant) if rng.random() < 0.35 else None
                    if prev:
                        rows.append(attach({"user": variant, "assistant": answer, "split": "train", "kind": "topic"}, prev))
        rows.append({"user": questions[-1], "assistant": answer, "split": "validation", "kind": "topic"})
    # Honest refusal for questions the corpus cannot answer.
    assert len(REFUSAL.encode("utf-8")) <= MAX_ANSWER_BYTES
    for question in OOS_TRAIN:
        for variant in dict.fromkeys((question, question.lower(), question.rstrip("?.!"), question.lower().rstrip("?.!"))):
            rows.append({"user": variant, "assistant": REFUSAL, "split": "train", "kind": "refusal"})
    for question in OOS_VALIDATION:
        rows.append({"user": question, "assistant": REFUSAL, "split": "validation", "kind": "refusal"})
    # Rule-generated arithmetic tables and definition frames (some frames and pairs held out).
    for question, answer, split in arithmetic_rows(rng):
        assert len(answer.encode("utf-8")) <= MAX_ANSWER_BYTES, answer
        rows.append({"user": question, "assistant": answer, "split": split, "kind": "arithmetic"})
    answers = {questions[0]: answer for questions, answer in TOPICS}
    for question, answer, split in definition_rows(answers):
        rows.append({"user": question, "assistant": answer, "split": split, "kind": "definition"})
    # Memory: a repeat request is answered with the previous answer.
    for index, question in enumerate(REPEAT):
        held_out = index == len(REPEAT) - 1
        variants = [question] if held_out else list(dict.fromkeys((question, question.lower(), question.rstrip("?.!"), question.lower().rstrip("?.!"))
                                                                  + tuple(p + question.lower().rstrip("?.!") for p in PREFIXES)))
        for variant in variants:
            for _ in range(6 if held_out else 3):
                prev = previous_that_fits(rng, pool, variant)
                if prev:
                    rows.append(attach({"user": variant, "assistant": prev[1], "split": "validation" if held_out else "train", "kind": "memory"}, prev))
    # Drop exact duplicates (same prompt, memory and answer).
    unique, seen_rows = [], set()
    for row in rows:
        key = (row["user"], json.dumps(row.get("previous"), ensure_ascii=False, sort_keys=True), row["assistant"], row["split"])
        if key not in seen_rows:
            seen_rows.add(key)
            unique.append(row)
    rows = unique
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    train = sum(1 for row in rows if row["split"] == "train")
    print(f"Wrote {len(rows)} original examples to {path} ({train} train, {len(rows) - train} validation)")
