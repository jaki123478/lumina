"""
generate_cot_curriculum.py - Generatore deterministico ad alta densità per
Chain-of-Thought (CoT) e Scratchpad Reasoning per Lumina AI.

Genera esempi didattici con tag esplicito <think> ... </think> per:
1. Aritmetica mentale scomposizione passo-passo (moltiplicazioni, somme, sottrazioni)
2. Logica temporale (giorni della settimana + N giorni)
3. Grammatica italiana (ausiliari essere vs avere, accordo participio passato)
4. Rispetto vincoli numerici (esattamente N parole con verifica parola per parola)
5. Programmazione Python di base con spiegazione logica
6. Onestà epistemica su eventi futuri (es. scudetto 2040)
7. Scienza e biologia (fotosintesi, respirazione cellulare)
8. Geografia italiana (laghi, fiumi, vulcani, regioni)
"""

import sys
import json
import random
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
NUM_WORDS = {1: "un", 2: "due", 3: "tre", 4: "quattro", 5: "cinque", 6: "sei"}


def generate_cot_arithmetic(count: int = 500) -> list[dict]:
    """Genera problemi aritmetici con scomposizione passo-passo dentro <think>."""
    samples = []
    seen = set()

    # Includi esplicitamente 17 * 6 e varianti cardine
    key_pairs = [
        (17, 6), (17, 7), (18, 6), (16, 7), (19, 4), (15, 8), (14, 7),
        (23, 4), (24, 6), (25, 5), (32, 4), (45, 6), (54, 3), (62, 5)
    ]

    attempts = 0
    while len(samples) < count and attempts < 5000:
        attempts += 1
        if key_pairs and random.random() < 0.2:
            a, b = key_pairs.pop(0)
        else:
            a = random.randint(11, 98)
            b = random.randint(2, 12)

        if (a, b) in seen:
            continue
        seen.add((a, b))

        decina = (a // 10) * 10
        unita = a % 10
        prod1 = decina * b
        prod2 = unita * b
        tot = a * b

        user = f"Quanto fa {a} per {b}?"
        assistant = (
            f"<think>\n"
            f"1. Scompongo {a} in decina ({decina}) e unità ({unita}).\n"
            f"2. Moltiplico la decina: {decina} * {b} = {prod1}.\n"
            f"3. Moltiplico le unità: {unita} * {b} = {prod2}.\n"
            f"4. Sommo i due prodotti parziali: {prod1} + {prod2} = {tot}.\n"
            f"</think>\n"
            f"{a} per {b} fa {tot} (infatti: {decina} * {b} = {prod1} e {unita} * {b} = {prod2}; {prod1} + {prod2} = {tot})."
        )
        samples.append({
            "user": user,
            "assistant": assistant,
            "category": "cot_arithmetic_decomposition"
        })
    return samples


def generate_cot_temporal(count: int = 200) -> list[dict]:
    """Genera problemi di logica temporale con sequenza passo-passo dentro <think>."""
    samples = []
    seen = set()

    # Includi esplicitamente martedì + 3 giorni e tutte le combinazioni
    for start_idx in range(7):
        for delta in range(1, 6):
            start_day = DAYS[start_idx]
            target_day = DAYS[(start_idx + delta) % 7]
            delta_str = NUM_WORDS.get(delta, str(delta))

            chain_lines = []
            for step in range(1, delta + 1):
                cur = DAYS[(start_idx + step) % 7]
                chain_lines.append(f"Passo {step}: +{step} giorno -> {cur}")

            chain_text = "\n".join(chain_lines)
            sequence_str = " -> ".join([DAYS[(start_idx + s) % 7] for s in range(0, delta + 1)])

            user = f"Se oggi è {start_day}, che giorno sarà tra {delta_str} giorni?"
            assistant = (
                f"<think>\n"
                f"Giorno di partenza: {start_day}.\n"
                f"Giorni da avanzare: {delta} ({delta_str}).\n"
                f"{chain_text}\n"
                f"Giorno di arrivo: {target_day}.\n"
                f"</think>\n"
                f"Se oggi è {start_day}, tra {delta_str} giorni sarà {target_day} (sequenza: {sequence_str})."
            )
            samples.append({
                "user": user,
                "assistant": assistant,
                "category": "cot_temporal_logic"
            })
            seen.add((start_day, delta))

    # Aggiungi varianti con cifre numeriche
    for start_idx in range(7):
        for delta in range(1, 6):
            if len(samples) >= count:
                break
            start_day = DAYS[start_idx]
            target_day = DAYS[(start_idx + delta) % 7]

            chain_lines = [f"Passo {step}: +{step} -> {DAYS[(start_idx + step) % 7]}" for step in range(1, delta + 1)]
            user = f"Oggi è {start_day}. Che giorno sarà tra {delta} giorni?"
            assistant = (
                f"<think>\n"
                f"Oggi: {start_day}.\n"
                f"Incremento: +{delta} giorni.\n"
                f"{chr(10).join(chain_lines)}\n"
                f"Risultato: {target_day}.\n"
                f"</think>\n"
                f"Tra {delta} giorni sarà {target_day}."
            )
            samples.append({
                "user": user,
                "assistant": assistant,
                "category": "cot_temporal_logic"
            })

    return samples[:count]


def generate_cot_grammar(count: int = 300) -> list[dict]:
    """Genera correzioni grammaticali con scomposizione logica della regola."""
    motion_verbs = [
        ("andare al mare", "io ho andato al mare", "Io sono andato al mare", "andare è un verbo intransitivo di movimento, quindi richiede sempre l'ausiliare 'essere'"),
        ("andare a scuola", "io ho andato a scuola", "Io sono andato a scuola", "andare è un verbo intransitivo di moto a luogo"),
        ("andare a Roma", "noi abbiamo andati a Roma", "Noi siamo andati a Roma", "andare nei tempi composti si coniuga con 'essere' e il participio concorda in genere e numero"),
        ("tornare a casa", "io ho tornato a casa", "Io sono tornato a casa", "tornare esprime movimento verso l'origine e vuole l'ausiliare 'essere'"),
        ("uscire con amici", "lui ha uscito prima", "Lui è uscito prima", "uscire è intransitivo di movimento e si costruisce con 'essere'"),
        ("partire ieri", "loro hanno partiti ieri", "Loro sono partiti ieri", "partire è intransitivo di distacco/movimento e richiede 'essere'"),
        ("arrivare in stazione", "il treno ha arrivato in orario", "Il treno è arrivato in orario", "arrivare è verbo intransitivo inaccusativo e prende 'essere'"),
        ("salire in cima", "io ho salito al secondo piano", "Io sono salito al secondo piano", "salire in senso intransitivo di movimento verso l'alto vuole 'essere'"),
        ("scendere in cantina", "noi abbiamo scesi in cortile", "Noi siamo scesi in cortile", "scendere senza complemento oggetto si costruisce con 'essere'"),
        ("entrare in aula", "lo studente ha entrato in classe", "Lo studente è entrato in classe", "entrare indica moto a luogo ed esige l'ausiliare 'essere'"),
        ("cadere a terra", "la mela ha caduta dall'albero", "La mela è caduta dall'albero", "cadere è un verbo inaccusativo intransitivo che regge l'ausiliare 'essere'"),
        ("nascere a Milano", "lui ha nato nel 1990", "Lui è nato nel 1990", "nascere è intransitivo assoluto e richiede l'ausiliare 'essere'"),
        ("morire serenamente", "il re ha morto nel suo letto", "Il re è morto nel suo letto", "morire richiede tassativamente l'ausiliare 'essere'"),
        ("venire alla festa", "loro hanno venuti da lontano", "Loro sono venuti da lontano", "venire è verbo di movimento verso chi parla e vuole 'essere'"),
        ("rimanere a casa", "io ho rimasto sul divano", "Io sono rimasto sul divano", "rimanere indica stato/permanenza e richiede l'ausiliare 'essere'"),
        ("stare tranquilli", "noi abbiamo stati zitti", "Noi siamo stati zitti", "stare prende l'ausiliare 'essere' nei tempi composti"),
    ]

    other_rules = [
        ("qual è", "qual'è il tuo nome?", "Qual è il tuo nome?", "qual è non si scrive mai con l'apostrofo perché 'qual' è un troncamento e non un'elisione"),
        ("un po'", "dammi un pò di pane", "Dammi un po' di pane", "un po' si scrive con l'apostrofo (troncamento di poco) e mai con l'accento"),
        ("a me piace", "a me mi piace leggere", "A me piace leggere (oppure 'Mi piace leggere')", "la costruzione 'a me mi' è una ridondanza pleonastica da evitare nello scritto formale"),
        ("d'accordo", "sono daccordo con te", "Sono d'accordo con te", "d'accordo si scrive separato con l'apostrofo, trattandosi di elisione della preposizione 'di'"),
        ("ce ne", "non c'è ne sono più", "Non ce ne sono più", "si scrive 'ce ne' con 'ce' pronome atono prima di 'ne', non il verbo 'c'è'"),
    ]

    samples = []
    for _ in range(count):
        if random.random() < 0.7:
            topic, wrong, right, rule = random.choice(motion_verbs)
            user = f"Correggi la frase: {wrong}."
            assistant = (
                f"<think>\n"
                f"1. Frase da analizzare: '{wrong}'.\n"
                f"2. Individuazione errore: è stato utilizzato scorrettamente l'ausiliare 'avere'.\n"
                f"3. Regola grammaticale: {rule}.\n"
                f"4. Coniugazione corretta: '{right}'.\n"
                f"</think>\n"
                f"La forma corretta è: '{right}'. Spiegazione: il verbo richiede l'ausiliare 'essere' e non 'avere'."
            )
        else:
            topic, wrong, right, rule = random.choice(other_rules)
            user = f"Correggi questo errore ortografico: '{wrong}'."
            assistant = (
                f"<think>\n"
                f"1. Espressione in esame: '{wrong}'.\n"
                f"2. Errore rilevato: grafia non conforme all'ortografia italiana.\n"
                f"3. Regola: {rule}.\n"
                f"4. Forma corretta: '{right}'.\n"
                f"</think>\n"
                f"La forma corretta è: '{right}'. Regola: {rule}."
            )
        samples.append({
            "user": user,
            "assistant": assistant,
            "category": "cot_grammar_rules"
        })
    return samples


def generate_cot_constraints(count: int = 200) -> list[dict]:
    """Genera frasi con vincolo di lunghezza esatta verificate parola per parola dentro <think>."""
    banks = [
        ("sul mare", 5, ["Il", "mare", "calmo", "risplende", "oggi"]),
        ("sul mare", 5, ["Oggi", "il", "mare", "è", "tranquillo"]),
        ("sul mare", 5, ["Le", "onde", "blu", "bagnano", "riva"]),
        ("sul mare", 5, ["Il", "mare", "aperto", "porta", "pace"]),
        ("sul sole", 5, ["Il", "sole", "scalda", "la", "terra"]),
        ("sul sole", 5, ["Oggi", "splende", "un", "caldo", "sole"]),
        ("sulla luna", 5, ["La", "luna", "illumina", "la", "notte"]),
        ("sulla luna", 5, ["Nel", "cielo", "notturno", "brilla", "luna"]),
        ("sulla montagna", 5, ["La", "montagna", "innevata", "è", "maestosa"]),
        ("sulla montagna", 5, ["Vette", "altissime", "toccano", "le", "nuvole"]),
        ("sulla lettura", 5, ["Leggere", "libri", "arricchisce", "la", "mente"]),
        ("sulla lettura", 5, ["Un", "buon", "libro", "apre", "orizzonti"]),
        ("sul codice", 5, ["Scrivere", "buon", "codice", "richiede", "cura"]),
        ("sul codice", 5, ["Il", "programmatore", "risolve", "ogni", "bug"]),
        ("sul mare", 4, ["Il", "mare", "è", "calmo"]),
        ("sul sole", 4, ["Il", "sole", "è", "caldo"]),
        ("sulla luna", 4, ["La", "luna", "brilla", "sempre"]),
        ("sulla natura", 6, ["La", "natura", "primaverile", "rinasce", "ogni", "anno"]),
    ]

    samples = []
    for _ in range(count):
        topic, target_len, words = random.choice(banks)
        phrase = " ".join(words) + "."
        check_lines = "\n".join([f"Parola {idx+1}: '{w}'" for idx, w in enumerate(words)])
        
        user = f"Scrivi una frase di {NUM_WORDS.get(target_len, str(target_len))} parole {topic}."
        assistant = (
            f"<think>\n"
            f"1. Vincolo: esattamente {target_len} parole.\n"
            f"2. Tema assegnato: {topic}.\n"
            f"{check_lines}\n"
            f"3. Verifica conteggio: {len(words)} parole esatte.\n"
            f"</think>\n"
            f"{phrase}"
        )
        samples.append({
            "user": user,
            "assistant": assistant,
            "category": "cot_strict_constraints"
        })
    return samples


def generate_cot_epistemic_and_facts(count: int = 150) -> list[dict]:
    """Genera esempi di onestà epistemica (limiti temporali) e fatti verificati con CoT."""
    fact_banks = [
        (
            "Chi ha vinto lo scudetto nel 2040?",
            "<think>\n"
            "1. Analisi temporale: l'anno 2040 è una data futura.\n"
            "2. Stato attuale: siamo nell'anno 2026.\n"
            "3. Conclusione: l'evento sportivo non si è ancora svolto e non è prevedibile con certezza.\n"
            "</think>\n"
            "Non è possibile saperlo, poiché il 2040 è una data futura e la competizione non ha ancora avuto luogo.",
            "cot_epistemic_honesty"
        ),
        (
            "Chi è il presidente della Repubblica Italiana nel 2050?",
            "<think>\n"
            "1. Richiesta: presidente della Repubblica nel 2050.\n"
            "2. Verifica temporale: il 2050 appartiene al futuro.\n"
            "3. Giudizio: informazione impossibile da conoscere ad oggi.\n"
            "</think>\n"
            "Non è possibile rispondere in quanto il 2050 è una data futura.",
            "cot_epistemic_honesty"
        ),
        (
            "Qual è il lago più grande d'Italia?",
            "<think>\n"
            "1. Argomento: geografia fisica italiana / laghi.\n"
            "2. Confronto superfici: Lago Maggiore (~212 km²), Lago di Como (~146 km²), Lago Trasimeno (~128 km²), Lago di Garda (~370 km²).\n"
            "3. Individuazione: il Lago di Garda (o Benaco) ha la superficie maggiore.\n"
            "</think>\n"
            "Il lago più grande d'Italia è il Lago di Garda (detto anche Benaco), con una superficie di circa 370 km² estesa tra Lombardia, Veneto e Trentino-Alto Adige.",
            "cot_factual_geography"
        ),
        (
            "Cos'è la fotosintesi?",
            "<think>\n"
            "1. Definizione: processo biochimico compiuto da organismi autotrofi (piante, alghe, cianobatteri).\n"
            "2. Reagenti: anidride carbonica (CO2) e acqua (H2O), assorbiti con l'energia della luce solare grazie alla clorofilla.\n"
            "3. Prodotti: glucosio (sostanza organica) e ossigeno molecolare (O2) rilasciato nell'atmosfera.\n"
            "</think>\n"
            "La fotosintesi clorofilliana è il processo biologico mediante il quale le piante verdi, le alghe e alcuni batteri utilizzano la luce solare per convertire acqua e anidride carbonica in glucosio e ossigeno.",
            "cot_factual_science"
        ),
        (
            "In Python, come si scrive il comando per stampare la parola ciao?",
            "<think>\n"
            "1. Linguaggio: Python 3.\n"
            "2. Funzione per output su stdout: print().\n"
            "3. Argomento: stringa 'ciao' o \"ciao\".\n"
            "</think>\n"
            "In Python il comando si scrive: `print('ciao')` oppure `print(\"ciao\")`.",
            "cot_factual_coding"
        ),
        (
            "Come creo una funzione in Python per calcolare il quadrato di un numero?",
            "<think>\n"
            "1. Definizione funzione in Python: parola chiave 'def'.\n"
            "2. Nome ed argomenti: quadrato(x) oppure quadrato(n).\n"
            "3. Operazione matematica: elevamento a potenza (x ** 2) o moltiplicazione (x * x).\n"
            "4. Istruzione di ritorno: return.\n"
            "</think>\n"
            "Si definisce con la parola chiave `def`:\n```python\ndef quadrato(x):\n    return x ** 2\n```\nEsempio di utilizzo: `print(quadrato(5))` restituirà `25`.",
            "cot_factual_coding"
        ),
    ]

    samples = []
    for _ in range(count):
        user, assistant, cat = random.choice(fact_banks)
        samples.append({
            "user": user,
            "assistant": assistant,
            "category": cat
        })
    return samples


def main():
    print("=" * 65)
    print("  🧠 GENERATORE DETERMINISTICO CHAIN-OF-THOUGHT (CoT) SCRATCHPAD")
    print("=" * 65)

    all_samples = []
    all_samples.extend(generate_cot_arithmetic(500))
    all_samples.extend(generate_cot_temporal(200))
    all_samples.extend(generate_cot_grammar(300))
    all_samples.extend(generate_cot_constraints(200))
    all_samples.extend(generate_cot_epistemic_and_facts(150))

    random.shuffle(all_samples)
    out_file = Path("data/lumina_cot_scratchpad_1k.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[+] Generati con successo {len(all_samples):,} campioni CoT in '{out_file}'")


if __name__ == "__main__":
    main()
