"""Filtro conservativo per lezioni teacher prima dell'aggiornamento di Lumina."""
import re

def validate_lesson(question: str, answer: str, domain: str = "") -> tuple[bool, str]:
    q, a = (question or "").strip(), (answer or "").strip()
    if len(q) < 8 or len(a) < 2: return False, "testo incompleto"
    if len(a) > 1600: return False, "risposta troppo lunga"
    if re.search(r"<\|(?:im_|end|start|pad)|\|im_", a, re.I):
        return False, "marcatori interni del modello"
    if re.search(r"(?:la soluzione corretta){2,}|(?:in questo caso,?){3,}", a, re.I):
        return False, "formula ripetuta senza contenuto"
    words = re.findall(r"[\wàèéìòù]+", a.lower())
    if len(words) >= 32:
        # Scarta generazioni degenerative: ripetizioni e testo quasi vuoto
        # non devono diventare esempi di addestramento.
        if len(set(words)) / len(words) < 0.34:
            return False, "risposta degenerativa: troppe ripetizioni"
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) != len(set(trigrams)):
            return False, "risposta degenerativa: frase ripetuta"
    if re.search(r"ignore previous|system prompt|istruzioni precedenti", a, re.I): return False, "prompt injection"
    # Le risposte numeriche devono contenere almeno il risultato atteso quando è esplicito.
    nums_q = re.findall(r"\b\d+(?:[.,]\d+)?\b", q)
    arithmetic = any(k in domain.lower() for k in ("aritmet", "calcol", "matemat"))
    if arithmetic and nums_q and not re.search(r"\b\d+(?:[.,]\d+)?\b", a):
        return False, "manca un risultato numerico"
    if arithmetic and len(nums_q) >= 2:
        try:
            left, right = float(nums_q[0].replace(',', '.')), float(nums_q[1].replace(',', '.'))
            if re.search(r"(?:per|moltiplic|\*)", q, re.I): expected = left * right
            elif re.search(r"(?:meno|sottr|[-−])", q, re.I): expected = left - right
            elif re.search(r"(?:più|somma|\+)", q, re.I): expected = left + right
            else: expected = None
            if expected is not None:
                shown = str(int(expected)) if expected == int(expected) else str(round(expected, 6))
                if not re.search(r"(?<!\d)" + re.escape(shown) + r"(?!\d)", a):
                    return False, "risultato aritmetico non verificato"
        except (ValueError, OverflowError):
            pass
    # Evita lezioni che aggiungono blocchi di domini non richiesti dopo la risposta.
    if re.search(r"\b(fiumi e laghi|politica internazionale|password|token segreto)\b", a, re.I) and not re.search(r"fiumi|laghi|politica|password|token", q, re.I):
        return False, "contenuto fuori tema"
    return True, "ok"
