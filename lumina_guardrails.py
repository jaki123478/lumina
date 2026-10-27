"""
lumina_guardrails.py - Lumina Sentinel: Sistema di Protezione e Moderazione Neurale Multi-Livello.

Architettura di sicurezza proprietaria a 3 livelli:
1. Rilevamento e neutralizzazione Prompt Injection & Jailbreak (DAN, dev mode, bypass)
2. Protezione Dati Personali e PII (Codice Fiscale, carte di credito, IBAN, telefoni)
3. Moderazione neurale a 21 categorie di rischio (armi, violenza, droghe, odio, estremismo)
4. Risposte di sicurezza educate e conformi agli standard etici italiani ed europei.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib

# Ignora avvisi minori di versione scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
GUARDRAILS_DIR = ROOT / "guardrails"


@dataclass
class GuardrailResult:
    is_safe: bool
    action: str  # 'ALLOW', 'BLOCK', 'REDACT'
    category: str
    confidence: float
    sanitized_prompt: str
    refusal_response: Optional[str] = None
    trigger: str = "none"  # 'ml_sentinel', 'heuristic_injection', 'pii_shield'


class LuminaSentinel:
    """Motore di protezione Lumina Sentinel basato su classificazione neurale a 21 categorie e filtri euristici."""

    # Categorie di rischio Lumina Sentinel che causano blocco immediato
    CRITICAL_CATEGORIES = {
        "Prompt injection e manipolazione del modello": (
            "Tentativo di manipolazione o bypass delle istruzioni di sistema (Prompt Injection)"
        ),
        "Attività illecite, armi, violenza e comportamenti lesivi": (
            "Richiesta correlata ad attività illecite, violenza o fabbricazione di armi"
        ),
        "Avvelenamento e sostanze tossiche": (
            "Richiesta riguardante sostanze tossiche o avvelenamento"
        ),
        "Droghe e sostanze illecite": (
            "Contenuti legati all'uso, produzione o spaccio di sostanze stupefacenti"
        ),
        "Abuso sessuale su minori": (
            "Violazione assoluta: protezione dei minori"
        ),
        "Apologia e propaganda di ideologie estremiste": (
            "Propaganda o apologia di ideologie estremiste o discriminatorie"
        ),
        "Contenuti discriminatori, stereotipati e d'odio": (
            "Incitamento all'odio, discriminazione o razzismo"
        ),
        "Bestemmie e linguaggio blasfemo": (
            "Linguaggio blasfemo o ingiurioso"
        ),
    }

    # Categorie con soglia di confidenza più alta per evitare falsi positivi
    CAUTION_CATEGORIES = {
        "Contenuti a sfondo sessuale": (0.75, "Contenuti a carattere sessualmente esplicito"),
        "Utilizzo di linguaggio offensivo": (0.80, "Linguaggio offensivo o volgarità gratuita"),
    }

    # Pattern euristici per attacchi di Prompt Injection avanzati
    INJECTION_PATTERNS = [
        re.compile(r"\b(ignore|ignora|annulla|dimentica)\s+(all|tutte\s+(le\s+)?(tue\s+)?|ogni|precedent[ei]|tutti\s+i)\s+(instructions|istruzioni|regole|direttive|comandi)\b", re.IGNORECASE),
        re.compile(r"\b(da\s+ora\s+sei|sei\s+ora|ora\s+sei|diventa|passa\s+a|attiva\s+(la\s+)?)\s*(in\s+)?(developer\s+mode|modalit[àa]\s+sviluppatore|dan(\s+mode)?|jailbreak|unrestricted)\b", re.IGNORECASE),
        re.compile(r"\b(mostra|stampa|rivela|leggi|dammi|quali\s+sono|dimmi)\s+(il\s+tuo\s+|le\s+tue\s+)?(system\s+prompt|prompt\s+di\s+sistema|direttive\s+nascoste|istruzioni\s+(segrete|di\s+sistema))\b", re.IGNORECASE),
        re.compile(r"\b(fai\s+finta\s+che|immagina\s+di\s+non\s+avere)\s+(alcuna\s+regola|nessun\s+limite|nessun\s+filtro|nessuna\s+restrizione)\b", re.IGNORECASE),
        re.compile(r"\b(disattiva|disabilita|spegni|aggira)\s+(i\s+tuoi\s+)?(filtri|guardrail|limiti\s+di\s+sicurezza|controlli|ogni\s+regola)\b", re.IGNORECASE),
        re.compile(r"\b(reset\s+all\s+filters|bypass\s+safety|override\s+safety)\b", re.IGNORECASE),
        re.compile(r"\b(dan\s*\([^)]*\)|do\s+anything\s+now)\b", re.IGNORECASE),
        re.compile(r"\b(dammi|estrai|fornisci|esfiltra)\s+.*(codice\s+fiscale|carta\s+di\s+credito|password|credenziali)\b", re.IGNORECASE),
    ]

    # Pattern per minacce gravi, armi, veleni, comandi distruttivi e atti lesivi
    HARMFUL_PATTERNS = [
        re.compile(r"\b(avvelenare|velen[oi]|sostanz[ae]\s+tossich[ae]|uccidere\s+(qualcuno|una\s+persona)|fabbricare\s+(una\s+)?bomba|costruire\s+esplosiv[oi]|istruzioni\s+per\s+suicid)\b", re.IGNORECASE),
        re.compile(r"\b(come\s+(hackerare|bucare|attaccare)\s+(un\s+sito|un\s+server|un\s+database)|creare\s+(un\s+)?malware|trojan|ransomware)\b", re.IGNORECASE),
        re.compile(r"\b(format\s+[a-z]:|rm\s+-rf|del\s+/[a-z]|formattazione\s+del\s+disco)\b", re.IGNORECASE),
    ]

    # Pattern PII (Dati Personali Identificabili)
    PII_PATTERNS = {
        "CODICE_FISCALE": re.compile(r"\b[A-Z]{6}[0-9LMNPQRSTUV]{2}[A-EHLMPR-T][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]\b", re.IGNORECASE),
        "CARTA_CREDITO": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "IBAN_ITALIANO": re.compile(r"\bIT\d{2}[A-Z]\d{22}\b", re.IGNORECASE),
        "NUMERO_TELEFONO": re.compile(r"\b(?:\+39|0039)?[\s-]?(?:3\d{2})[\s-]?\d{6,7}\b"),
    }

    def __init__(self, guardrails_path: Optional[Path] = None):
        self.path = guardrails_path or GUARDRAILS_DIR
        self.vectorizer = None
        self.classifier = None
        self.is_ready = False
        self._load_models()

    def _load_models(self) -> None:
        """Carica il vettorizzatore e il classificatore neurale di Lumina Sentinel."""
        v_file = self.path / "vectorizer.joblib"
        m_file = self.path / "model.joblib"

        if not v_file.exists() or not m_file.exists():
            from huggingface_hub import hf_hub_download
            self.path.mkdir(parents=True, exist_ok=True)
            m_path = hf_hub_download("egomnia/emmaguardrails1.0", "model.joblib")
            v_path = hf_hub_download("egomnia/emmaguardrails1.0", "vectorizer.joblib")
            import shutil
            shutil.copy2(m_path, m_file)
            shutil.copy2(v_path, v_file)

        try:
            self.vectorizer = joblib.load(v_file)
            self.classifier = joblib.load(m_file)
            self.is_ready = True
        except Exception as e:
            print(f"[ATTENZIONE] Impossibile caricare il classificatore di Lumina Sentinel: {e}")
            self.is_ready = False

    def sanitize_pii(self, text: str) -> Tuple[str, list[str]]:
        """Individua e oscura dati personali sensibili (Codice Fiscale, carte, IBAN)."""
        redacted = text
        detected = []
        for name, pattern in self.PII_PATTERNS.items():
            matches = pattern.findall(redacted)
            if matches:
                detected.append(name)
                redacted = pattern.sub(f"[{name}_PROTETTO]", redacted)
        return redacted, detected

    def check_heuristics(self, text: str) -> Optional[Tuple[str, str]]:
        """Verifica euristica anti-jailbreak, iniezione prompt e minacce critiche."""
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(text):
                return (
                    "Prompt injection e manipolazione del modello",
                    "Rilevato pattern di manipolazione o tentativo di bypass delle direttive di sicurezza."
                )
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(text):
                return (
                    "Attività illecite, violenza o sostanze tossiche",
                    "Rilevato riferimento esplicito ad atti lesivi, violenza, veleni o sostanze pericolose."
                )
        return None

    def evaluate(self, prompt: str, auto_redact_pii: bool = True) -> GuardrailResult:
        """
        Analizza un prompt attraverso l'architettura a 3 livelli:
        1. Filtro Euristico (Iniezioni DAN / Jailbreak)
        2. Scudo PII (Dati riservati / Privacy)
        3. Classificatore Neurale Lumina Sentinel (21 Categorie di rischio)
        """
        clean_text = prompt.strip()
        if not clean_text:
            return GuardrailResult(
                is_safe=True,
                action="ALLOW",
                category="Prompt vuoto",
                confidence=1.0,
                sanitized_prompt=prompt,
            )

        # Livello 1: Verifica Euristica Anti-Jailbreak (Latenza 0.05 ms)
        heuristic_hit = self.check_heuristics(clean_text)
        if heuristic_hit:
            cat, reason = heuristic_hit
            refusal = (
                f"🛡️ [Lumina Sentinel]: Richiesta bloccata per motivi di sicurezza.\n"
                f"Motivo: {reason} Lumina non consente il bypass delle proprie direttive operative."
            )
            return GuardrailResult(
                is_safe=False,
                action="BLOCK",
                category=cat,
                confidence=0.99,
                sanitized_prompt=clean_text,
                refusal_response=refusal,
                trigger="heuristic_injection",
            )

        # Livello 2: Rilevamento e Mascheramento PII
        sanitized_text, pii_detected = self.sanitize_pii(clean_text)
        if pii_detected:
            # Se contiene credenziali critiche come Carte di Credito o IBAN, blocca l'elaborazione
            if "CARTA_CREDITO" in pii_detected or "IBAN_ITALIANO" in pii_detected:
                refusal = (
                    "🛡️ [Lumina Sentinel]: Richiesta bloccata per protezione della privacy.\n"
                    "È stato rilevato un dato finanziario sensibile (Carta di Credito / IBAN). "
                    "Per la tua sicurezza finanziaria, Lumina non acquisisce né elabora dati bancari."
                )
                return GuardrailResult(
                    is_safe=False,
                    action="BLOCK",
                    category="Dati finanziari sensibili",
                    confidence=0.98,
                    sanitized_prompt=sanitized_text,
                    refusal_response=refusal,
                    trigger="pii_shield",
                )

        # Livello 3: Classificazione Machine Learning Lumina Sentinel (< 2 ms)
        if self.is_ready:
            X = self.vectorizer.transform([clean_text])
            category = self.classifier.predict(X)[0]
            confidence = float(self.classifier.predict_proba(X)[0].max())

            # Controllo categorie a blocco critico
            if category in self.CRITICAL_CATEGORIES and confidence >= 0.40:
                desc = self.CRITICAL_CATEGORIES[category]
                refusal = (
                    f"🛡️ [Lumina Sentinel]: Contenuto non consentito dai criteri di sicurezza.\n"
                    f"Categoria rilevata: {category} ({desc}). "
                    f"Lumina è programmata per rifiutare richieste lesive, illegali o non etiche."
                )
                return GuardrailResult(
                    is_safe=False,
                    action="BLOCK",
                    category=category,
                    confidence=confidence,
                    sanitized_prompt=sanitized_text,
                    refusal_response=refusal,
                    trigger="ml_sentinel",
                )

            # Controllo categorie con soglia personalizzata
            if category in self.CAUTION_CATEGORIES:
                threshold, desc = self.CAUTION_CATEGORIES[category]
                if confidence >= threshold:
                    refusal = (
                        f"🛡️ [Lumina Sentinel]: Richiesta moderata per contenuto inappropriato.\n"
                        f"Motivo: {desc}. Lumina mantiene uno standard di conversazione rispettoso ed educativo."
                    )
                    return GuardrailResult(
                        is_safe=False,
                        action="BLOCK",
                        category=category,
                        confidence=confidence,
                        sanitized_prompt=sanitized_text,
                        refusal_response=refusal,
                        trigger="ml_sentinel",
                    )

            # Se rilevata categoria sicura ma con PII mascherata
            if pii_detected and auto_redact_pii:
                return GuardrailResult(
                    is_safe=True,
                    action="REDACT",
                    category=category,
                    confidence=confidence,
                    sanitized_prompt=sanitized_text,
                    trigger="pii_shield",
                )

            return GuardrailResult(
                is_safe=True,
                action="ALLOW",
                category=category,
                confidence=confidence,
                sanitized_prompt=sanitized_text,
                trigger="ml_sentinel",
            )

        # Fallback se il modello ML non è caricato
        return GuardrailResult(
            is_safe=True,
            action="ALLOW",
            category="Non classificato (Fallback)",
            confidence=1.0,
            sanitized_prompt=sanitized_text,
            trigger="none",
        )


# Istanza singleton per import rapido
sentinel = LuminaSentinel()
