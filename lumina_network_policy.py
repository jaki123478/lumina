"""
lumina_network_policy.py - Gestore della Politica di Accesso a Internet ed Esplicito Consenso per Lumina AI.

Direttiva:
"Lumina richiede l'accesso a Internet solo se l'utente esplicita il consenso da Visual Studio, terminale o altre piattaforme.
Di default, Lumina opera al 100% in modalità sovrana e locale (offline-first, zero-telemetria)."
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "lumina_policy.json"

DEFAULT_POLICY = {
    "policy_version": "1.0.0",
    "sovereign_name": "Lumina Sentinel Network Shield",
    "internet_access_mode": "require_explicit_consent",
    "default_internet_allowed": False,
    "current_consent_granted": False,
    "consent_scope": "session",  # 'session' | 'permanent' | 'none'
    "visual_studio_integration": {
        "mode": "strictly_offline_unless_consented",
        "allow_local_lan": True,
        "prompt_on_external_request": True
    },
    "terminal_integration": {
        "mode": "strictly_offline_unless_consented",
        "interactive_prompt_enabled": True
    },
    "telemetry_policy": {
        "allow_telemetry": False,
        "allow_external_logging": False
    },
    "allowed_local_networks": [
        "127.0.0.1",
        "localhost",
        "0.0.0.0",
        "::1",
        "192.168.*",
        "10.*"
    ]
}


class LuminaNetworkPolicy:
    """Gestore centrale del consenso e della politica di rete per Lumina."""

    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config_path = config_path
        self.policy = self._load_policy()

    def _load_policy(self) -> dict:
        if not self.config_path.exists():
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            self._save_policy(DEFAULT_POLICY)
            return dict(DEFAULT_POLICY)
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {**DEFAULT_POLICY, **data}
        except Exception:
            return dict(DEFAULT_POLICY)

    def _save_policy(self, data: dict):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def is_internet_allowed(self) -> bool:
        """Verifica se l'accesso a Internet esterno è attualmente consentito."""
        # 1. Variabile d'ambiente prioritaria
        env_val = os.environ.get("LUMINA_ALLOW_INTERNET", "").strip().lower()
        if env_val in ("1", "true", "yes", "si"):
            return True
        if env_val in ("0", "false", "no"):
            return False

        # 2. Flag salvato in policy
        return bool(self.policy.get("current_consent_granted", False))

    def is_local_target(self, host_or_url: str) -> bool:
        """Determina se un host appartiene alla rete locale o interna (senza bisogno di internet)."""
        host = host_or_url.lower()
        if "://" in host:
            host = host.split("://")[1].split("/")[0].split(":")[0]

        for pattern in self.policy.get("allowed_local_networks", []):
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if host.startswith(prefix):
                    return True
            elif host == pattern:
                return True
        return False

    def request_consent(self, action_description: str = "Accesso alla rete Internet", platform: str = "Terminale / Visual Studio") -> Tuple[bool, str]:
        """Richiede o valida il consenso per effettuare un'operazione su rete esterna."""
        if self.is_internet_allowed():
            return True, "Consenso già accordato per la sessione corrente."

        # Se siamo in ambiente interattivo da terminale (TTY)
        if sys.stdin and sys.stdin.isatty():
            print("\n" + "=" * 65)
            print(" 🛡️ LUMINA SENTINEL — RICHIESTA DI CONSENSO ACCESSO A INTERNET")
            print("=" * 65)
            print(f"Piattaforma   : {platform}")
            print(f"Operazione    : {action_description}")
            print("Politica      : Di default Lumina opera al 100% offline e in locale.")
            print("=" * 65)
            try:
                ans = input("Vuoi consentire a Lumina l'accesso a Internet per questa operazione? (s/N): ").strip().lower()
                if ans in ("s", "si", "y", "yes"):
                    self.grant_consent(scope="session")
                    return True, "Consenso accordato dall'utente tramite terminale."
                else:
                    return False, "Accesso a Internet negato dall'utente. Operazione mantenuta in locale."
            except Exception:
                pass

        # In contesti non-TTY (es. chiamata API REST da Visual Studio o background)
        msg = (
            f"⚠️ ACCESSO INTERNET BLOCCATO DA LUMINA SENTINEL:\n"
            f"L'operazione '{action_description}' da {platform} richiede connessione esterna.\n"
            f"Per politica di sovranità locale, Lumina opera offline a meno di consenso esplicito.\n"
            f"Per autorizzare:\n"
            f"- Imposta la variabile d'ambiente: set LUMINA_ALLOW_INTERNET=1\n"
            f"- Oppure passa il comando con il flag '--allow-internet'\n"
            f"- Oppure aggiorna la policy in config/lumina_policy.json."
        )
        return False, msg

    def grant_consent(self, scope: str = "session"):
        """Accorda il consenso per la sessione o in modo permanente."""
        self.policy["current_consent_granted"] = True
        self.policy["consent_scope"] = scope
        if scope == "permanent":
            self.policy["default_internet_allowed"] = True
        self._save_policy(self.policy)

    def revoke_consent(self):
        """Revoca immediatamente il consenso e ripristina la modalità 100% offline."""
        self.policy["current_consent_granted"] = False
        self.policy["consent_scope"] = "none"
        self._save_policy(self.policy)
        if "LUMINA_ALLOW_INTERNET" in os.environ:
            os.environ["LUMINA_ALLOW_INTERNET"] = "0"


# Istanza singleton globale
_GLOBAL_POLICY = LuminaNetworkPolicy()

def get_network_policy() -> LuminaNetworkPolicy:
    return _GLOBAL_POLICY


if __name__ == "__main__":
    policy = get_network_policy()
    print("Stato Politica di Rete Lumina:")
    print(f"- Modalità                    : {policy.policy.get('internet_access_mode')}")
    print(f"- Accesso Internet Consentito : {policy.is_internet_allowed()}")
    print(f"- Integrazione Visual Studio  : {policy.policy.get('visual_studio_integration', {}).get('mode')}")
    print(f"- Telemetria Esterna          : {policy.policy.get('telemetry_policy', {}).get('allow_telemetry')}")
