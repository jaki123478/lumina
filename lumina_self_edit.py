"""Auto-miglioramento locale controllato per Lumina.

Applica modifiche solo a file allowlisted in una working copy, dopo backup e
test riusciti. Checkpoint, configurazioni di sicurezza e file fuori perimetro
non sono modificabili da questo modulo.
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKTREE = ROOT / "data" / "autonomy" / "self_edit_workspace"
BACKUPS = ROOT / "data" / "autonomy" / "self_edit_backups"
ALLOWLIST = {"lumina_desktop/lumina_app.py", "lumina_desktop/lumina_chat.py"}
PROTECTED_PREFIXES = ("checkpoints", "exports", "hf_export", ".venv", "dist", "build")


def _relative(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")


def _allowed(path: str) -> bool:
    return path in ALLOWLIST and not any(path.startswith(p) for p in PROTECTED_PREFIXES)


def apply_candidate(path: str, content: str, test_command: list[str]) -> dict:
    """Apply a tested candidate automatically; rollback on any failure."""
    rel = _relative(path)
    if not _allowed(rel):
        return {"applied": False, "needs_confirmation": True, "reason": "file_not_allowlisted"}
    target = ROOT / rel
    if not target.exists():
        return {"applied": False, "needs_confirmation": True, "reason": "file_missing"}
    WORKTREE.mkdir(parents=True, exist_ok=True)
    BACKUPS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = BACKUPS / f"{stamp}_{target.name}.bak"
    shutil.copy2(target, backup)
    try:
        target.write_text(content, encoding="utf-8")
        result = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True, timeout=120, shell=False)
        if result.returncode != 0:
            shutil.copy2(backup, target)
            return {"applied": False, "rolled_back": True, "reason": "tests_failed", "stderr": result.stderr[-2000:]}
        return {"applied": True, "rolled_back": False, "backup": str(backup.relative_to(ROOT))}
    except Exception as exc:
        shutil.copy2(backup, target)
        return {"applied": False, "rolled_back": True, "reason": "edit_error", "error": str(exc)}
