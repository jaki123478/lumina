import tempfile
from pathlib import Path
from unittest.mock import patch

import lumina_self_edit as se


def test_rejects_unprotected_files():
    result = se.apply_candidate("serve.py", "bad", ["cmd", "/c", "exit", "0"])
    assert result["needs_confirmation"] is True


def test_applies_allowlisted_candidate_and_keeps_backup(tmp_path):
    target = tmp_path / "lumina_app.py"
    target.write_text("old", encoding="utf-8")
    with patch.object(se, "ROOT", tmp_path), patch.object(se, "WORKTREE", tmp_path / "workspace"), patch.object(se, "BACKUPS", tmp_path / "backups"), patch.object(se, "ALLOWLIST", {"lumina_app.py"}):
        result = se.apply_candidate("lumina_app.py", "new", ["cmd", "/c", "exit", "0"])
    assert result["applied"] is True
    assert target.read_text(encoding="utf-8") == "new"
    assert list((tmp_path / "backups").glob("*.bak"))


def test_rolls_back_failed_candidate(tmp_path):
    target = tmp_path / "lumina_app.py"
    target.write_text("old", encoding="utf-8")
    with patch.object(se, "ROOT", tmp_path), patch.object(se, "WORKTREE", tmp_path / "workspace"), patch.object(se, "BACKUPS", tmp_path / "backups"), patch.object(se, "ALLOWLIST", {"lumina_app.py"}):
        result = se.apply_candidate("lumina_app.py", "bad", ["cmd", "/c", "exit", "1"])
    assert result["rolled_back"] is True
    assert target.read_text(encoding="utf-8") == "old"
