"""Granite / encoder detection for the AstrBot adapter.

The plugin never loads the model itself. It only checks whether the IBM Granite
Embedding 97M R2 checkpoint and the respire weekly encoder script exist, and
reports that in ``/mem_status`` and to the job scheduler.
"""

from __future__ import annotations

import json
import os

#: Entries that must exist inside the Granite model directory.
GRANITE_REQUIRED = ("model.safetensors", "config.json", "1_Pooling")

#: Path of the weekly encoder script inside the respire repo.
ENCODER_SCRIPT_REL = os.path.join("scripts", "encoder", "weekly_encoder.sh")


def _is_file(path):
    try:
        return os.path.isfile(path)
    except OSError:
        return False


def _is_dir(path):
    try:
        return os.path.isdir(path)
    except OSError:
        return False


def detect_granite(model_dir):
    """Detect the Granite checkpoint.

    ``1_Pooling`` may be a directory (sentence-transformers layout) or a file, so
    both are accepted.
    """
    directory = model_dir if isinstance(model_dir, str) else ""
    missing = []
    found = []
    for entry in GRANITE_REQUIRED:
        target = os.path.join(directory, entry) if directory else ""
        present = (_is_dir(target) or _is_file(target)) if entry == "1_Pooling" else _is_file(target)
        (found if present else missing).append(entry)
    return {"model_dir": directory, "installed": not missing, "missing": missing, "found": found}


def detect_encoder_script(repo_dir):
    """Detect the weekly encoder script inside a respire checkout."""
    repo = repo_dir if isinstance(repo_dir, str) else ""
    script = os.path.join(repo, ENCODER_SCRIPT_REL) if repo else ""
    return {"repo": repo, "script": script, "exists": bool(script) and _is_file(script)}


def encoder_status(granite_model_dir=None, respire_repo=None):
    """Combined encoder readiness used by ``/mem_status`` and the weekly job."""
    model = detect_granite(granite_model_dir)
    script = detect_encoder_script(respire_repo)
    reason = None
    if not model["installed"]:
        reason = "granite model incomplete (missing {})".format(", ".join(model["missing"]))
    elif not script["exists"]:
        reason = "weekly encoder script not found"
    out = {"model": model, "script": script, "ready": model["installed"] and script["exists"]}
    if reason:
        out["reason"] = reason
    return out


def read_index_meta(anchor_dir):
    """Read ``<anchor>/index/current/meta.json`` for version + metrics.

    Returns ``{"version": str, "docs": int, "dim": int, "metrics": dict}`` with
    ``None`` fields when the index is absent or unreadable. Never raises.
    """
    out = {"version": None, "docs": None, "dim": None, "metrics": {}, "path": None}
    if not isinstance(anchor_dir, str) or anchor_dir == "":
        return out
    meta_path = os.path.join(anchor_dir, "index", "current", "meta.json")
    out["path"] = meta_path
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return out
    if not isinstance(meta, dict):
        return out
    out["version"] = meta.get("version") if isinstance(meta.get("version"), str) else None
    out["docs"] = meta.get("docs") if isinstance(meta.get("docs"), int) else None
    out["dim"] = meta.get("dim") if isinstance(meta.get("dim"), int) else None
    test = meta.get("test")
    if isinstance(test, dict):
        out["metrics"] = {
            key: value for key, value in test.items() if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    return out
