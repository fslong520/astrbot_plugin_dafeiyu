"""Shared test helpers (no AstrBot import anywhere in this directory)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


def make_fake_rsrs(directory, *, stdout="", stderr="", exit_code=0, sleep=0.0, payload=None):
    """Write an executable shell script that behaves like the rsrs CLI."""
    body = ["#!/bin/sh"]
    if sleep:
        body.append(f"sleep {sleep}")
    if stderr:
        body.append(f"printf '%s' {json.dumps(stderr)} 1>&2")
    if payload is not None:
        body.append(f"printf '%s' {json.dumps(json.dumps(payload))}")
    elif stdout:
        body.append(f"printf '%s' {json.dumps(stdout)}")
    body.append(f"exit {exit_code}")
    path = os.path.join(str(directory), "rsrs")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(body) + "\n")
    os.chmod(path, 0o755)
    return path


class TempDirTest(unittest.TestCase):
    """Base class giving every test an isolated temporary directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, content, mode=0o644):
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        os.chmod(path, mode)
        return path
