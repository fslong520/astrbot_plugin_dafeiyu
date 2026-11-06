"""Granite detection and encoder script tests."""

from __future__ import annotations

import json
import os
import unittest

from helpers import TempDirTest

from dafeiyu_granite import (
    ENCODER_SCRIPT_REL,
    detect_encoder_script,
    detect_granite,
    encoder_status,
    read_index_meta,
)


class GraniteTest(TempDirTest):
    def make_model(self, pooling="dir", complete=True, name="granite-97m-r2"):
        model = self.tmp / "models" / name
        model.mkdir(parents=True, exist_ok=True)
        if complete:
            (model / "model.safetensors").write_text("weights")
            (model / "config.json").write_text("{}")
        if pooling == "dir":
            (model / "1_Pooling").mkdir()
        elif pooling == "file":
            (model / "1_Pooling").write_text("x")
        return model

    def test_complete_model_with_directory_pooling(self):
        model = self.make_model()
        status = detect_granite(str(model))
        self.assertTrue(status["installed"])
        self.assertEqual(status["missing"], [])
        self.assertEqual(sorted(status["found"]), ["1_Pooling", "config.json", "model.safetensors"])

    def test_pooling_may_be_a_file(self):
        model = self.make_model(pooling="file")
        self.assertTrue(detect_granite(str(model))["installed"])

    def test_missing_entries_are_listed(self):
        model = self.make_model(pooling="dir", complete=False)
        status = detect_granite(str(model))
        self.assertFalse(status["installed"])
        self.assertEqual(sorted(status["missing"]), ["config.json", "model.safetensors"])

    def test_empty_and_missing_directories(self):
        self.assertFalse(detect_granite("")["installed"])
        self.assertEqual(len(detect_granite("")["missing"]), 3)
        self.assertFalse(detect_granite(str(self.tmp / "nope"))["installed"])

    def test_encoder_script_detection(self):
        repo = self.tmp / "repo"
        script = repo / ENCODER_SCRIPT_REL
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/sh\nexit 0\n")
        found = detect_encoder_script(str(repo))
        self.assertTrue(found["exists"])
        self.assertEqual(found["script"], str(script))
        self.assertFalse(detect_encoder_script("")["exists"])
        self.assertFalse(detect_encoder_script(str(self.tmp / "other"))["exists"])

    def test_encoder_status_reasons(self):
        model = self.make_model()
        repo = self.tmp / "repo"
        status = encoder_status(granite_model_dir=str(model), respire_repo=str(repo))
        self.assertFalse(status["ready"])
        self.assertEqual(status["reason"], "weekly encoder script not found")

        script = repo / ENCODER_SCRIPT_REL
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/sh\n")
        self.assertTrue(encoder_status(str(model), str(repo))["ready"])

        incomplete = self.make_model(pooling="dir", complete=False, name="granite-broken")
        broken = encoder_status(str(incomplete), str(repo))
        self.assertFalse(broken["ready"])
        self.assertIn("granite model incomplete", broken["reason"])

    def test_index_meta(self):
        anchor = self.tmp / "encoder"
        index = anchor / "index" / "current"
        index.mkdir(parents=True)
        meta = {
            "version": "20261102-164914",
            "docs": 4862,
            "dim": 384,
            "test": {"n": 191, "hit20": 0.7644, "mrr20": 0.34, "note": "ignored"},
        }
        (index / "meta.json").write_text(json.dumps(meta))
        read = read_index_meta(str(anchor))
        self.assertEqual(read["version"], "20261102-164914")
        self.assertEqual(read["docs"], 4862)
        self.assertEqual(read["dim"], 384)
        self.assertEqual(read["metrics"], {"n": 191, "hit20": 0.7644, "mrr20": 0.34})

    def test_index_meta_missing_or_corrupt(self):
        absent = read_index_meta(str(self.tmp / "absent"))
        self.assertIsNone(absent["version"])
        self.assertIsNone(read_index_meta("")["version"])
        anchor = self.tmp / "bad"
        index = anchor / "index" / "current"
        index.mkdir(parents=True)
        (index / "meta.json").write_text("{not json")
        self.assertIsNone(read_index_meta(str(anchor))["version"])


if __name__ == "__main__":
    unittest.main()
