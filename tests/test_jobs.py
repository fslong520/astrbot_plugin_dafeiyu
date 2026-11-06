"""Jobs tests: nightly read-only maintenance, records and scheduler gating.

The maintenance pass must call ``audit`` then ``review --due``, never mutate a
memory, never raise, and keep running when ``jobs.enabled`` is false.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest

from helpers import TempDirTest

import dafeiyu_jobs
from dafeiyu_jobs import (
    MAINTENANCE_INTERVAL_MS,
    Jobs,
    summarize_audit,
    summarize_review,
)
from dafeiyu_rsrs import Rsrs
from dafeiyu_state import StateStore

AUDIT_PAYLOAD = {
    "command": "audit",
    "status": "warn",
    # The real CLI keeps the verdict at the top level and counters in summary.
    "summary": {
        "total_active": 4911,
        "orphans": 0,
        "invalid_importance": 0,
        "same_title_groups": 0,
        "truncated_titles": 0,
        "max_depth": 21,
        "hot_untyped": 870,
        "failing_skills": 0,
        "skill_template_gaps": 620,
        "unknown_key": 1,
    },
    "items": [],
    "details": {
        "hot_untyped": [{"id": "deadbeef", "title": "DETAIL_TITLE_SHOULD_NOT_LEAK"}],
    },
    "errors": [],
}

REVIEW_PAYLOAD = {
    "command": "review",
    "status": "ok",
    "summary": {"status": "due", "due": 1, "title": "SUMMARY_TITLE_SHOULD_NOT_LEAK"},
    "items": [{"name": "92cfc0da", "value": "first remembered title"}],
    "details": [{"id": "92cfc0da-1111-2222-3333-444444444444", "title": "first remembered title"}],
    "errors": [],
}

REVIEW_ERROR_PAYLOAD = {
    "command": "review",
    "status": "error",
    "summary": {"status": "error"},
    "items": [],
    "errors": [{"message": "corrupt snapshot: due list unreadable"}],
}

AUDIT_ERROR_PAYLOAD = {
    "command": "audit",
    "status": "error",
    "summary": {"status": "error"},
    "items": [],
    "errors": [{"message": "audit index unreadable"}],
}

#: ``list --metadata`` rows: the P0-1 checks (clock = 2023-11-14T22:13:20Z).
LIST_META_PAYLOAD = {
    "command": "list",
    "status": "ok",
    "summary": {"count": 6, "limit": 10000},
    "items": [],
    "details": [
        {"id": "aaaa1111-1111-1111-1111-111111111111", "kind": "Hypothesis",
         "title": "HYP_TITLE_SHOULD_NOT_LEAK"},
        {"id": "bbbb2222-2222-2222-2222-222222222222", "kind": "Context",
         "parent_id": "aaaa1111-1111-1111-1111-111111111111",
         "title": "FACT_TITLE_SHOULD_NOT_LEAK"},
        {"id": "cccc3333-3333-3333-3333-333333333333", "kind": "Context",
         "valid_until": "2023-11-01T00:00:00.000Z", "title": "EXPIRED_TITLE_SHOULD_NOT_LEAK"},
        {"id": "dddd4444-4444-4444-4444-444444444444", "kind": "Context",
         "review_at": "2023-11-10T00:00:00.000Z", "title": "DUE_TITLE_SHOULD_NOT_LEAK"},
        {"id": "eeee5555-5555-5555-5555-555555555555", "kind": "Context",
         "superseded_by": "ffff6666-6666-6666-6666-666666666666", "title": "OLD_CAUSE"},
        {"id": "99988877-7777-7777-7777-777777777777", "kind": "Context",
         "parent_id": "eeee5555-5555-5555-5555-555555555555", "title": "STANDS_ON_OLD_CAUSE"},
    ],
    "errors": [],
}

#: ``list`` body rows: the P2-2 source check (window cutoff 2023-10-31).
LIST_BODY_PAYLOAD = {
    "command": "list",
    "status": "ok",
    "summary": {"count": 4, "limit": 10000},
    "items": [],
    "details": [
        {"id": "aaaa1111-1111-1111-1111-111111111111", "kind": "Decision",
         "importance": "important", "created_at": "2023-11-12T00:00:00.000Z",
         "content": "改 src/main.py 的 bug", "title": "BODY_OK_TITLE"},
        {"id": "bbbb2222-2222-2222-2222-222222222222", "kind": "Decision",
         "importance": "important", "created_at": "2023-11-11T00:00:00.000Z",
         "content": "决定用方案 B", "title": "BODY_MISSING_SHOULD_NOT_LEAK"},
        {"id": "cccc3333-3333-3333-3333-333333333333", "kind": "Decision",
         "importance": "important", "created_at": "2023-09-01T00:00:00.000Z",
         "content": "窗口外的旧决定", "title": "OLD_DECISION"},
        {"id": "dddd4444-4444-4444-4444-444444444444", "kind": "Decision",
         "importance": "normal", "created_at": "2023-11-10T00:00:00.000Z",
         "content": "非重要记录", "title": "MINOR_DECISION"},
    ],
    "errors": [],
}

LIST_ERROR_PAYLOAD = {
    "command": "list",
    "status": "error",
    "summary": {"status": "error"},
    "items": [],
    "errors": [{"message": "list index unreadable"}],
}


class Clock:
    def __init__(self, value=1_700_000_000_000):
        self.value = value

    def __call__(self):
        return self.value


class Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, *args, **kwargs):
        self.warnings.append(" ".join(str(item) for item in args))


def make_binary(directory, *, audit_error=False, review_error=False, spotcheck_error=False):
    """Fake CLI that logs each argv line and answers audit/review/list."""
    log = directory / "calls.log"
    audit = json.dumps(json.dumps(AUDIT_ERROR_PAYLOAD if audit_error else AUDIT_PAYLOAD))
    review = json.dumps(json.dumps(REVIEW_ERROR_PAYLOAD if review_error else REVIEW_PAYLOAD))
    meta = json.dumps(json.dumps(LIST_ERROR_PAYLOAD if spotcheck_error else LIST_META_PAYLOAD))
    body = json.dumps(json.dumps(LIST_ERROR_PAYLOAD if spotcheck_error else LIST_BODY_PAYLOAD))
    script = (
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {json.dumps(str(log))}\n"
        'case "$1" in\n'
        f"  audit) printf '%s' {audit} ;;\n"
        f"  review) printf '%s' {review} ;;\n"
        "  list) case \"$*\" in\n"
        f"        *--metadata*) printf '%s' {meta} ;;\n"
        f"        *) printf '%s' {body} ;;\n"
        "        esac ;;\n"
        """  *) printf '%s' '{"status":"ok"}' ;;\n"""
        "esac\n"
        "exit 0\n"
    )
    path = directory / "rsrs"
    path.write_text(script, encoding="utf-8")
    os.chmod(path, 0o755)
    return path, log


class JobsTest(TempDirTest):
    def setUp(self):
        super().setUp()
        self.clock = Clock()
        self.logger = Logger()

    def make(self, *, enabled=False, maintenance_enabled=True, review_limit=20, **binary_options):
        binary, log = make_binary(self.tmp, **binary_options)
        self.store = StateStore(str(self.tmp / "state"), now=self.clock)
        rsrs = Rsrs({"rsrs_bin": str(binary), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        config = {
            "jobs": {
                "enabled": enabled,
                "maintenance_enabled": maintenance_enabled,
                "maintenance_hour": 4,
                "maintenance_minute": 0,
                "review_limit": review_limit,
                "daily_limit": 10,
                "respire_repo": str(self.tmp / "no-repo"),
                "granite_model_dir": str(self.tmp / "no-model"),
                "anchor_dir": str(self.tmp / "no-anchor"),
            }
        }
        self.jobs = Jobs(config, rsrs, self.store, logger=self.logger, now=self.clock)
        self.log = log
        return self.jobs

    def calls(self):
        if not self.log.exists():
            return []
        return [line for line in self.log.read_text(encoding="utf-8").splitlines() if line]

    def command_names(self):
        return [line.split()[0] for line in self.calls()]

    # -- maintenance -------------------------------------------------------
    def test_runs_audit_then_review_and_stores_counts_only(self):
        jobs = self.make()
        out = asyncio.run(jobs.run_maintenance(force=True))
        self.assertTrue(out["ran"])
        self.assertEqual(self.command_names(), ["audit", "review", "list", "list"])
        self.assertEqual(self.calls()[0], "audit --json")
        self.assertEqual(self.calls()[1], "review --due --json --limit 20")
        self.assertEqual(self.calls()[2], "list --metadata --limit 10000 --json")
        self.assertEqual(self.calls()[3], "list --limit 10000 --json")

        record = self.store.read_jobs()["maintenance"]
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["audit"]["status"], "warn")
        self.assertEqual(record["audit"]["counts"]["total_active"], 4911)
        self.assertEqual(record["audit"]["counts"]["max_depth"], 21)
        self.assertNotIn("unknown_key", record["audit"]["counts"])
        self.assertEqual(record["review"]["status"], "due")
        self.assertEqual(record["review"]["counts"]["items"], 1)

        # Nightly spot-check: the plan's rule-layer audit, counts only.
        self.assertEqual(record["spotcheck"]["status"], "ok")
        counts = record["spotcheck"]["counts"]
        self.assertEqual(counts["checked"], 6)
        self.assertEqual(counts["unverified"], 1)
        self.assertEqual(counts["reflection_as_evidence"], 1)
        self.assertEqual(counts["expired"], 1)
        self.assertEqual(counts["review_due"], 1)
        self.assertEqual(counts["derived_from_superseded"], 1)
        self.assertEqual(counts["source_checked"], 2)
        self.assertEqual(counts["source_missing"], 1)

        serialized = json.dumps(record)
        self.assertNotIn("DETAIL_TITLE_SHOULD_NOT_LEAK", serialized)
        self.assertNotIn("SUMMARY_TITLE_SHOULD_NOT_LEAK", serialized)
        self.assertNotIn("first remembered title", serialized)
        self.assertNotIn("deadbeef", serialized)
        for marker in (
            "HYP_TITLE_SHOULD_NOT_LEAK",
            "FACT_TITLE_SHOULD_NOT_LEAK",
            "EXPIRED_TITLE_SHOULD_NOT_LEAK",
            "DUE_TITLE_SHOULD_NOT_LEAK",
            "BODY_MISSING_SHOULD_NOT_LEAK",
            "改 src/main.py",
        ):
            self.assertNotIn(marker, serialized)

    def test_review_failure_still_records_the_audit(self):
        jobs = self.make(review_error=True)
        out = asyncio.run(jobs.run_maintenance(force=True))
        self.assertTrue(out["ran"])
        self.assertEqual(out["status"], "partial")
        record = self.store.read_jobs()["maintenance"]
        self.assertEqual(record["audit"]["status"], "warn")
        self.assertEqual(record["audit"]["counts"]["total_active"], 4911)
        self.assertEqual(record["review"]["status"], "error")
        self.assertIn("corrupt snapshot", record["review"]["error"])

    def test_audit_failure_does_not_block_the_review(self):
        jobs = self.make(audit_error=True)
        out = asyncio.run(jobs.run_maintenance(force=True))
        self.assertTrue(out["ran"])
        self.assertEqual(self.command_names(), ["audit", "review", "list", "list"])
        record = self.store.read_jobs()["maintenance"]
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["audit"]["status"], "error")
        self.assertIn("audit index unreadable", record["audit"]["error"])
        self.assertEqual(record["review"]["status"], "due")
        self.assertEqual(record["spotcheck"]["status"], "ok")

    def test_spotcheck_failure_still_records_audit_and_review(self):
        jobs = self.make(spotcheck_error=True)
        out = asyncio.run(jobs.run_maintenance(force=True))
        self.assertTrue(out["ran"])
        self.assertEqual(out["status"], "partial")
        record = self.store.read_jobs()["maintenance"]
        self.assertEqual(record["audit"]["status"], "warn")
        self.assertEqual(record["review"]["status"], "due")
        self.assertEqual(record["spotcheck"]["status"], "error")
        self.assertIn("list index unreadable", record["spotcheck"]["error"])
        self.assertEqual(record["spotcheck"]["counts"], {})
        # The metadata scan failed first, so only one list call went out.
        self.assertEqual(self.command_names(), ["audit", "review", "list"])

    def test_maintenance_is_at_most_daily(self):
        jobs = self.make()
        asyncio.run(jobs.run_maintenance(force=True))
        self.clock.value += 60 * 1000
        out = asyncio.run(jobs.run_maintenance())
        self.assertEqual(out["reason"], "not-due")
        self.assertEqual(self.command_names(), ["audit", "review", "list", "list"])
        self.clock.value += MAINTENANCE_INTERVAL_MS
        out = asyncio.run(jobs.run_maintenance())
        self.assertTrue(out["ran"])
        self.assertEqual(
            self.command_names(),
            ["audit", "review", "list", "list", "audit", "review", "list", "list"],
        )

    def test_lock_held_skips_without_calling_the_cli(self):
        jobs = self.make()
        release = self.store.acquire_job_lock()
        try:
            out = asyncio.run(jobs.run_maintenance(force=True))
        finally:
            release()
        self.assertEqual(out["reason"], "locked")
        self.assertEqual(self.calls(), [])

    def test_review_limit_is_clamped_by_the_argv_builder(self):
        jobs = self.make(review_limit=1000)
        asyncio.run(jobs.run_maintenance(force=True))
        self.assertEqual(self.calls()[1], "review --due --json --limit 100")

    def test_maintenance_never_uses_a_mutating_command(self):
        jobs = self.make()
        asyncio.run(jobs.run_maintenance(force=True))
        for line in self.calls():
            for word in ("forget", "purge", "delete", "update", "--force"):
                self.assertNotIn(word, line)

    # -- summaries ---------------------------------------------------------
    def test_summarize_audit_whitelists_scalars(self):
        counts = summarize_audit(AUDIT_PAYLOAD)
        self.assertEqual(counts["status"], "warn")
        self.assertEqual(counts["hot_untyped"], 870)
        self.assertEqual(counts["skill_template_gaps"], 620)
        self.assertNotIn("unknown_key", counts)
        self.assertNotIn("DETAIL_TITLE_SHOULD_NOT_LEAK", json.dumps(counts))

    def test_summarize_review_keeps_no_titles(self):
        counts = summarize_review(REVIEW_PAYLOAD)
        self.assertEqual(counts["status"], "due")
        self.assertEqual(counts["items"], 1)
        self.assertEqual(counts["due"], 1)
        self.assertNotIn("title", counts)

    def test_summarize_review_reads_the_top_level_status(self):
        # Shape of the real failure: errors + top-level "fail", summary {"saved": false}.
        counts = summarize_review(
            {
                "command": "review",
                "status": "fail",
                "summary": {"saved": False},
                "items": [],
                "errors": ["decode review snapshot: nonce length must be 12 bytes"],
            }
        )
        self.assertEqual(counts["status"], "fail")
        self.assertEqual(counts["items"], 0)
        self.assertEqual(counts["saved"], False)

    # -- scheduler ---------------------------------------------------------
    def _spin(self, iterations=5):
        original = dafeiyu_jobs.asyncio.sleep
        count = {"n": 0}

        async def fast_sleep(_delay, *args, **kwargs):
            count["n"] += 1
            if count["n"] > iterations:
                raise asyncio.CancelledError()
            return None

        async def go():
            dafeiyu_jobs.asyncio.sleep = fast_sleep
            try:
                task = asyncio.ensure_future(self.jobs._loop())
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            finally:
                dafeiyu_jobs.asyncio.sleep = original

        asyncio.run(go())

    def test_loop_runs_maintenance_but_not_training_when_disabled(self):
        self.make(enabled=False, maintenance_enabled=True)
        calls = []

        async def maintenance(force=False):
            calls.append("maintenance")
            return {"ran": True}

        async def daily(force=False):
            calls.append("daily")
            return {"ran": True}

        async def weekly():
            calls.append("weekly")
            return {"ran": True}

        self.jobs.run_maintenance = maintenance
        self.jobs.run_daily_learn = daily
        self.jobs.run_weekly_encoder = weekly
        # All three deadlines are already in the past.
        self.jobs._initial_deadlines = lambda: (1, 2, 3)
        self._spin()
        self.assertEqual(calls, ["maintenance"])

    def test_loop_runs_every_schedule_when_enabled(self):
        self.make(enabled=True, maintenance_enabled=True)
        calls = []

        async def maintenance(force=False):
            calls.append("maintenance")
            return {"ran": True}

        async def daily(force=False):
            calls.append("daily")
            return {"ran": True}

        async def weekly():
            calls.append("weekly")
            return {"ran": True}

        self.jobs.run_maintenance = maintenance
        self.jobs.run_daily_learn = daily
        self.jobs.run_weekly_encoder = weekly
        self.jobs._initial_deadlines = lambda: (1, 2, 3)
        self._spin()
        self.assertEqual(calls, ["maintenance", "weekly", "daily"])

    def test_no_maintenance_deadline_when_switched_off(self):
        jobs = self.make(enabled=True, maintenance_enabled=False)
        daily_at, weekly_at, maintenance_at = jobs._initial_deadlines()
        self.assertIsInstance(daily_at, int)
        self.assertIsInstance(weekly_at, int)
        self.assertIsNone(maintenance_at)


if __name__ == "__main__":
    unittest.main()
