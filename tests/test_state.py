"""State tests: atomic files, modes, mutes, pending TTL, cooldowns, locks."""

from __future__ import annotations

import json
import os
import stat
import unittest

from helpers import TempDirTest

from dafeiyu_state import (
    FILE_MODE,
    PENDING_TTL_MS,
    StateStore,
    append_log,
    read_json,
    write_json_atomic,
)


class Clock:
    def __init__(self, value=1_700_000_000_000):
        self.value = value

    def __call__(self):
        return self.value


class AtomicFileTest(TempDirTest):
    def test_write_read_and_modes(self):
        path = self.tmp / "nested" / "state.json"
        write_json_atomic(str(path), {"a": 1})
        self.assertEqual(read_json(str(path), None), {"a": 1})
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), FILE_MODE)
        self.assertEqual(stat.S_IMODE(os.stat(path.parent).st_mode), 0o700)
        # No temp files are left behind.
        self.assertEqual([p.name for p in path.parent.iterdir()], ["state.json"])

    def test_corrupt_and_missing_files_fall_back(self):
        path = self.write("bad.json", "{not json")
        self.assertEqual(read_json(str(path), {"fallback": True}), {"fallback": True})
        self.assertEqual(read_json(str(self.tmp / "absent.json"), []), [])
        self.assertEqual(read_json(str(self.write("empty.json", "")), "x"), "x")

    def test_overwrite_is_atomic(self):
        path = str(self.tmp / "s.json")
        write_json_atomic(path, {"n": 1})
        write_json_atomic(path, {"n": 2})
        self.assertEqual(read_json(path, None), {"n": 2})

    def test_append_log_creates_0600(self):
        log = self.tmp / "logs" / "x.log"
        append_log(str(log), "one")
        append_log(str(log), "two")
        self.assertEqual(log.read_text(), "one\ntwo\n")
        self.assertEqual(stat.S_IMODE(os.stat(log).st_mode), FILE_MODE)


class StoreTest(TempDirTest):
    def make(self):
        self.clock = Clock()
        return StateStore(str(self.tmp / "state"), debug=True, now=self.clock)

    def test_state_dir_mode(self):
        store = self.make()
        self.assertEqual(stat.S_IMODE(os.stat(store.state_dir).st_mode), 0o700)
        self.assertTrue(os.path.isdir(store.logs_dir))

    def test_mutes_are_scoped_and_expire(self):
        store = self.make()
        store.add_mute("scope-a", "topic:aaa", until_ms=self.clock.value + 1000, kind="topic", label="deploy")
        store.add_mute("scope-b", "topic:aaa", until_ms=self.clock.value + 1000, kind="topic")
        self.assertTrue(store.is_muted("scope-a", "topic:aaa"))
        self.assertFalse(store.is_muted("scope-c", "topic:aaa"))
        self.assertEqual(store.count_mutes("scope-a"), 1)
        self.assertEqual(store.count_mutes(), 2)
        self.assertEqual(store.list_mutes("scope-a")[0]["label"], "deploy")

        self.clock.value += 2000
        self.assertFalse(store.is_muted("scope-a", "topic:aaa"))
        self.assertEqual(store.count_mutes(), 0)
        # Expired records are pruned from listings but still removable.
        self.assertTrue(store.remove_mute("scope-a", "topic:aaa"))
        self.assertFalse(store.remove_mute("scope-a", "topic:aaa"))

    def test_pending_ttl_and_scope(self):
        store = self.make()
        store.record_pending("s1", ["92cfc0da", "b43746e5"], request_id="uuid-1", query_hash="h")
        latest = store.latest_pending("s1")
        self.assertEqual(latest["ids"], ["92cfc0da", "b43746e5"])
        self.assertEqual(latest["request_id"], "uuid-1")
        self.assertIsNone(store.latest_pending("s2"))

        self.clock.value += PENDING_TTL_MS + 1
        self.assertIsNone(store.latest_pending("s1"))
        self.assertEqual(store.pending_count("s1"), 0)

    def test_recent_pending_aggregates_several_recalls(self):
        store = self.make()
        store.record_pending(
            "s1",
            ["aaaaaaaa"],
            full_ids=["aaaaaaaa-1111-2222-3333-444444444444"],
            id_map={"aaaaaaaa": "aaaaaaaa-1111-2222-3333-444444444444"},
            ts=self.clock.value - 5000,
        )
        store.record_pending("s1", ["bbbbbbbb"], ts=self.clock.value - 1000)
        store.record_pending("s1", ["cccccccc"], ts=self.clock.value)
        store.record_pending("s2", ["dddddddd"], ts=self.clock.value)
        recent = store.recent_pending("s1")
        self.assertEqual([item["ids"] for item in recent], [["cccccccc"], ["bbbbbbbb"], ["aaaaaaaa"]])
        self.assertEqual(recent[-1]["id_map"]["aaaaaaaa"], "aaaaaaaa-1111-2222-3333-444444444444")
        self.assertEqual(len(store.recent_pending("s2")), 1)
        self.assertEqual(store.recent_pending("s3"), [])
        # latest_pending still returns only the newest snapshot.
        self.assertEqual(store.latest_pending("s1")["ids"], ["cccccccc"])

    def test_recent_pending_respects_ttl_and_bound(self):
        store = self.make()
        store.record_pending("s1", ["eeeeeeee"], ts=self.clock.value - PENDING_TTL_MS - 1)
        store.record_pending("s1", ["ffffffff"], ts=self.clock.value)
        self.assertEqual([item["ids"] for item in store.recent_pending("s1")], [["ffffffff"]])

        for index in range(60):
            store.record_pending("s1", [f"{index:08x}"], ts=self.clock.value)
        recent = store.recent_pending("s1")
        self.assertEqual(len(recent), 50)
        self.assertEqual(recent[0]["ids"], ["0000003b"])

    def test_recent_pending_is_read_only(self):
        store = self.make()
        store.record_pending("s1", ["aaaaaaaa"])
        before = (self.tmp / "state" / "pending.json").read_text(encoding="utf-8")
        store.recent_pending("s1", ttl_ms=PENDING_TTL_MS // 2)
        store.recent_pending("other")
        after = (self.tmp / "state" / "pending.json").read_text(encoding="utf-8")
        self.assertEqual(before, after)

    def test_take_pending_partial(self):
        store = self.make()
        store.record_pending(
            "s1",
            ["92cfc0da", "b43746e5"],
            full_ids=[
                "92cfc0da-1111-2222-3333-444444444444",
                "b43746e5-5555-6666-7777-888888888888",
            ],
            id_map={
                "92cfc0da": "92cfc0da-1111-2222-3333-444444444444",
                "b43746e5": "b43746e5-5555-6666-7777-888888888888",
            },
        )
        taken = store.take_pending("s1", ["92cfc0da"])
        self.assertEqual(taken["ids"], ["92cfc0da"])
        self.assertEqual(taken["id_map"]["92cfc0da"], "92cfc0da-1111-2222-3333-444444444444")
        remaining = store.latest_pending("s1")
        self.assertEqual(remaining["ids"], ["b43746e5"])
        self.assertEqual(remaining["full_ids"], ["b43746e5-5555-6666-7777-888888888888"])
        self.assertEqual(remaining["id_map"]["b43746e5"], "b43746e5-5555-6666-7777-888888888888")
        self.assertEqual(store.take_pending("s1", ["deadbeef"]), None)
        whole = store.take_pending("s1")
        self.assertEqual(whole["ids"], ["b43746e5"])
        self.assertIsNone(store.latest_pending("s1"))

    def test_pending_ring_is_bounded(self):
        store = self.make()
        for index in range(60):
            store.record_pending("s1", [f"{index:08x}"])
        self.assertEqual(store.pending_count(), 50)
        self.assertEqual(store.latest_pending("s1")["ids"], ["0000003b"])

    def test_cooldowns_and_clear_scope(self):
        store = self.make()
        store.set_cooldown("s1", last_inject_ms=111, query_hash="h1", query_ms=112)
        entry = store.get_cooldown("s1")
        self.assertEqual(entry["last_inject_ms"], 111)
        self.assertEqual(entry["query_hash"], "h1")
        store.set_cooldown("s1", query_ms=200)
        self.assertEqual(store.get_cooldown("s1")["last_inject_ms"], 111)

        store.set_topic_cooldown("s1", "topic:aaa", 999)
        self.assertEqual(store.get_topic_cooldown("s1", "topic:aaa"), 999)
        self.assertEqual(store.get_topic_cooldown("s2", "topic:aaa"), 0)

        store.record_pending("s1", ["92cfc0da"])
        store.record_pending("s2", ["b43746e5"])
        cleared = store.clear_scope("s1")
        self.assertEqual(cleared["pending"], 1)
        self.assertTrue(cleared["cooldown"])
        self.assertEqual(store.pending_count("s1"), 0)
        self.assertEqual(store.pending_count("s2"), 1)
        self.assertEqual(store.get_cooldown("s1"), {})
        self.assertEqual(store.get_topic_cooldown("s1", "topic:aaa"), 0)

    def test_jobs_record_merges(self):
        store = self.make()
        store.write_jobs({"daily": {"status": "ok"}})
        store.write_jobs({"last_train_ms": 5})
        jobs = store.read_jobs()
        self.assertEqual(jobs["daily"]["status"], "ok")
        self.assertEqual(jobs["last_train_ms"], 5)

    def test_lock_is_exclusive_and_releasable(self):
        store = self.make()
        release = store.acquire_job_lock()
        self.assertIsNotNone(release)
        self.assertIsNone(store.acquire_job_lock())
        release()
        release2 = store.acquire_job_lock()
        self.assertIsNotNone(release2)
        release2()

    def test_stale_lock_is_reclaimed(self):
        store = self.make()
        release = store.acquire_job_lock()
        self.assertIsNotNone(release)
        # Age the lock beyond the 25 h stale window.
        old = (self.clock.value - 26 * 60 * 60 * 1000) / 1000.0
        os.utime(store.files["job_lock"], (old, old))
        again = store.acquire_job_lock()
        self.assertIsNotNone(again)
        again()

    def test_selfnote_pointer_roundtrip_is_atomic(self):
        store = self.make()
        scope = "qq:abcd1234:selfnote"
        self.assertIsNone(store.read_selfnote(scope))
        store.write_selfnote(
            scope,
            {
                "id": "abcd1234-0000-0000-0000-000000000000",
                "brief": "summary",
                "updated_at": "2026-01-01T00:00:00Z",
                "sources": ["92cfc0da"],
                "ignored": "drop me",
            },
        )
        record = store.read_selfnote(scope)
        self.assertEqual(
            record,
            {
                "id": "abcd1234-0000-0000-0000-000000000000",
                "brief": "summary",
                "updated_at": "2026-01-01T00:00:00Z",
                "sources": ["92cfc0da"],
                "auto_turns": 0,
                "last_auto_at": "",
                "auto_failures": 0,
            },
        )
        self.assertEqual(stat.S_IMODE(os.stat(store.files["selfnote"]).st_mode), FILE_MODE)
        self.assertEqual([p.name for p in (self.tmp / "state").iterdir() if p.name.endswith(".tmp")], [])
        self.assertIsNone(store.read_selfnote("qq:other:selfnote"))

        self.assertTrue(store.clear_selfnote(scope))
        self.assertIsNone(store.read_selfnote(scope))
        self.assertFalse(store.clear_selfnote(scope))

    def test_selfnote_auto_counters_bump_and_mark(self):
        store = self.make()
        scope = "qq:abcd1234:selfnote"
        # A missing record is created with defaults on the first counted turn.
        record = store.bump_selfnote_turn(scope, ts=self.clock.value)
        self.assertEqual(record["auto_turns"], 1)
        self.assertEqual(record["auto_failures"], 0)
        self.assertEqual(record["last_auto_at"], "")
        self.assertEqual(record["id"], "")
        store.bump_selfnote_turn(scope)
        self.assertEqual(store.read_selfnote(scope)["auto_turns"], 2)

        # Success resets the counter and the failure count, stamps the time.
        record = store.mark_selfnote_auto(scope, ok=True, ts="2026-01-01T00:00:00Z")
        self.assertEqual(record["auto_turns"], 0)
        self.assertEqual(record["auto_failures"], 0)
        self.assertEqual(record["last_auto_at"], "2026-01-01T00:00:00Z")

        # Failure keeps the counter and increments the failure count.
        store.bump_selfnote_turn(scope)
        store.bump_selfnote_turn(scope)
        record = store.mark_selfnote_auto(scope, ok=False, ts="2026-01-01T01:00:00Z")
        self.assertEqual(record["auto_turns"], 2)
        self.assertEqual(record["auto_failures"], 1)
        self.assertEqual(record["last_auto_at"], "2026-01-01T01:00:00Z")
        record = store.mark_selfnote_auto(scope, ok=False, failures_delta=3)
        self.assertEqual(record["auto_failures"], 4)
        # A mark on an unknown scope creates a record instead of raising.
        fresh = store.mark_selfnote_auto("qq:other:selfnote", ok=False)
        self.assertEqual(fresh["auto_failures"], 1)

    def test_selfnote_write_preserves_auto_fields(self):
        store = self.make()
        scope = "qq:abcd1234:selfnote"
        store.bump_selfnote_turn(scope)
        store.mark_selfnote_auto(scope, ok=False, ts="2026-01-01T01:00:00Z")
        # A plain pointer write (as done by apply) must not reset progress.
        store.write_selfnote(
            scope,
            {"id": "abcd1234-0000-0000-0000-000000000000", "brief": "b", "updated_at": "t"},
        )
        record = store.read_selfnote(scope)
        self.assertEqual(record["auto_turns"], 1)
        self.assertEqual(record["auto_failures"], 1)
        self.assertEqual(record["last_auto_at"], "2026-01-01T01:00:00Z")
        # Explicit values still win.
        store.write_selfnote(scope, {"brief": "b", "auto_turns": 7, "auto_failures": 0, "last_auto_at": ""})
        record = store.read_selfnote(scope)
        self.assertEqual(record["auto_turns"], 7)
        self.assertEqual(record["auto_failures"], 0)
        self.assertEqual(record["last_auto_at"], "")

    def test_selfnote_legacy_record_reads_with_auto_defaults(self):
        store = self.make()
        scope = "qq:abcd1234:selfnote"
        # An old selfnote.json written before the auto fields existed.
        (self.tmp / "state" / "selfnote.json").write_text(
            json.dumps(
                {
                    "scopes": {
                        scope: {
                            "id": "abcd1234-0000-0000-0000-000000000000",
                            "brief": "旧摘要",
                            "updated_at": "2026-01-01T00:00:00Z",
                            "sources": ["92cfc0da"],
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        record = store.read_selfnote(scope)
        self.assertEqual(record["brief"], "旧摘要")
        self.assertEqual(record["auto_turns"], 0)
        self.assertEqual(record["last_auto_at"], "")
        self.assertEqual(record["auto_failures"], 0)
        bumped = store.bump_selfnote_turn(scope)
        self.assertEqual(bumped["auto_turns"], 1)
        self.assertEqual(bumped["brief"], "旧摘要")

    def test_selfnote_auto_writes_are_atomic_and_0600(self):
        store = self.make()
        scope = "qq:abcd1234:selfnote"
        store.bump_selfnote_turn(scope)
        store.mark_selfnote_auto(scope, ok=True)
        self.assertEqual(stat.S_IMODE(os.stat(store.files["selfnote"]).st_mode), FILE_MODE)
        self.assertEqual(
            [p.name for p in (self.tmp / "state").iterdir() if p.name.endswith(".tmp")],
            [],
        )
        self.assertEqual(json.loads((self.tmp / "state" / "selfnote.json").read_text())["scopes"][scope]["auto_turns"], 0)

    def test_debug_log_written_only_when_enabled(self):
        store = self.make()
        store.log("hello")
        self.assertIn("hello", (self.tmp / "state" / "logs" / "dafeiyu.log").read_text())
        quiet = StateStore(str(self.tmp / "quiet"), debug=False, now=self.clock)
        quiet.log("nope")
        self.assertFalse(os.path.exists(quiet.files["debug_log"]))

    def test_state_files_are_0600(self):
        store = self.make()
        store.add_mute("s", "topic:a", until_ms=self.clock.value + 10)
        store.record_pending("s", ["92cfc0da"])
        store.set_cooldown("s", last_inject_ms=1)
        store.write_jobs({"x": 1})
        for key in ("permissions", "pending", "cooldowns", "jobs"):
            mode = stat.S_IMODE(os.stat(store.files[key]).st_mode)
            self.assertEqual(mode, FILE_MODE, key)
        with open(store.files["pending"], encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(len(payload["items"]), 1)


if __name__ == "__main__":
    unittest.main()
