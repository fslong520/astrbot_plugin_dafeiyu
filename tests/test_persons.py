"""Person profile tests: persons.json, nickname history, relations, admin."""

from __future__ import annotations

import json
import os
import stat
import unittest

from helpers import TempDirTest

from dafeiyu_state import (
    FILE_MODE,
    NICKNAMES_MAX,
    PERSON_RELATIONS,
    StateStore,
    person_hash,
    person_is_admin,
)


class Clock:
    def __init__(self, value=1_700_000_000_000):
        self.value = value

    def __call__(self):
        return self.value


class PersonHashTest(unittest.TestCase):
    def test_hash_is_sha256_prefix_16_and_stable(self):
        self.assertEqual(person_hash("123456"), "8d969eef6ecad3c2")
        self.assertEqual(len(person_hash("anything")), 16)
        # int and str forms of the same id hash identically.
        self.assertEqual(person_hash(123456), person_hash("123456"))

    def test_hash_does_not_contain_raw_id(self):
        raw = "10001"
        hashed = person_hash(raw)
        self.assertNotIn(raw, hashed)

    def test_admin_detection(self):
        self.assertTrue(person_is_admin("123456", ["123456"]))
        self.assertTrue(person_is_admin(123456, ["123456"]))
        self.assertTrue(person_is_admin("123456", "123456, 789"))
        self.assertTrue(person_is_admin("789", ("789",)))
        self.assertFalse(person_is_admin("555", ["123456"]))
        self.assertFalse(person_is_admin("123456", None))
        self.assertFalse(person_is_admin("123456", []))
        # normalize_admin_ids produces strings; an empty config fails closed.
        self.assertFalse(person_is_admin("123456", ()))


class PersonStoreTest(TempDirTest):
    def make(self):
        self.clock = Clock()
        return StateStore(str(self.tmp / "state"), debug=False, now=self.clock)

    def test_upsert_creates_defaults(self):
        store = self.make()
        key = person_hash("10001")
        record = store.upsert_person(key, "张三", now=self.clock.value)
        self.assertEqual(
            record,
            {
                "current_nickname": "张三",
                "nicknames": ["张三"],
                "first_seen": str(self.clock.value),
                "last_seen": str(self.clock.value),
                "relation": "stranger",
                "note": "",
                "groups": [],
                "updated_at": str(self.clock.value),
            },
        )
        # File structure and field set are exactly as specified.
        raw = json.loads((self.tmp / "state" / "persons.json").read_text(encoding="utf-8"))
        self.assertEqual(list(raw.keys()), [key])
        self.assertEqual(
            sorted(raw[key].keys()),
            [
                "current_nickname",
                "first_seen",
                "groups",
                "last_seen",
                "nicknames",
                "note",
                "relation",
                "updated_at",
            ],
        )
        self.assertEqual(stat.S_IMODE(os.stat(store.files["persons"]).st_mode), FILE_MODE)
        # No temp files left behind.
        self.assertEqual([p.name for p in (self.tmp / "state").iterdir() if p.name.endswith(".tmp")], [])

    def test_upsert_is_idempotent_for_same_nickname(self):
        store = self.make()
        key = person_hash("10001")
        first = store.upsert_person(key, "张三", now=self.clock.value)
        self.clock.value += 5000
        second = store.upsert_person(key, "张三", now=self.clock.value)
        self.assertEqual(second["nicknames"], ["张三"])
        self.assertEqual(second["first_seen"], first["first_seen"])
        self.assertEqual(second["last_seen"], str(self.clock.value))
        self.assertEqual(second["updated_at"], str(self.clock.value))
        # Exactly one record on disk.
        raw = json.loads((self.tmp / "state" / "persons.json").read_text(encoding="utf-8"))
        self.assertEqual(len(raw), 1)

    def test_nickname_change_goes_to_history(self):
        store = self.make()
        key = person_hash("10001")
        store.upsert_person(key, "张三", now=self.clock.value)
        self.clock.value += 1000
        store.upsert_person(key, "小三", now=self.clock.value)
        record = store.read_person(key)
        self.assertEqual(record["current_nickname"], "小三")
        self.assertEqual(record["nicknames"], ["张三", "小三"])
        self.assertEqual(record["first_seen"], "1700000000000")
        # A third change appends again.
        self.clock.value += 1000
        store.upsert_person(key, "老三", now=self.clock.value)
        self.assertEqual(store.read_person(key)["nicknames"], ["张三", "小三", "老三"])

    def test_nickname_history_is_bounded(self):
        store = self.make()
        key = person_hash("10001")
        for index in range(NICKNAMES_MAX + 5):
            store.upsert_person(key, f"昵称{index}", now=self.clock.value + index)
        record = store.read_person(key)
        self.assertEqual(len(record["nicknames"]), NICKNAMES_MAX)
        self.assertEqual(record["nicknames"][-1], f"昵称{NICKNAMES_MAX + 4}")

    def test_default_relation_is_stranger(self):
        store = self.make()
        for relation in PERSON_RELATIONS:
            self.assertIn(relation, ("stranger", "friend", "master", "other_fish"))
        record = store.upsert_person(person_hash("20002"), "路人", now=self.clock.value)
        self.assertEqual(record["relation"], "stranger")
        # Non-admin upserts never promote.
        record = store.upsert_person(person_hash("20002"), "路人", now=self.clock.value, admin=False)
        self.assertEqual(record["relation"], "stranger")

    def test_admin_is_auto_marked_master(self):
        store = self.make()
        key = person_hash("10001")
        record = store.upsert_person(key, "主人", now=self.clock.value, admin=True)
        self.assertEqual(record["relation"], "master")
        # Survives later non-admin sightings.
        self.clock.value += 1000
        record = store.upsert_person(key, "主人", now=self.clock.value, admin=False)
        self.assertEqual(record["relation"], "master")
        # person_is_admin drives the flag the caller passes.
        admins = ["10001"]
        self.assertTrue(person_is_admin("10001", admins))
        promoted = store.upsert_person(
            person_hash("10001"), "主人", now=self.clock.value, admin=person_is_admin("10001", admins)
        )
        self.assertEqual(promoted["relation"], "master")

    def test_manual_relation_wins_over_auto_master(self):
        store = self.make()
        key = person_hash("10001")
        store.upsert_person(key, "主人", now=self.clock.value, admin=True)
        store.set_person_relation(key, "friend", now=self.clock.value, set_by="admin-command")
        record = store.read_person(key)
        self.assertEqual(record["relation"], "friend")
        self.assertEqual(record["set_by"], "admin-command")
        # Every later admin upsert must leave the human decision alone. The
        # plugin upserts on each message, so one pass is not enough evidence:
        # set_by used to be dropped by _person_entry, which only showed up on
        # the SECOND subsequent upsert.
        for _ in range(3):
            self.clock.value += 1000
            record = store.upsert_person(key, "主人", now=self.clock.value, admin=True)
            self.assertEqual(record["relation"], "friend")
            self.assertEqual(record["set_by"], "admin-command")
        # Nickname history is preserved across the relation change.
        self.assertIn("主人", record["nicknames"])

    def test_manual_relation_survives_repeated_upserts(self):
        """Guard: the manual marker must outlive any number of admin upserts."""
        store = self.make()
        key = person_hash("20002")
        store.upsert_person(key, "小明", now=self.clock.value, admin=True)
        store.set_person_relation(key, "friend", now=self.clock.value, set_by="admin-command")
        for _ in range(5):
            self.clock.value += 1000
            store.upsert_person(key, "小明", now=self.clock.value, admin=True)
            record = store.read_person(key)
            self.assertEqual(record["relation"], "friend")
            self.assertEqual(record["set_by"], "admin-command")

    def test_auto_promote_still_works_for_a_new_admin(self):
        """Guard: carrying set_by must not disable auto-promote."""
        store = self.make()
        key = person_hash("30003")
        record = store.upsert_person(key, "新主人", now=self.clock.value, admin=True)
        self.assertEqual(record["relation"], "master")
        self.assertNotIn("set_by", record)
        # A non-admin first sight still defaults to stranger.
        other = person_hash("40004")
        self.assertEqual(store.upsert_person(other, "路人", now=self.clock.value)["relation"], "stranger")
        # ...and only becomes master once it is genuinely an admin id.
        promoted = store.upsert_person(other, "路人", now=self.clock.value, admin=True)
        self.assertEqual(promoted["relation"], "master")

    def test_empty_set_by_counts_as_missing(self):
        """缺陷③: an empty marker is not a human decision, so auto-promote applies.

        The guard reads ``not entry.get("set_by")`` so empty and absent mean the
        same thing — matching how the marker itself is carried forward.
        """
        store = self.make()
        key = person_hash("50005")
        store.upsert_person(key, "阿强", now=self.clock.value, admin=True)
        store.set_person_relation(key, "friend", now=self.clock.value, set_by="")
        self.assertEqual(store.read_person(key)["relation"], "friend")
        self.assertEqual(store.read_person(key).get("set_by"), "")

        promoted = store.upsert_person(key, "阿强", now=self.clock.value, admin=True)
        self.assertEqual(promoted["relation"], "master", "an empty marker must not block promotion")

        # A real marker still blocks it.
        store.set_person_relation(key, "friend", now=self.clock.value, set_by="admin-command")
        blocked = store.upsert_person(key, "阿强", now=self.clock.value, admin=True)
        self.assertEqual(blocked["relation"], "friend")
        self.assertEqual(blocked.get("set_by"), "admin-command")

    def test_groups_field_is_always_carried(self):
        """列表字段永远 carry — the set_by lesson, applied to ``groups``."""
        store = self.make()
        key = person_hash("60006")
        store.upsert_person(key, "阿伟", now=self.clock.value, group_hash="group-a")
        self.assertEqual(store.read_person(key)["groups"], ["group-a"])
        # A later upsert without a group must not drop what was recorded.
        store.upsert_person(key, "阿伟", now=self.clock.value)
        self.assertEqual(store.read_person(key)["groups"], ["group-a"])
        # A second group is appended, and repeats are ignored.
        store.upsert_person(key, "阿伟", now=self.clock.value, group_hash="group-b")
        store.upsert_person(key, "阿伟", now=self.clock.value, group_hash="group-b")
        self.assertEqual(store.read_person(key)["groups"], ["group-a", "group-b"])

    def test_set_relation_rejects_unknown_value(self):
        store = self.make()
        key = person_hash("10001")
        store.upsert_person(key, "x", now=self.clock.value)
        with self.assertRaises(ValueError):
            store.set_person_relation(key, "boss", now=self.clock.value)

    def test_read_person_missing_returns_none(self):
        store = self.make()
        self.assertIsNone(store.read_person("0000000000000000"))
        self.assertEqual(store.list_persons(), {})

    def test_debug_log_never_contains_raw_id_or_nickname(self):
        self.clock = Clock()
        store = StateStore(str(self.tmp / "state"), debug=True, now=self.clock)
        raw_id = "10001"
        store.upsert_person(person_hash(raw_id), "秘密昵称", now=self.clock.value, admin=True)
        log_text = (self.tmp / "state" / "logs" / "dafeiyu.log").read_text(encoding="utf-8")
        self.assertNotIn(raw_id, log_text)
        self.assertNotIn("秘密昵称", log_text)
        self.assertIn(person_hash(raw_id), log_text)

    def test_raw_user_id_never_written_to_persons_json(self):
        store = self.make()
        raw_id = "10001"
        store.upsert_person(person_hash(raw_id), "张三", now=self.clock.value)
        payload = (self.tmp / "state" / "persons.json").read_text(encoding="utf-8")
        self.assertNotIn(raw_id, payload)
        self.assertIn(person_hash(raw_id), payload)

    def test_persons_file_survives_reopen(self):
        store = self.make()
        key = person_hash("10001")
        store.upsert_person(key, "张三", now=self.clock.value)
        reopened = StateStore(str(self.tmp / "state"), debug=False, now=self.clock)
        record = reopened.read_person(key)
        self.assertIsNotNone(record)
        self.assertEqual(record["current_nickname"], "张三")
        self.assertEqual(record["relation"], "stranger")


if __name__ == "__main__":
    unittest.main()
