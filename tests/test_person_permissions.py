"""Relation-based permission matrix and the ``/mem_person`` commands.

Covers the 好友关系分层权限 work: which relation may read, write, run admin
commands or destructive ones; the two ``/mem_person`` subcommands; and their
fail-closed behaviour.

The manual-wins guard lives in ``test_manual_relation_wins.py``; it needs a
state-layer fix first (see the report on ``StateStore._person_entry``).
"""

from __future__ import annotations

import asyncio
import unittest

from helpers import TempDirTest  # noqa: F401  (puts the plugin root on sys.path)
from test_runtime import FakeEvent, Harness, _collect

from dafeiyu_policy import bot_tag, hashed_id, legacy_group_scope
from dafeiyu_state import person_hash

GROUP = "777888"
ADMIN = "9001"
USER = "5001"

#: relation -> (may reveal in a group, may create in a group, admin/destructive)
#:
#: 用户裁决 2026-11-05 放开群写入: creating is open to every sender, so the
#: write column is now True for friend/other_fish/stranger. Reading and
#: admin/destructive powers are unchanged from task-19.
GROUP_MATRIX = {
    "master": (True, True, True),
    "friend": (True, True, False),
    "other_fish": (True, True, False),
    "stranger": (False, True, False),
}


class PersonPermissionTest(Harness):
    def run_command(self, handler, event):
        return _collect(handler(event))

    def out(self, handler, event):
        return self.run_command(handler, event)[0][1]

    def set_relation(self, user_id, relation):
        self.plugin.state.upsert_person(person_hash(user_id), "tester")
        self.plugin.state.set_person_relation(person_hash(user_id), relation, set_by="test")

    def group_turn(self, user_id=USER):
        return self.plugin._turn(FakeEvent("x", group_id=GROUP, user_id=user_id))

    def private_turn(self, user_id=USER):
        return self.plugin._turn(FakeEvent("x", group_id="", user_id=user_id))

    # -- ① permission matrix ---------------------------------------------
    def test_group_matrix_read_write_admin(self):
        for relation, (reveal, write, admin) in GROUP_MATRIX.items():
            with self.subTest(relation=relation):
                self.set_relation(USER, relation)
                self.assertEqual(self.plugin._may_reveal(self.group_turn()), reveal, relation)
                self.assertEqual(self.plugin._may_write(self.group_turn()), write, relation)
                self.assertEqual(self.plugin._admin_allowed(USER), admin, relation)

    def test_no_profile_is_treated_as_stranger(self):
        # An unknown sender gains no READ or admin rights by being unknown...
        self.assertFalse(self.plugin._may_reveal(self.group_turn()))
        self.assertFalse(self.plugin._admin_allowed(USER))
        # ...but creating a memory is open to everyone per the 2026-11-05 ruling.
        self.assertTrue(self.plugin._may_write(self.group_turn()))

    def test_admin_ids_member_is_allowed_in_a_group(self):
        self.assertTrue(self.plugin._may_reveal(self.group_turn(ADMIN)))
        self.assertTrue(self.plugin._may_write(self.group_turn(ADMIN)))
        self.assertTrue(self.plugin._admin_allowed(ADMIN))

    def test_master_equals_admin_even_without_admin_ids(self):
        self.plugin._admins = set()
        self.set_relation(USER, "master")
        self.assertTrue(self.plugin._may_reveal(self.group_turn()))
        self.assertTrue(self.plugin._may_write(self.group_turn()))
        self.assertTrue(self.plugin._admin_allowed(USER))

    def test_audit_stays_admin_equivalent(self):
        # friend/other_fish were widened for content reads only.
        for relation in ("friend", "other_fish"):
            with self.subTest(relation=relation):
                self.set_relation(USER, relation)
                self.assertFalse(self.plugin._may_admin(self.group_turn()), relation)
        self.assertTrue(self.plugin._may_admin(self.group_turn(ADMIN)))

    def test_private_chat_reads_and_writes_are_unchanged(self):
        self.set_relation(USER, "stranger")
        turn = self.private_turn()
        self.assertTrue(self.plugin._may_reveal(turn))
        self.assertTrue(self.plugin._may_write(turn))
        # Destructive stays fail-closed for a non-admin in private too.
        self.assertFalse(self.plugin._admin_allowed(USER))

    # -- ③ /mem_person list ----------------------------------------------
    def test_mem_person_list_shows_hash_nickname_relation_and_dates(self):
        self.set_relation(USER, "friend")
        out = self.out(self.plugin.mem_person, FakeEvent("/mem_person list", group_id=GROUP, user_id=ADMIN))
        self.assertIn(person_hash(USER), out)
        self.assertIn("tester", out)
        self.assertIn("friend", out)
        self.assertIn("first=", out)
        self.assertIn("last=", out)

    def test_mem_person_list_empty(self):
        out = self.out(self.plugin.mem_person, FakeEvent("/mem_person list", group_id=GROUP, user_id=ADMIN))
        self.assertIn("no person profiles", out)

    # -- ② /mem_person set ------------------------------------------------
    def test_mem_person_set_by_hash(self):
        self.plugin.state.upsert_person(person_hash(USER), "tester")
        out = self.out(
            self.plugin.mem_person,
            FakeEvent(f"/mem_person set {person_hash(USER)} friend", group_id=GROUP, user_id=ADMIN),
        )
        self.assertIn("-> friend", out)
        self.assertEqual(self.plugin.state.read_person(person_hash(USER))["relation"], "friend")

    def test_mem_person_set_by_nickname(self):
        self.plugin.state.upsert_person(person_hash(USER), "阿鲤")
        out = self.out(
            self.plugin.mem_person,
            FakeEvent("/mem_person set 阿鲤 master", group_id=GROUP, user_id=ADMIN),
        )
        self.assertIn("-> master", out)
        self.assertEqual(self.plugin.state.read_person(person_hash(USER))["relation"], "master")

    def test_mem_person_set_records_set_by(self):
        self.plugin.state.upsert_person(person_hash(USER), "tester")
        self.out(
            self.plugin.mem_person,
            FakeEvent(f"/mem_person set {person_hash(USER)} friend", group_id=GROUP, user_id=ADMIN),
        )
        entry = self.plugin.state.read_person(person_hash(USER))
        self.assertEqual(entry.get("set_by"), f"user:{person_hash(ADMIN)}")

    # -- ⑥ fail-closed ----------------------------------------------------
    def test_mem_person_requires_admin(self):
        out = self.out(self.plugin.mem_person, FakeEvent("/mem_person list", group_id=GROUP, user_id=USER))
        self.assertEqual(out, "admin only")
        out = self.out(
            self.plugin.mem_person,
            FakeEvent(f"/mem_person set {person_hash(USER)} friend", group_id=GROUP, user_id=USER),
        )
        self.assertEqual(out, "admin only")

    def test_mem_person_fail_closed_with_empty_admin_ids(self):
        self.plugin._admins = set()
        out = self.out(self.plugin.mem_person, FakeEvent("/mem_person list", group_id=GROUP, user_id=ADMIN))
        self.assertEqual(out, "admin only")

    def test_unknown_relation_is_refused_before_any_write(self):
        self.plugin.state.upsert_person(person_hash(USER), "tester")
        out = self.out(
            self.plugin.mem_person,
            FakeEvent(f"/mem_person set {person_hash(USER)} bestie", group_id=GROUP, user_id=ADMIN),
        )
        self.assertIn("relation must be one of", out)
        self.assertEqual(self.plugin.state.read_person(person_hash(USER))["relation"], "stranger")

    def test_ambiguous_nickname_is_refused(self):
        self.plugin.state.upsert_person(person_hash("111111"), "同名")
        self.plugin.state.upsert_person(person_hash("222222"), "同名")
        out = self.out(
            self.plugin.mem_person,
            FakeEvent("/mem_person set 同名 friend", group_id=GROUP, user_id=ADMIN),
        )
        self.assertIn("no single person matches", out)

    def test_unknown_nickname_is_refused(self):
        out = self.out(
            self.plugin.mem_person,
            FakeEvent("/mem_person set 不存在 friend", group_id=GROUP, user_id=ADMIN),
        )
        self.assertIn("no single person matches", out)

    def test_mem_person_usage(self):
        self.assertIn("usage", self.out(self.plugin.mem_person, FakeEvent("/mem_person", group_id=GROUP, user_id=ADMIN)))
        self.assertIn(
            "usage",
            self.out(self.plugin.mem_person, FakeEvent("/mem_person set only", group_id=GROUP, user_id=ADMIN)),
        )
        self.assertIn(
            "usage",
            self.out(self.plugin.mem_person, FakeEvent("/mem_person frobnicate", group_id=GROUP, user_id=ADMIN)),
        )

    # -- ⑤ manual relation must survive the per-message turn hook ----------
    def test_manual_relation_survives_repeated_messages(self):
        """The hook upserts on EVERY message, so one pass proves nothing.

        ``set_by`` used to be dropped by ``_person_entry``; an admin's manual
        relation was then overwritten on their second message.
        """
        key = person_hash(ADMIN)
        self.plugin.state.upsert_person(key, "tester", admin=True)
        self.plugin.state.set_person_relation(key, "friend", set_by="user:test")
        for _ in range(3):
            self.inject("why did the napcat bridge crash yesterday", group_id=GROUP, user_id=ADMIN)
            entry = self.plugin.state.read_person(key)
            self.assertEqual(entry["relation"], "friend")
            self.assertEqual(entry["set_by"], "user:test")

    def test_new_admin_is_still_auto_promoted_by_the_turn_hook(self):
        """Carrying set_by must not disable auto-promote for a fresh admin."""
        self.inject("why did the napcat bridge crash yesterday", group_id=GROUP, user_id=ADMIN)
        entry = self.plugin.state.read_person(person_hash(ADMIN))
        self.assertEqual(entry["relation"], "master")
        self.assertNotIn("set_by", entry)

    def test_non_admin_sender_is_not_promoted(self):
        self.inject("why did the napcat bridge crash yesterday", group_id=GROUP, user_id=USER)
        entry = self.plugin.state.read_person(person_hash(USER))
        self.assertEqual(entry["relation"], "stranger")

    # -- task-21 缺陷①: audit is admin-equivalent, not a content read ------
    def test_memory_audit_tool_denied_for_friend_in_group(self):
        self.set_relation(USER, "friend")
        out = asyncio.run(
            self.plugin.memory_audit_tool(FakeEvent("audit", group_id=GROUP, user_id=USER))
        )
        self.assertEqual(out, "group audit denied: admin only (master/管理员)")

    def test_memory_audit_tool_denied_for_other_fish_in_group(self):
        self.set_relation(USER, "other_fish")
        out = asyncio.run(
            self.plugin.memory_audit_tool(FakeEvent("audit", group_id=GROUP, user_id=USER))
        )
        self.assertEqual(out, "group audit denied: admin only (master/管理员)")

    def test_memory_audit_tool_allowed_for_admin_in_group(self):
        out = asyncio.run(
            self.plugin.memory_audit_tool(FakeEvent("audit", group_id=GROUP, user_id=ADMIN))
        )
        self.assertNotIn("denied", out)

    # -- task-21 全群成员解析 ---------------------------------------------
    def test_group_member_hashes_only_includes_this_groups_speakers(self):
        self.inject("why did the napcat bridge crash yesterday", group_id=GROUP, user_id=USER)
        self.inject("why did the napcat bridge crash yesterday", group_id="555555555", user_id="7007")
        hashes = self.plugin._group_member_hashes(self.group_turn())
        self.assertIn(person_hash(USER), hashes)
        self.assertNotIn(person_hash("7007"), hashes)

    def test_group_member_hashes_caps_at_eight_and_keeps_the_speaker(self):
        self.inject("why did the napcat bridge crash yesterday", group_id=GROUP, user_id=USER)
        for index in range(10):
            self.inject(
                "why did the napcat bridge crash yesterday", group_id=GROUP, user_id=str(8000 + index)
            )
        hashes = self.plugin._group_member_hashes(self.group_turn())
        self.assertEqual(len(hashes), 8, "a busy group must not fan out unbounded")
        self.assertEqual(hashes[0], person_hash(USER), "the speaker's bucket is never dropped")

    def test_group_member_hashes_is_empty_for_private_chats(self):
        turn = self.plugin._turn(FakeEvent("hi", group_id="", user_id=USER))
        self.assertEqual(self.plugin._group_member_hashes(turn), [])

    # -- task-25 _may_edit: creating opened, editing did not ---------------
    def _remember_origin(self, short_key, scope):
        """Simulate a recall that recorded where this id came from."""
        self.plugin._id_scope[short_key] = scope

    def _other_member_bucket(self):
        return f"qq:{bot_tag('10001')}:group:{hashed_id(GROUP)}:{hashed_id('7007007')}"

    def test_may_edit_admin_can_edit_another_members_bucket(self):
        self._remember_origin("deadbeef", self._other_member_bucket())
        self.assertTrue(self.plugin._may_edit(self.group_turn(user_id=ADMIN), "deadbeef"))

    def test_may_edit_master_can_edit_another_members_bucket(self):
        self.set_relation(USER, "master")
        self._remember_origin("deadbeef", self._other_member_bucket())
        self.assertTrue(self.plugin._may_edit(self.group_turn(user_id=USER), "deadbeef"))

    def test_may_edit_member_can_edit_their_own_bucket(self):
        self._remember_origin("cafebabe", self.group_turn(user_id=USER).scope)
        self.assertTrue(self.plugin._may_edit(self.group_turn(user_id=USER), "cafebabe"))

    def test_may_edit_member_cannot_edit_another_members_bucket(self):
        self._remember_origin("cafebabe", self._other_member_bucket())
        self.assertFalse(
            self.plugin._may_edit(self.group_turn(user_id=USER), "cafebabe"),
            "the global view hands out other people's ids; editing one must be refused",
        )

    def test_may_edit_member_cannot_edit_the_legacy_bucket(self):
        self._remember_origin("cafebabe", legacy_group_scope("per_conversation", "10001", GROUP))
        self.assertFalse(self.plugin._may_edit(self.group_turn(user_id=USER), "cafebabe"))

    def test_may_edit_fails_closed_when_the_origin_is_unknown(self):
        # No mapping entry (restart, or an id that never came through a recall):
        # unknown must mean "refused", never "allowed".
        self.assertFalse(self.plugin._may_edit(self.group_turn(user_id=USER), "00000000"))

    def test_create_is_open_but_edit_is_not(self):
        """The 2026-11-05 ruling opened creating only."""
        self.set_relation(USER, "stranger")
        self.assertTrue(self.plugin._may_write(self.group_turn()))
        self._remember_origin("cafebabe", self._other_member_bucket())
        self.assertFalse(self.plugin._may_edit(self.group_turn(user_id=USER), "cafebabe"))

    def test_private_chat_edits_even_when_the_origin_map_is_empty(self):
        """A restart empties the in-memory id→bucket map.

        A private chat has exactly one bucket, so ``_recent_id`` alone already
        proves the id came from this conversation. Requiring the map would break
        the first edit after a restart and answer with a group-flavoured message
        inside a private chat.
        """
        self.inject("why did the napcat bridge crash yesterday", group_id="")
        plugin = self._make_plugin()  # fresh instance == a restart
        self.assertEqual(plugin._id_scope, {}, "the map is in-memory state")

        event = FakeEvent("x", group_id="", user_id=USER)
        turn = plugin._turn(event)
        self.assertTrue(plugin._may_edit(turn, "92cfc0da"))

        out = asyncio.run(
            plugin.memory_update_tool(
                event,
                "92cfc0da",
                "the bridge listens on 8080 now",
                "operator corrected the port yesterday",
                "",
                "",
            )
        )
        self.assertIn("updated [92cfc0da]", out)

    def test_private_chat_edit_gate_never_reaches_the_group_message(self):
        """In a private chat the gate passes outright, whatever the origin.

        That is the whole point of the branch: the group-flavoured denial must be
        unreachable there, so a private user never sees "group edit denied".
        """
        turn = self.plugin._turn(FakeEvent("x", group_id="", user_id=USER))
        self.assertTrue(self.plugin._may_edit(turn, "cafebabe"))
        self.assertTrue(self.plugin._may_edit(turn, "00000000"))  # unknown id
        self.assertTrue(self.plugin._may_edit(turn, ""))


if __name__ == "__main__":
    unittest.main()
