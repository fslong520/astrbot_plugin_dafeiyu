"""Speaker attribution in injected recall, plus the person档案 hook.

Covers the 注入 speaker标注 work: every recalled title is prefixed with
``[昵称/relation]``, a missing档案 degrades to ``[未知/stranger]`` instead of
being guessed, each message upserts the sender's person record, and a group
injection carries a ``本群成员`` line while a private chat does not.

Reuses the runtime harness so the real plugin path is exercised (no AstrBot
install required).
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path

from helpers import TempDirTest  # noqa: F401  (puts the plugin root on sys.path)
from test_runtime import FakeEvent, FakeReq, Harness  # installs the astrbot stubs

import main as plugin_main
from dafeiyu_state import person_hash

GROUP = "888888888"
USER_A = "5001"
USER_B = "6002"


class NamedEvent(FakeEvent):
    """A FakeEvent whose sender name can differ per sender."""

    def __init__(self, text, name, **kwargs):
        super().__init__(text, **kwargs)
        self._name = name

    def get_sender_name(self):
        return self._name


class SpeakerInjectTest(Harness):
    def _persons(self):
        path = Path(self.plugin.state.files["persons"])
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _inject_named(self, name, text="why did the napcat bridge crash yesterday", **kwargs):
        event = NamedEvent(text, name, **kwargs)
        req = FakeReq()

        async def go():
            await self.plugin.on_llm_request(event, req)
            await self._drain()

        asyncio.run(go())
        return req

    # -- ① turn hook ------------------------------------------------------
    def test_each_message_upserts_the_sender_profile(self):
        self.inject("why did the napcat bridge crash yesterday")
        raw = self._persons()
        key = person_hash(USER_A)
        self.assertIn(key, raw, "the turn hook must write the sender's person record")
        self.assertEqual(raw[key]["current_nickname"], "tester")
        self.assertEqual(raw[key]["relation"], "stranger")

    def test_two_senders_in_one_group_each_get_a_profile(self):
        self._inject_named("阿鲤", group_id=GROUP, user_id=USER_A)
        self._inject_named("老王", group_id=GROUP, user_id=USER_B)
        raw = self._persons()
        self.assertEqual(raw[person_hash(USER_A)]["current_nickname"], "阿鲤")
        self.assertEqual(raw[person_hash(USER_B)]["current_nickname"], "老王")

    def test_admin_sender_is_recorded_as_master(self):
        self.plugin.config["admin_ids"] = [USER_A]
        self.inject("why did the napcat bridge crash yesterday")
        raw = self._persons()
        self.assertEqual(raw[person_hash(USER_A)]["relation"], "master")

    # -- ② injection format ----------------------------------------------
    def test_injected_title_carries_the_speaker_label(self):
        req = self.inject("why did the napcat bridge crash yesterday")
        part = req.extra_user_content_parts[0]
        self.assertIn("- [92cfc0da] [tester/stranger] first remembered title", part.text)

    def test_label_uses_the_person_relation(self):
        # The turn hook refreshes the nickname, so drive a non-default relation
        # through admin_ids and check it reaches the injected label.
        self.plugin.config["admin_ids"] = [USER_A]
        req = self.inject("why did the napcat bridge crash yesterday")
        part = req.extra_user_content_parts[0]
        self.assertIn("- [92cfc0da] [tester/master] first remembered title", part.text)

    def test_private_chat_is_also_labelled(self):
        req = self.inject("why did the napcat bridge crash yesterday", group_id="")
        part = req.extra_user_content_parts[0]
        self.assertIn("- [92cfc0da] [tester/stranger] first remembered title", part.text)

    # -- ③ degrade when there is no档案 ----------------------------------
    def test_missing_profile_degrades_to_unknown(self):
        # Drive recall WITHOUT the profiling hook, so no person record exists.
        event = FakeEvent("why did the napcat bridge crash yesterday")
        req = FakeReq()

        async def go():
            await self.plugin._prepare_recall(event, req)
            await self._drain()

        asyncio.run(go())
        part = req.extra_user_content_parts[0]
        self.assertIn("- [92cfc0da] [未知/stranger] first remembered title", part.text)
        self.assertEqual(self._persons(), {})

    # -- ④ group members line --------------------------------------------
    def test_group_injection_lists_current_speakers(self):
        self._inject_named("阿鲤", group_id=GROUP, user_id=USER_A)
        self._inject_named("老王", group_id=GROUP, user_id=USER_B)
        part = self._inject_named("阿鲤", group_id=GROUP, user_id=USER_A).extra_user_content_parts[0]
        self.assertIn("本群成员：", part.text)
        self.assertIn("阿鲤/stranger", part.text)
        self.assertIn("老王/stranger", part.text)
        self.assertIn("</system-reminder>", part.text)
        # The line belongs inside the reminder block, not appended after it.
        self.assertLess(
            part.text.index("本群成员："), part.text.index("</system-reminder>")
        )

    def test_private_chat_has_no_members_line(self):
        req = self.inject("why did the napcat bridge crash yesterday", group_id="")
        self.assertNotIn("本群成员：", req.extra_user_content_parts[0].text)

    def test_group_with_no_seen_speakers_omits_the_line(self):
        # Recall only, no message ever recorded a speaker for this group.
        event = FakeEvent("why did the napcat bridge crash yesterday", group_id="777777777")
        req = FakeReq()

        async def go():
            await self.plugin._prepare_recall(event, req)
            await self._drain()

        asyncio.run(go())
        if req.extra_user_content_parts:
            self.assertNotIn("本群成员：", req.extra_user_content_parts[0].text)

    # -- task-21: whole-group fan-out ------------------------------------
    def test_group_recall_fans_out_over_every_member(self):
        """A speaking must read B's bucket too, plus the legacy bucket.

        The old behaviour only asked for the caller's own bucket, which hid
        everything the rest of the group remembered.
        """
        self._inject_named("阿鲤", group_id=GROUP, user_id=USER_A)
        self._inject_named("老王", group_id=GROUP, user_id=USER_B)

        # Isolate the triggering message: each earlier message already fanned
        # out over however many members existed at that moment.
        for stale in self.calls.glob("*.log"):
            stale.unlink()
        self._inject_named("阿鲤", group_id=GROUP, user_id=USER_A)  # triggers recall

        recalls = [args for args in self.commands() if args == "recall"]
        self.assertEqual(len(recalls), 3, f"expected 2 member legs + legacy, got {self.commands()}")

        projects = [a for a in self.args_for("recall") if a]
        self.assertEqual(len(projects), 3)
        for args in projects:
            self.assertIn("--project", args)

    def test_private_recall_stays_a_single_leg(self):
        self.inject("why did the napcat bridge crash yesterday", group_id="")
        recalls = [args for args in self.commands() if args == "recall"]
        self.assertEqual(len(recalls), 1, "a private chat must not fan out")

    def test_budget_is_split_across_the_full_eight_members_plus_legacy(self):
        """8 member buckets + legacy = 9 legs; per-leg limit stays >= 1 and the
        merged total never exceeds the requested budget."""
        scopes = [f"qq:b:group:g:{index:016x}" for index in range(8)] + ["qq:b:group:g"]
        self.assertEqual(len(scopes), 9)

        envelope = asyncio.run(self.plugin._recall_fanout("napcat", scopes, 5, False))

        recalls = [args for args in self.commands() if args == "recall"]
        self.assertEqual(len(recalls), 9, "one leg per bucket, legacy included")
        args = [args for args in self.args_for("recall") if args]
        self.assertEqual(len(args), 9, "every leg must have been issued")
        # per = max(1, 5 // 9) = 1, so no leg is ever given a zero limit.
        self.assertTrue(all("--limit 1" in item for item in args), args)
        # merge_recall_envelopes applies the total budget.
        self.assertLessEqual(len(envelope["items"]), 5)

    # -- task-21 缺陷②: single-scope failure must fail open ---------------
    def test_single_scope_recall_failure_returns_an_empty_envelope(self):
        broken = self.tmp / "broken-rsrs"
        broken.write_text("#!/bin/sh\necho boom 1>&2\nexit 3\n", encoding="utf-8")
        os.chmod(broken, 0o755)
        plugin = self._make_plugin(rsrs_bin=str(broken))

        envelope = asyncio.run(plugin._recall_fanout("napcat", ["qq:x:private:y"], 5, False))
        self.assertEqual(envelope, {"items": [], "details": []})

    def test_single_scope_failure_does_not_break_the_injection_path(self):
        broken = self.tmp / "broken-rsrs"
        broken.write_text("#!/bin/sh\necho boom 1>&2\nexit 3\n", encoding="utf-8")
        os.chmod(broken, 0o755)
        plugin = self._make_plugin(rsrs_bin=str(broken))
        event = FakeEvent("why did the napcat bridge crash yesterday")
        req = FakeReq()

        async def go():
            await plugin.on_llm_request(event, req)

        asyncio.run(go())
        # Recall fails open; the standing format rule is still injected.
        self.assert_no_recall(req)

    # -- task-25: rejection text must name a reason, not the group config ---
    def test_old_misleading_denial_string_is_gone(self):
        """The old wording made the LLM restate a permission problem as
        "this group has write permission off", which is wrong and unactionable."""
        source = Path(plugin_main.__file__).read_text(encoding="utf-8")
        self.assertNotIn("not allowed in this group; ask an admin", source)
        self.assertNotIn("ask an admin", source)

    def test_denial_texts_name_the_reason(self):
        turn = self.plugin._turn(FakeEvent("x", group_id="777888", user_id="5001"))
        self.assertEqual(
            self.plugin._read_denied(turn),
            "group read denied: sender relation is stranger (需要 friend/master/管理员)",
        )
        self.assertEqual(
            self.plugin._edit_denied(turn, "target memory belongs to another member (admin only)"),
            "group edit denied: target memory belongs to another member (admin only)",
        )
        self.assertEqual(
            self.plugin._audit_denied(), "group audit denied: admin only (master/管理员)"
        )
        # Nothing blames the group's configuration.
        for text in (
            self.plugin._read_denied(turn),
            self.plugin._audit_denied(),
        ):
            self.assertNotIn("group has", text)
            self.assertNotIn("permission off", text)

    def test_stranger_group_read_denies_with_the_relation_named(self):
        self.inject("why did the napcat bridge crash yesterday", group_id="777888", user_id="5001")
        out = asyncio.run(
            self.plugin.memory_show_tool(
                FakeEvent("x", group_id="777888", user_id="5001"), "92cfc0da"
            )
        )
        self.assertIn("group read denied: sender relation is stranger", out)
        self.assertNotIn("BODY_SHOULD_NOT_LEAK", out)

    # -- task-26: standing 【】 / no-asterisk output rule -------------------
    def test_group_turn_carries_the_standing_format_rule(self):
        req = self.inject(
            "why did the napcat bridge crash yesterday", group_id=GROUP, user_id=USER_A
        )
        parts = self.assert_standing_format_rule(req)
        # A group turn still gets its recall block, before the rule.
        self.assertEqual(len(parts), 2)
        self.assertIn("Recalled memory", parts[0].text)

    def test_private_turn_carries_the_standing_format_rule(self):
        req = self.inject("why did the napcat bridge crash yesterday", group_id="")
        parts = self.assert_standing_format_rule(req)
        self.assertEqual(len(parts), 2)
        self.assertIn("Recalled memory", parts[0].text)

    def test_skipped_recall_still_carries_the_standing_format_rule(self):
        # Too short to recall: the rule must survive, since that is exactly the
        # turn where the fish would otherwise answer unformatted.
        req = self.inject("ok")
        self.assert_no_recall(req)

    def test_format_rule_names_the_bracket_example_and_forbids_asterisks(self):
        req = self.inject("why did the napcat bridge crash yesterday", group_id=GROUP, user_id=USER_A)
        text = req.extra_user_content_parts[-1].text
        self.assertIn("【百年江湖 随意沉浮】", text)
        self.assertIn("禁止星号", text)
        self.assertIn("**", text)  # the rule must name the offending syntax
        self.assertIn("<system-reminder>", text)

    def test_format_rule_names_the_nickname_change(self):
        """认人 H1: the standing instruction tells the model renames happen."""
        self.assertIn(
            "昵称会变：#括号内『曾用』名与现名指同一人",
            plugin_main.FORMAT_INSTRUCTION,
        )
        self.assertIn("以注入档案为准，不以昵称为准", plugin_main.FORMAT_INSTRUCTION)

    def test_speaker_label_carries_past_names(self):
        record = {
            "current_nickname": "新名",
            "nicknames": ["旧名甲", "新名"],
            "relation": "master",
        }
        self.assertEqual(
            plugin_main.speaker_label("k", lambda _key: record),
            "[新名(曾用:旧名甲)/master]",
        )

    def test_machine_annotations_are_left_alone(self):
        """The rule governs what the fish SAYS, not the model-facing notes."""
        req = self.inject("why did the napcat bridge crash yesterday", group_id=GROUP, user_id=USER_A)
        recall_text = req.extra_user_content_parts[0].text
        self.assertIn("[tester/stranger]", recall_text)
        self.assertNotIn("【tester/stranger】", recall_text)


if __name__ == "__main__":
    unittest.main()
