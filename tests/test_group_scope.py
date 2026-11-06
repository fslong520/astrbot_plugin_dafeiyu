"""Per-member group scope, speaker attribution and merged group recall.

Covers the A+B 认人 change: a group conversation now writes to a bucket per
member, memories carry a speaker attribution, and a group read fans out over the
member bucket plus the legacy member-less bucket so older history stays visible.

Everything here is a pure function, so no AstrBot import is needed.
"""

from __future__ import annotations

import types
import unittest

from helpers import PLUGIN_ROOT  # noqa: F401  (puts the plugin root on sys.path)

from dafeiyu_policy import (
    Turn,
    derive_scope,
    group_recall_scopes,
    bot_tag,
    hashed_id,
    legacy_group_scope,
    merge_recall_envelopes,
    sender_tag,
    speaker_header,
    turn_from_event,
)
from dafeiyu_state import person_hash

BOT = "123456789"
GROUP = "987654321"
ALICE = "111111111"
BOB = "222222222"


def group_turn(user_id=ALICE, sender_name="阿鲤", scope_mode="per_conversation"):
    scope = derive_scope(scope_mode, BOT, GROUP, user_id)
    return Turn(
        self_id=BOT,
        group_id=GROUP,
        user_id=user_id,
        session_id=f"group_{GROUP}",
        sender_name=sender_name,
        is_group=True,
        scope=scope,
        scope_mode=scope_mode,
    )


def private_turn(user_id=ALICE, sender_name="阿鲤", scope_mode="per_conversation"):
    scope = derive_scope(scope_mode, BOT, "", user_id)
    return Turn(
        self_id=BOT,
        group_id="",
        user_id=user_id,
        session_id=f"private_{user_id}",
        sender_name=sender_name,
        is_group=False,
        scope=scope,
        scope_mode=scope_mode,
    )


class GroupBucketTests(unittest.TestCase):
    def test_two_members_of_one_group_get_different_buckets(self):
        alice = derive_scope("per_conversation", BOT, GROUP, ALICE)
        bob = derive_scope("per_conversation", BOT, GROUP, BOB)
        self.assertNotEqual(alice, bob)
        # Both still belong to the same group: the shared prefix proves it.
        self.assertEqual(alice.rsplit(":", 1)[0], bob.rsplit(":", 1)[0])
        self.assertIn(":group:", alice)

    def test_bucket_carries_no_raw_id(self):
        for value in (BOT, GROUP, ALICE, BOB):
            self.assertNotIn(value, derive_scope("per_conversation", BOT, GROUP, ALICE))

    def test_private_and_per_user_scopes_are_unchanged(self):
        # Private keeps its historical shape (no member segment).
        self.assertEqual(
            derive_scope("per_conversation", BOT, "", ALICE),
            f"qq:{bot_tag(BOT)}:private:{hashed_id(ALICE)}",
        )
        # per_user collapses group and private onto one user bucket, as before.
        self.assertEqual(
            derive_scope("per_user", BOT, GROUP, ALICE),
            f"qq:{bot_tag(BOT)}:user:{hashed_id(ALICE)}",
        )
        self.assertEqual(derive_scope("global", BOT, GROUP, ALICE), "qq:global")

    def test_global_has_no_legacy_group_bucket(self):
        self.assertIsNone(legacy_group_scope("global", BOT, GROUP))


class LegacyBucketTests(unittest.TestCase):
    def test_legacy_bucket_is_the_pre_change_group_scope(self):
        # Exactly the shape derive_scope returned before the member segment.
        self.assertEqual(
            legacy_group_scope("per_conversation", BOT, GROUP),
            f"qq:{bot_tag(BOT)}:group:{hashed_id(GROUP)}",
        )

    def test_group_read_covers_member_and_legacy_buckets(self):
        scopes = group_recall_scopes(group_turn())
        self.assertEqual(len(scopes), 2)
        self.assertEqual(scopes[0], group_turn().scope)
        self.assertEqual(scopes[1], legacy_group_scope("per_conversation", BOT, GROUP))
        self.assertNotEqual(scopes[0], scopes[1])

    def test_private_chat_reads_exactly_one_scope(self):
        turn = private_turn()
        self.assertEqual(group_recall_scopes(turn), [turn.scope])

    def test_per_user_and_global_reads_are_unchanged(self):
        for mode in ("per_user", "global"):
            turn = group_turn(scope_mode=mode)
            self.assertEqual(group_recall_scopes(turn), [turn.scope], mode)


class WholeGroupScopeTests(unittest.TestCase):
    """A group read must see the WHOLE group, not just the caller's bucket."""

    def _bucket(self, user_id):
        return f"qq:{bot_tag(BOT)}:group:{hashed_id(GROUP)}:{hashed_id(user_id)}"

    def test_group_read_covers_every_member_bucket(self):
        scopes = group_recall_scopes(group_turn(user_id=ALICE), [hashed_id(ALICE), hashed_id(BOB)])
        self.assertEqual(scopes, [self._bucket(ALICE), self._bucket(BOB),
                                  legacy_group_scope("per_conversation", BOT, GROUP)])

    def test_caller_bucket_is_included_even_when_absent_from_the_list(self):
        # A caller that resolved members must still read its own bucket.
        scopes = group_recall_scopes(group_turn(user_id=ALICE), [hashed_id(BOB)])
        self.assertIn(self._bucket(ALICE), scopes)

    def test_duplicate_member_hashes_are_collapsed(self):
        scopes = group_recall_scopes(
            group_turn(user_id=ALICE), [hashed_id(ALICE), hashed_id(ALICE), hashed_id(BOB)]
        )
        self.assertEqual(scopes, [self._bucket(ALICE), self._bucket(BOB),
                                  legacy_group_scope("per_conversation", BOT, GROUP)])

    def test_empty_members_falls_back_to_the_current_speaker(self):
        scopes = group_recall_scopes(group_turn(user_id=ALICE), [])
        self.assertEqual(scopes, [self._bucket(ALICE),
                                  legacy_group_scope("per_conversation", BOT, GROUP)])

    def test_private_bucket_never_mixes_in_member_buckets(self):
        turn = private_turn(user_id=ALICE)
        scopes = group_recall_scopes(turn, [hashed_id(ALICE), hashed_id(BOB)])
        self.assertEqual(scopes, [turn.scope])
        for scope in scopes:
            self.assertNotIn(":group:", scope)

    def test_per_user_and_global_stay_single_scope_with_members_given(self):
        for mode in ("per_user", "global"):
            turn = group_turn(user_id=ALICE, scope_mode=mode)
            scopes = group_recall_scopes(turn, [hashed_id(ALICE), hashed_id(BOB)])
            self.assertEqual(scopes, [turn.scope], mode)

    def test_member_bucket_shape_matches_derive_scope(self):
        # The bucket a read asks for must be exactly what the write produced.
        scopes = group_recall_scopes(group_turn(user_id=BOB), [hashed_id(BOB)])
        self.assertEqual(scopes[0], derive_scope("per_conversation", BOT, GROUP, BOB))


class AttributionTests(unittest.TestCase):
    def test_speaker_header_names_the_sender(self):
        self.assertEqual(speaker_header(group_turn(sender_name="阿鲤")), "【发言人】阿鲤")

    def test_speaker_header_falls_back_to_the_hash_not_the_raw_id(self):
        header = speaker_header(group_turn(sender_name=""))
        self.assertEqual(header, f"【发言人】{person_hash(ALICE)}")
        self.assertNotIn(ALICE, header)

    def test_sender_tag_uses_the_person_hash(self):
        self.assertEqual(sender_tag(group_turn()), f"sender:{person_hash(ALICE)}")
        # The tag must join against the persons.json key, which is person_hash.
        self.assertEqual(sender_tag(group_turn()), f"sender:{hashed_id(ALICE)}")
        self.assertNotIn(ALICE, sender_tag(group_turn()))

    def test_attribution_differs_between_two_speakers(self):
        self.assertNotEqual(
            sender_tag(group_turn(user_id=ALICE)), sender_tag(group_turn(user_id=BOB))
        )


class MergeRecallTests(unittest.TestCase):
    @staticmethod
    def envelope(rows):
        """rows: list of (short_id, score)."""
        return {
            "items": [{"id": short_id, "score": score, "value": f"{score} title"} for short_id, score in rows],
            "details": [{"id": f"full-{short_id}"} for short_id, _score in rows],
            "summary": {"mode": "hybrid"},
        }

    def test_merged_recall_contains_both_buckets(self):
        member = self.envelope([("aaaa1111", 0.9), ("bbbb2222", 0.5)])
        legacy = self.envelope([("cccc3333", 0.7)])
        merged = merge_recall_envelopes([member, legacy], limit=10)

        ids = [item["id"] for item in merged["items"]]
        self.assertEqual(sorted(ids), ["aaaa1111", "bbbb2222", "cccc3333"])
        # Ranked: the legacy hit must not stay buried behind the member bucket.
        self.assertEqual(ids[0], "aaaa1111")
        self.assertEqual(ids[1], "cccc3333")

    def test_merged_details_stay_index_aligned(self):
        merged = merge_recall_envelopes(
            [self.envelope([("aaaa1111", 0.9), ("bbbb2222", 0.5)]), self.envelope([("cccc3333", 0.7)])],
            limit=10,
        )
        self.assertEqual(len(merged["items"]), len(merged["details"]))
        for item, detail in zip(merged["items"], merged["details"]):
            self.assertEqual(detail["id"], f"full-{item['id']}")

    def test_duplicate_ids_across_buckets_are_dropped_once(self):
        both = self.envelope([("aaaa1111", 0.9)])
        merged = merge_recall_envelopes([both, both], limit=10)
        self.assertEqual([item["id"] for item in merged["items"]], ["aaaa1111"])

    def test_merge_respects_the_total_budget(self):
        member = self.envelope([("aaaa1111", 0.9), ("bbbb2222", 0.8)])
        legacy = self.envelope([("cccc3333", 0.7), ("dddd4444", 0.6)])
        merged = merge_recall_envelopes([member, legacy], limit=3)
        self.assertEqual(len(merged["items"]), 3)

    def test_merge_of_nothing_is_empty_not_an_error(self):
        merged = merge_recall_envelopes([], limit=5)
        self.assertEqual(merged["items"], [])
        self.assertEqual(merged["details"], [])


class FakeSenderEvent:
    """Minimal event surface for ``turn_from_event`` (no AstrBot import)."""

    def __init__(self, user_id, sender_name, group_id=GROUP, self_id=BOT):
        sender = types.SimpleNamespace(user_id=user_id)
        self.message_obj = types.SimpleNamespace(
            self_id=self_id,
            group_id=group_id,
            session_id=group_id,
            sender=sender,
        )
        self._name = sender_name

    def get_sender_name(self):
        return self._name


class TwoFakeSendersTests(unittest.TestCase):
    """End-to-end: two senders in one group each land in their own bucket."""

    def test_two_senders_each_get_their_own_bucket(self):
        config = {"scope_mode": "per_conversation"}
        alice = turn_from_event(FakeSenderEvent(ALICE, "阿鲤"), config)
        bob = turn_from_event(FakeSenderEvent(BOB, "老王"), config)

        # Same group, different members -> different write buckets.
        self.assertNotEqual(alice.scope, bob.scope)
        self.assertIn(":group:", alice.scope)
        self.assertIn(":group:", bob.scope)

        # Both reads still start from their own bucket...
        self.assertEqual(group_recall_scopes(alice)[0], alice.scope)
        self.assertEqual(group_recall_scopes(bob)[0], bob.scope)
        # ...and both still reach the shared legacy bucket, so neither loses
        # group history written before per-member scoping.
        self.assertEqual(group_recall_scopes(alice)[1], group_recall_scopes(bob)[1])
        self.assertEqual(
            group_recall_scopes(alice)[1], legacy_group_scope("per_conversation", BOT, GROUP)
        )

        # Attribution follows the sender.
        self.assertEqual(speaker_header(alice), "【发言人】阿鲤")
        self.assertEqual(speaker_header(bob), "【发言人】老王")
        self.assertNotEqual(sender_tag(alice), sender_tag(bob))

    def test_private_sender_is_unaffected(self):
        turn = turn_from_event(FakeSenderEvent(ALICE, "阿鲤", group_id=""), {"scope_mode": "per_conversation"})
        self.assertFalse(turn.is_group)
        self.assertEqual(group_recall_scopes(turn), [turn.scope])
        self.assertNotIn(":group:", turn.scope)


if __name__ == "__main__":
    unittest.main()
