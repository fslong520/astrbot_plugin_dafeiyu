"""Self-note tests: managed block, validation, generation and versioned apply."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from helpers import TempDirTest

from dafeiyu_config import resolve_config
from dafeiyu_rsrs import RsrsError
from dafeiyu_selfnote import (
    MAX_BRIEF,
    MAX_FULL,
    SELFNOTE_END,
    SELFNOTE_START,
    SelfNoteError,
    SelfNoteService,
    extract_managed_block,
    parse_candidate,
    replace_managed_block,
    strip_managed_block,
    validate_candidate,
)
from dafeiyu_state import StateStore

BASE = "底座规则：只调用 memory_recall 查证，不要读 data/memories。"


class FakePersona:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt


class FakePersonaManager:
    def __init__(self, system_prompt=BASE):
        self.prompt = system_prompt
        self.updates = 0
        self.update_error = None

    async def get_persona(self, persona_id="default"):
        return FakePersona(self.prompt)

    async def update_persona(self, persona_id, system_prompt=None):
        if self.update_error:
            raise self.update_error
        self.prompt = system_prompt
        self.updates += 1
        return True


class FakeProvider:
    def meta(self):
        return SimpleNamespace(id="fake-provider")


class FakeContext:
    def __init__(self, completion=None, persona_prompt=BASE, provider=True):
        self.persona_manager = FakePersonaManager(persona_prompt)
        self.completion = completion if completion is not None else json.dumps(
            {
                "full": "我说话简短直接，先查证再回答，和主人是长期搭档。",
                "brief": "简短直接，先查证再回答，与主人长期搭档。",
                "source_ids": ["92cfc0da"],
            },
            ensure_ascii=False,
        )
        self.llm_calls = []
        self._provider = FakeProvider() if provider else None

    async def llm_generate(self, chat_provider_id=None, prompt=None, system_prompt=None):
        self.llm_calls.append(
            {"provider": chat_provider_id, "prompt": prompt, "system_prompt": system_prompt}
        )
        return SimpleNamespace(completion_text=self.completion)

    async def get_using_provider_async(self):
        return self._provider


class FakeRsrs:
    learn_available = False

    def __init__(self, remember_error=None, show_body="FULL BODY", metadata_enabled=True):
        self.calls = []
        self.remember_error = remember_error
        self.show_body = show_body
        self.metadata_enabled = metadata_enabled
        self.recall_envelope = {
            "items": [
                {"name": "92cfc0da", "value": "0.51 主人喜欢先查证再回答"},
                {"name": "deadbeef", "value": "0.44 主人偏好简短直接的回复"},
            ]
        }
        self.list_envelope = {"details": []}

    async def run(self, args, *, timeout_ms=None, **kwargs):
        self.calls.append(list(args))
        command = args[0]
        if command == "recall":
            return {"envelope": self.recall_envelope, "code": 0}
        if command == "list":
            return {"envelope": self.list_envelope, "code": 0}
        if command == "remember":
            if self.remember_error:
                raise self.remember_error
            return {
                "envelope": {"status": "ok", "summary": {"id": "abcd1234-0000-0000-0000-000000000000"}},
                "code": 0,
            }
        if command == "show":
            return {
                "envelope": {"status": "ok", "details": {"entry": {"content": self.show_body}}},
                "code": 0,
            }
        return {"envelope": {"status": "ok"}, "code": 0}

    async def metadata_status(self, force=False):
        return {"enabled": self.metadata_enabled, "mode": "v1" if self.metadata_enabled else "isolated"}

    def args_for(self, command):
        return [call for call in self.calls if call and call[0] == command]


def make_config(tmp, **selfnote):
    payload = {
        "admin_ids": ["9001"],
        "scope_mode": "per_conversation",
        "state_dir": str(tmp / "state"),
        "selfnote": {"persona_id": "default", **selfnote},
    }
    return resolve_config(payload, env={"HOME": str(tmp)})


def make_turn(**overrides):
    values = {
        "self_id": "10001",
        "user_id": "5001",
        "group_id": "",
        "scope": "qq:test:self",
    }
    values.update(overrides)
    values["is_group"] = bool(values.get("group_id"))
    return SimpleNamespace(**values)


def make_service(tmp, *, context=None, rsrs=None, now=None, **selfnote):
    config = make_config(tmp, **selfnote)
    state = StateStore(config["state_dir"], debug=False)
    return (
        SelfNoteService(config, rsrs or FakeRsrs(), state, context or FakeContext(), now=now),
        state,
        config,
    )


class ManagedBlockTest(unittest.TestCase):
    def test_extract_without_block_returns_prompt(self):
        self.assertEqual(extract_managed_block(BASE), (BASE, "", ""))
        self.assertEqual(extract_managed_block(""), ("", "", ""))

    def test_replace_appends_with_a_blank_line(self):
        result = replace_managed_block(BASE, "新的沉淀")
        self.assertTrue(result.startswith(BASE + "\n\n"))
        self.assertIn(f"{SELFNOTE_START}\n新的沉淀\n{SELFNOTE_END}", result)
        self.assertEqual(result.count(SELFNOTE_START), 1)

    def test_replace_keeps_base_and_suffix(self):
        prompt = f"{BASE}\n\n{SELFNOTE_START}\n旧沉淀\n{SELFNOTE_END}\n\n尾部规则：不要泄露。"
        result = replace_managed_block(prompt, "新沉淀")
        self.assertIn(BASE, result)
        self.assertIn("尾部规则：不要泄露。", result)
        self.assertIn("新沉淀", result)
        self.assertNotIn("旧沉淀", result)
        self.assertEqual(result.count(SELFNOTE_START), 1)
        prefix, block, suffix = extract_managed_block(result)
        self.assertEqual(block, "新沉淀")
        self.assertTrue(prefix.startswith(BASE))
        self.assertTrue(suffix.strip().startswith("尾部规则"))

    def test_replace_rejects_invalid_brief(self):
        with self.assertRaises(SelfNoteError):
            replace_managed_block(BASE, "")
        with self.assertRaises(SelfNoteError):
            replace_managed_block(BASE, "x" * (MAX_BRIEF + 1))
        with self.assertRaises(SelfNoteError):
            replace_managed_block(BASE, f"{SELFNOTE_START} bad")
        with self.assertRaises(SelfNoteError):
            replace_managed_block(BASE, f"bad {SELFNOTE_END}")

    def test_orphan_or_multiple_blocks_fail_closed(self):
        orphan = f"{BASE}\n\n{SELFNOTE_START}\nold"
        with self.assertRaises(SelfNoteError) as caught:
            replace_managed_block(orphan, "new")
        self.assertIn("malformed", str(caught.exception))
        with self.assertRaises(SelfNoteError):
            strip_managed_block(orphan)

        one = replace_managed_block(BASE, "one")
        two = replace_managed_block(one, "two")
        doubled = f"{two}\n\n{replace_managed_block('', 'extra')}"
        with self.assertRaises(SelfNoteError) as caught:
            replace_managed_block(doubled, "three")
        self.assertIn("multiple", str(caught.exception))

        # Two starts with one end must not be rewritten as one block.
        double_start = f"{BASE}\n{SELFNOTE_START}\nA\n{SELFNOTE_START}\nB\n{SELFNOTE_END}"
        with self.assertRaises(SelfNoteError):
            replace_managed_block(double_start, "new")
        with self.assertRaises(SelfNoteError):
            extract_managed_block(double_start)

        # An orphan END before a complete block is also refused.
        before = f"{BASE}\n{SELFNOTE_END}\n{two}"
        with self.assertRaises(SelfNoteError):
            replace_managed_block(before, "four")
        with self.assertRaises(SelfNoteError):
            strip_managed_block(before)
        with self.assertRaises(SelfNoteError):
            extract_managed_block(before)

    def test_strip_removes_block_and_extra_blank_lines(self):
        prompt = f"{BASE}\n\n{SELFNOTE_START}\n沉淀\n{SELFNOTE_END}\n\n尾部规则"
        self.assertEqual(strip_managed_block(prompt), f"{BASE}\n\n尾部规则")
        self.assertEqual(strip_managed_block(BASE), BASE)
        # A start marker without an end marker is refused, never half-deleted.
        broken = f"{BASE}\n{SELFNOTE_START}\n沉淀"
        with self.assertRaises(SelfNoteError):
            strip_managed_block(broken)


class ValidationTest(unittest.TestCase):
    def test_length_and_secret_limits(self):
        self.assertEqual(validate_candidate("full", "brief", ["92cfc0da"], ["92cfc0da"]), (True, "ok"))
        self.assertFalse(validate_candidate("", "brief", ["92cfc0da"], ["92cfc0da"])[0])
        self.assertFalse(validate_candidate("x" * (MAX_FULL + 1), "brief", ["92cfc0da"], ["92cfc0da"])[0])
        self.assertFalse(validate_candidate("full", "", ["92cfc0da"], ["92cfc0da"])[0])
        self.assertFalse(validate_candidate("full", "x" * (MAX_BRIEF + 1), ["92cfc0da"], ["92cfc0da"])[0])
        self.assertFalse(validate_candidate("full api_key 123", "brief", ["92cfc0da"], ["92cfc0da"])[0])
        self.assertFalse(validate_candidate("full", "密码 123", ["92cfc0da"], ["92cfc0da"])[0])
        self.assertFalse(validate_candidate(f"{SELFNOTE_START}", "brief", ["92cfc0da"], ["92cfc0da"])[0])

    def test_configured_caps_are_clamped_and_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _, _ = make_service(Path(tmp), max_brief_chars=80)
        self.assertEqual(service.max_brief, 80)
        brief = "x" * 81
        ok, reason = validate_candidate(
            "F", brief, ["92cfc0da"], ["92cfc0da"], service.max_full, service.max_brief
        )
        self.assertFalse(ok)
        self.assertIn("brief exceeds", reason)

    def test_source_subset(self):
        allowed = ["92cfc0da", "deadbeef-1111-2222-3333-444444444444"]
        self.assertTrue(validate_candidate("full", "brief", allowed, ["92cfc0da"])[0])
        self.assertTrue(validate_candidate("full", "brief", allowed, ["deadbeef"])[0])
        self.assertFalse(validate_candidate("full", "brief", allowed, ["12345678"])[0])
        self.assertFalse(validate_candidate("full", "brief", allowed, [])[0])
        self.assertFalse(validate_candidate("full", "brief", allowed, ["92cfc0d"])[0])
        # Manual mode: allowed_ids=None means no evidence, so no sources.
        self.assertTrue(validate_candidate("full", "brief", None, [])[0])
        self.assertFalse(validate_candidate("full", "brief", None, ["92cfc0da"])[0])

    def test_parse_tolerates_fences_and_prose(self):
        text = (
            "好的，结果如下：\n```json\n"
            '{"full": "F", "brief": "B", "source_ids": "92cfc0da, deadbeef"}\n'
            "```\n希望有帮助"
        )
        parsed = parse_candidate(text)
        self.assertEqual(parsed["full"], "F")
        self.assertEqual(parsed["brief"], "B")
        self.assertEqual(parsed["source_ids"], ["92cfc0da", "deadbeef"])
        with self.assertRaises(ValueError):
            parse_candidate("no json here")
        with self.assertRaises(ValueError):
            parse_candidate("{broken")


class ServiceGenerateTest(TempDirTest):
    def test_generate_reads_owner_scope_and_writes_nothing(self):
        context = FakeContext()
        service, state, config = make_service(self.tmp, context=context)
        result = asyncio.run(service.generate(make_turn()))
        self.assertEqual(result["brief"], "简短直接，先查证再回答，与主人长期搭档。")
        self.assertEqual(result["source_ids"], ["92cfc0da"])
        self.assertEqual(result["provider_id"], "fake-provider")
        self.assertEqual(result["evidence_ids"], ["92cfc0da", "deadbeef"])

        recall = service._rsrs.args_for("recall")[0]
        self.assertIn("--project qq:", " ".join(recall))
        # Evidence comes from the owner's private scope, never a raw QQ id.
        project = recall[recall.index("--project") + 1]
        self.assertIn(":private:", project)
        self.assertNotIn("9001", project)
        # The prompt carries titles only and the JSON contract is the system prompt.
        self.assertIn("[92cfc0da]", context.llm_calls[0]["prompt"])
        self.assertIn("source_ids", context.llm_calls[0]["system_prompt"])
        self.assertIn("不可信历史文本", context.llm_calls[0]["prompt"])
        self.assertIn("不可信历史文本", context.llm_calls[0]["system_prompt"])
        self.assertEqual(context.persona_manager.updates, 0)
        self.assertIsNone(state.read_selfnote(service.build_scope(make_turn())))
        self.assertEqual(service._rsrs.args_for("remember"), [])

    def test_evidence_prefers_important_owner_records(self):
        rsrs = FakeRsrs()
        service, _, _ = make_service(self.tmp, rsrs=rsrs)
        owner = service._evidence_scope(make_turn())
        rsrs.list_envelope = {
            "details": [
                {
                    "id": "11112222-3333-4444-5555-666666666666",
                    "project": owner,
                    "importance": "important",
                    "kind": "Context",
                    "title": "主人喜欢蓝色",
                },
                {
                    "id": "22223333-3333-4444-5555-666666666666",
                    "project": owner,
                    "importance": "trivial",
                    "kind": "Context",
                    "title": "随便聊聊",
                },
                {
                    "id": "33334444-3333-4444-5555-666666666666",
                    "project": "other",
                    "importance": "important",
                    "kind": "Context",
                    "title": "别人的事实",
                },
            ]
        }
        evidence = asyncio.run(service.evidence(make_turn()))
        self.assertEqual(evidence, [{"id": "11112222", "title": "主人喜欢蓝色"}])
        self.assertEqual(rsrs.args_for("recall"), [])

    def test_generate_requires_evidence_and_provider(self):
        service, _, _ = make_service(self.tmp, context=FakeContext(provider=False))
        with self.assertRaises(SelfNoteError) as caught:
            asyncio.run(service.generate(make_turn()))
        self.assertIn("provider unavailable", str(caught.exception))

        rsrs = FakeRsrs()
        rsrs.recall_envelope = {"items": []}
        service, _, _ = make_service(self.tmp, rsrs=rsrs)
        with self.assertRaises(SelfNoteError) as caught:
            asyncio.run(service.generate(make_turn()))
        self.assertIn("no evidence", str(caught.exception))

    def test_generate_rejects_unparseable_or_invalid_output(self):
        service, _, _ = make_service(self.tmp, context=FakeContext(completion="not json"))
        with self.assertRaises(SelfNoteError) as caught:
            asyncio.run(service.generate(make_turn()))
        self.assertIn("candidate parse failed", str(caught.exception))

        bad = json.dumps({"full": "F", "brief": "B", "source_ids": ["12345678"]})
        service, _, _ = make_service(self.tmp, context=FakeContext(completion=bad))
        with self.assertRaises(SelfNoteError) as caught:
            asyncio.run(service.generate(make_turn()))
        self.assertIn("not in evidence", str(caught.exception))


class ServiceApplyTest(TempDirTest):
    def turn(self):
        return make_turn()

    def test_apply_versions_memory_persona_and_state(self):
        service, state, _ = make_service(self.tmp)
        scope = service.build_scope(self.turn())
        state.write_selfnote(
            scope,
            {"id": "11112222-3333-4444-5555-666666666666", "brief": "旧", "updated_at": "t", "sources": []},
        )
        candidate = {
            "full": "新的完整沉淀，只写风格与关系。",
            "brief": "新的简短沉淀。",
            "source_ids": ["92cfc0da"],
        }
        result = asyncio.run(service.apply(self.turn(), candidate))
        self.assertEqual(result["supersedes"], "11112222-3333-4444-5555-666666666666")
        self.assertTrue(result["persona_updated"])

        remember = service._rsrs.args_for("remember")[0]
        self.assertIn("--type self", " ".join(remember))
        self.assertIn("--importance important", " ".join(remember))
        self.assertIn("--tags selfnote,persona", " ".join(remember))
        self.assertIn("--project qq:", " ".join(remember))
        self.assertIn(":selfnote", remember[remember.index("--project") + 1])
        self.assertIn("--supersedes 11112222-3333-4444-5555-666666666666", " ".join(remember))
        # T1-3: the typed self-note write carries a write-time summary.
        self.assertIn("--summary", remember)
        self.assertNotIn("--force", remember)
        self.assertIn("92cfc0da", remember[1])

        prompt = service._context.persona_manager.prompt
        self.assertEqual(prompt.count(SELFNOTE_START), 1)
        _, block, _ = extract_managed_block(prompt)
        self.assertEqual(block, "新的简短沉淀。")
        self.assertTrue(prompt.startswith(BASE))

        record = state.read_selfnote(scope)
        self.assertEqual(record["id"], "abcd1234-0000-0000-0000-000000000000")
        self.assertEqual(record["brief"], "新的简短沉淀。")
        self.assertEqual(record["sources"], ["92cfc0da"])
        self.assertTrue(record["updated_at"])

    def test_legacy_store_uses_tags_parent_and_no_typed_args(self):
        rsrs = FakeRsrs(metadata_enabled=False)
        service, state, _ = make_service(
            self.tmp,
            rsrs=rsrs,
            parent_id="a80729d1-e98d-43da-a51b-a601cb09d859",
        )
        state.write_selfnote(
            service.build_scope(self.turn()),
            {"id": "11112222-3333-4444-5555-666666666666", "brief": "旧", "updated_at": "t", "sources": []},
        )
        result = asyncio.run(
            service.apply(
                self.turn(),
                {
                    "full": "F",
                    "brief": "B",
                    "source_ids": ["92cfc0da"],
                    "evidence_ids": ["92cfc0da"],
                },
            )
        )
        self.assertTrue(result["persona_updated"])
        args = " ".join(service._rsrs.args_for("remember")[0])
        self.assertNotIn("--type", args)
        self.assertNotIn("--supersedes", args)
        # T1-3: --summary is contract-1; the legacy write must not carry it.
        self.assertNotIn("--summary", args)
        self.assertIn("--importance important", args)
        self.assertIn("--parent a80729d1-e98d-43da-a51b-a601cb09d859", args)
        self.assertIn("--tags selfnote,persona", args)
        self.assertIn("【版本】替代 11112222", args)

    def test_apply_prefers_store_truth_over_a_stale_pointer(self):
        service, state, _ = make_service(self.tmp)
        scope = service.build_scope(self.turn())
        state.write_selfnote(
            scope,
            {"id": "11112222-3333-4444-5555-666666666666", "brief": "旧", "updated_at": "t", "sources": []},
        )
        service._rsrs.list_envelope = {
            "details": [
                {
                    "id": "99990000-3333-4444-5555-666666666666",
                    "project": scope,
                    "title": "selfnote persona",
                    "created_at": "2026-11-03T00:00:00Z",
                    "superseded_by": "",
                }
            ]
        }
        candidate = {
            "full": "新的完整沉淀。",
            "brief": "新的简短沉淀。",
            "source_ids": ["92cfc0da"],
            "evidence_ids": ["92cfc0da"],
        }
        result = asyncio.run(service.apply(self.turn(), candidate))
        self.assertEqual(result["supersedes"], "99990000-3333-4444-5555-666666666666")
        remember = service._rsrs.args_for("remember")[0]
        self.assertIn("--supersedes 99990000-3333-4444-5555-666666666666", " ".join(remember))

    def test_apply_failure_leaves_the_old_persona_and_state(self):
        rsrs = FakeRsrs(remember_error=RsrsError("store unavailable"))
        service, state, _ = make_service(self.tmp, rsrs=rsrs)
        before = service._context.persona_manager.prompt
        with self.assertRaises(SelfNoteError) as caught:
            asyncio.run(
                service.apply(
                    self.turn(),
                    {"full": "F", "brief": "B", "source_ids": ["92cfc0da"]},
                )
            )
        self.assertIn("remember failed", str(caught.exception))
        self.assertEqual(service._context.persona_manager.prompt, before)
        self.assertIsNone(state.read_selfnote(service.build_scope(self.turn())))
        self.assertEqual(service._context.persona_manager.updates, 0)

    def test_persona_write_failure_leaves_the_old_block(self):
        service, state, _ = make_service(self.tmp)
        # Seed a valid existing block; the failed write must not corrupt it.
        service._context.persona_manager.prompt = replace_managed_block(BASE, "旧沉淀")
        service._context.persona_manager.update_error = RuntimeError("db locked")
        before = service._context.persona_manager.prompt
        with self.assertRaises(SelfNoteError) as caught:
            asyncio.run(
                service.apply(
                    self.turn(),
                    {"full": "F", "brief": "新", "source_ids": ["92cfc0da"]},
                )
            )
        self.assertIn("persona write failed", str(caught.exception))
        self.assertEqual(service._context.persona_manager.prompt, before)
        self.assertIsNone(state.read_selfnote(service.build_scope(self.turn())))

    def test_state_write_failure_is_reported_as_pending(self):
        service, state, _ = make_service(self.tmp)
        scope = service.build_scope(self.turn())

        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        state.write_selfnote = fail
        result = asyncio.run(
            service.apply(
                self.turn(),
                {
                    "full": "F",
                    "brief": "B",
                    "source_ids": ["92cfc0da"],
                    "evidence_ids": ["92cfc0da"],
                },
            )
        )
        self.assertFalse(result["state_saved"])
        self.assertIn("B", service._context.persona_manager.prompt)

    def test_next_apply_after_persona_failure_supersedes_the_written_record(self):
        service, state, _ = make_service(self.tmp)
        scope = service.build_scope(self.turn())
        service._context.persona_manager.update_error = RuntimeError("db locked")
        candidate = {
            "full": "F",
            "brief": "B",
            "source_ids": ["92cfc0da"],
            "evidence_ids": ["92cfc0da"],
        }
        with self.assertRaises(SelfNoteError):
            asyncio.run(service.apply(self.turn(), candidate))
        self.assertIsNone(state.read_selfnote(scope))

        service._context.persona_manager.update_error = None
        service._rsrs.list_envelope = {
            "details": [
                {
                    "id": "abcd1234-0000-0000-0000-000000000000",
                    "project": scope,
                    "title": "selfnote persona",
                    "created_at": "2026-11-03T00:00:00Z",
                    "superseded_by": "",
                }
            ]
        }
        asyncio.run(service.apply(self.turn(), candidate))
        second = service._rsrs.args_for("remember")[1]
        self.assertIn("--supersedes abcd1234-0000-0000-0000-000000000000", " ".join(second))

    def test_apply_rejects_a_source_outside_evidence(self):
        service, _, _ = make_service(self.tmp)
        with self.assertRaises(SelfNoteError) as caught:
            asyncio.run(
                service.apply(
                    self.turn(),
                    {"full": "F", "brief": "B", "source_ids": ["12345678"]},
                )
            )
        self.assertIn("not in evidence", str(caught.exception))
        self.assertEqual(service._rsrs.args_for("remember"), [])

    def test_set_is_manual_and_cites_nothing(self):
        service, state, _ = make_service(self.tmp)
        result = asyncio.run(service.set(self.turn(), "手动设定：说话更温和。"))
        self.assertEqual(result["brief"], "手动设定：说话更温和。")
        remember = service._rsrs.args_for("remember")[0]
        self.assertNotIn("--supersedes", remember)
        self.assertIn("主人手动设定", remember[1])
        record = state.read_selfnote(service.build_scope(self.turn()))
        self.assertEqual(record["sources"], [])

    def test_clear_removes_block_but_keeps_memories(self):
        service, state, _ = make_service(self.tmp)
        prompt = replace_managed_block(BASE, "旧沉淀")
        service._context.persona_manager.prompt = prompt
        state.write_selfnote(
            service.build_scope(self.turn()),
            {"id": "abcd1234-0000-0000-0000-000000000000", "brief": "旧沉淀"},
        )
        removed = asyncio.run(service.clear(self.turn()))
        self.assertTrue(removed)
        self.assertEqual(service._context.persona_manager.prompt, BASE)
        self.assertIsNone(state.read_selfnote(service.build_scope(self.turn())))
        # Never forgets a memory.
        self.assertEqual(service._rsrs.args_for("forget"), [])
        # Clearing again is a no-op.
        self.assertFalse(asyncio.run(service.clear(self.turn())))

    def test_show_reads_brief_pointer_and_full_body(self):
        service, state, _ = make_service(self.tmp)
        state.write_selfnote(
            service.build_scope(self.turn()),
            {
                "id": "abcd1234-0000-0000-0000-000000000000",
                "brief": "摘要",
                "updated_at": "2026-01-01T00:00:00Z",
                "sources": ["92cfc0da"],
            },
        )
        data = asyncio.run(service.show(self.turn()))
        self.assertEqual(data["brief"], "摘要")
        self.assertEqual(data["full"], "FULL BODY")
        self.assertEqual(data["id"], "abcd1234-0000-0000-0000-000000000000")
        self.assertEqual(data["sources"], ["92cfc0da"])
        self.assertEqual(service._rsrs.args_for("show")[0][0], "show")

    def test_show_falls_back_to_the_persona_block(self):
        service, _, _ = make_service(self.tmp)
        service._context.persona_manager.prompt = replace_managed_block(BASE, "手工块")
        data = asyncio.run(service.show(self.turn()))
        self.assertEqual(data["brief"], "手工块")
        self.assertEqual(data["id"], "")
        self.assertEqual(data["full"], "")


BASE_MS = 1_700_000_000_000


def iso(ms):
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AutoClock:
    def __init__(self, value=BASE_MS):
        self.value = value

    def __call__(self):
        return self.value


class AutoDueTest(TempDirTest):
    def make(self, **selfnote):
        self.clock = AutoClock()
        return make_service(self.tmp, now=self.clock, **selfnote)

    def seed(self, state, scope, **fields):
        record = {
            "id": "abcd1234-0000-0000-0000-000000000000",
            "brief": "旧沉淀",
            "updated_at": "2026-01-01T00:00:00Z",
            "sources": ["92cfc0da"],
        }
        record.update(fields)
        state.write_selfnote(scope, record)

    def test_disabled_and_group_never_due(self):
        service, state, _ = self.make()
        turn = make_turn()
        self.assertFalse(service.auto_enabled)
        self.assertEqual(service.auto_due(turn), (False, "auto disabled"))
        self.assertIsNone(service.note_turn(turn))
        self.assertIsNone(state.read_selfnote(service.build_scope(turn)))

        service, state, _ = self.make(auto_enabled=True, every_sessions=5)
        group = make_turn(group_id="777888")
        self.assertTrue(service.auto_enabled)
        self.assertEqual(service.auto_due(group), (False, "group chat"))
        self.assertIsNone(service.note_turn(group))
        self.assertIsNone(state.read_selfnote(service.build_scope(group)))

    def test_threshold_and_min_interval(self):
        service, state, _ = self.make(auto_enabled=True, every_sessions=5, min_interval_ms=3600000)
        turn = make_turn()
        scope = service.build_scope(turn)
        for _ in range(4):
            service.note_turn(turn)
        self.assertEqual(service.auto_due(turn), (False, "turns 4/5"))
        service.note_turn(turn)
        # No previous automatic attempt: due right away.
        self.assertEqual(service.auto_due(turn), (True, "due"))

        self.seed(state, scope, auto_turns=5, last_auto_at=iso(BASE_MS - 600_000))
        self.assertEqual(service.auto_due(turn), (False, "min interval"))
        self.seed(state, scope, auto_turns=5, last_auto_at=iso(BASE_MS - 7_200_000))
        self.assertEqual(service.auto_due(turn), (True, "due"))

    def test_failure_backoff_replaces_the_interval(self):
        service, state, _ = self.make(
            auto_enabled=True, every_sessions=5, min_interval_ms=3600000, retry_backoff_ms=21600000
        )
        turn = make_turn()
        scope = service.build_scope(turn)
        self.seed(
            state,
            scope,
            auto_turns=9,
            last_auto_at=iso(BASE_MS - 7_200_000),
            auto_failures=2,
        )
        self.assertEqual(service.auto_due(turn), (False, "retry backoff"))
        self.seed(
            state,
            scope,
            auto_turns=9,
            last_auto_at=iso(BASE_MS - 25200000),
            auto_failures=2,
        )
        self.assertEqual(service.auto_due(turn), (True, "due"))
        # An unreadable stamp with a failure history must not hammer the store.
        self.seed(state, scope, auto_turns=9, last_auto_at="not-a-time", auto_failures=1)
        self.assertEqual(service.auto_due(turn), (False, "retry backoff"))

    def test_auto_status_reports_progress_and_next_due(self):
        service, state, _ = self.make(auto_enabled=True, every_sessions=5, min_interval_ms=3600000)
        turn = make_turn()
        scope = service.build_scope(turn)
        status = service.auto_status(turn)
        self.assertEqual(status["enabled"], True)
        self.assertEqual(status["turns"], 0)
        self.assertEqual(status["every_sessions"], 5)
        self.assertEqual(status["next_due_at"], "")

        self.seed(state, scope, auto_turns=5, last_auto_at=iso(BASE_MS - 600_000))
        status = service.auto_status(turn)
        self.assertEqual(status["turns"], 5)
        self.assertEqual(status["last_auto_at"], iso(BASE_MS - 600_000))
        self.assertEqual(status["next_due_at"], iso(BASE_MS + 3000000))

        off = make_service(self.tmp / "off", now=self.clock)[0]
        self.assertEqual(off.auto_status(turn)["enabled"], False)


class MaybeAutoTest(TempDirTest):
    def make(self, **selfnote):
        self.clock = AutoClock()
        context = FakeContext()
        rsrs = selfnote.pop("rsrs", None) or FakeRsrs()
        service, state, _ = make_service(self.tmp, context=context, rsrs=rsrs, now=self.clock, **selfnote)
        return service, state, context, rsrs

    def ready(self, service, turn, count=5):
        for _ in range(count):
            service.note_turn(turn)
        # No prior automatic attempt, so the interval gate is already open.

    def test_success_applies_once_and_resets_the_counter(self):
        service, state, context, rsrs = self.make(auto_enabled=True, every_sessions=5)
        turn = make_turn()
        scope = service.build_scope(turn)
        self.ready(service, turn)
        result = asyncio.run(service.maybe_auto(turn))
        self.assertEqual(result["ran"], True)
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["reason"], "ok")
        self.assertEqual(context.persona_manager.updates, 1)
        self.assertEqual(len(rsrs.args_for("remember")), 1)
        record = state.read_selfnote(scope)
        self.assertEqual(record["auto_turns"], 0)
        self.assertEqual(record["auto_failures"], 0)
        self.assertTrue(record["last_auto_at"])
        self.assertEqual(record["id"], "abcd1234-0000-0000-0000-000000000000")
        self.assertIn("简短直接", context.persona_manager.prompt)
        # Nothing is due right after a success.
        self.assertEqual(service.auto_due(turn), (False, "turns 0/5"))

    def test_failure_is_counted_and_never_raises(self):
        service, state, context, rsrs = self.make(
            auto_enabled=True, every_sessions=5, rsrs=FakeRsrs(remember_error=RsrsError("store unavailable"))
        )
        turn = make_turn()
        scope = service.build_scope(turn)
        before_prompt = context.persona_manager.prompt
        self.ready(service, turn)
        result = asyncio.run(service.maybe_auto(turn))
        self.assertEqual(result["ran"], True)
        self.assertEqual(result["ok"], False)
        self.assertIn("remember failed", result["reason"])
        # Old persona and pointer untouched; only the failure counter moves.
        self.assertEqual(context.persona_manager.prompt, before_prompt)
        self.assertEqual(context.persona_manager.updates, 0)
        record = state.read_selfnote(scope)
        self.assertEqual(record["id"], "")
        self.assertEqual(record["auto_turns"], 5)
        self.assertEqual(record["auto_failures"], 1)
        self.assertTrue(record["last_auto_at"])
        self.assertEqual(service.auto_due(turn), (False, "retry backoff"))

    def test_generate_failure_is_counted_and_never_raises(self):
        rsrs = FakeRsrs()
        rsrs.recall_envelope = {"items": []}
        rsrs.list_envelope = {"details": []}
        service, state, context, _ = self.make(auto_enabled=True, every_sessions=5, rsrs=rsrs)
        turn = make_turn()
        self.ready(service, turn)
        result = asyncio.run(service.maybe_auto(turn))
        self.assertEqual(result["ok"], False)
        self.assertIn("no evidence", result["reason"])
        self.assertEqual(context.persona_manager.updates, 0)
        self.assertEqual(state.read_selfnote(service.build_scope(turn))["auto_failures"], 1)

    def test_not_due_writes_nothing(self):
        service, state, context, rsrs = self.make(auto_enabled=True, every_sessions=5)
        turn = make_turn()
        service.note_turn(turn)
        service.note_turn(turn)
        result = asyncio.run(service.maybe_auto(turn))
        self.assertEqual(result["ran"], False)
        self.assertEqual(result["reason"], "turns 2/5")
        self.assertEqual(context.persona_manager.updates, 0)
        self.assertEqual(rsrs.args_for("remember"), [])
        self.assertEqual(state.read_selfnote(service.build_scope(turn))["auto_turns"], 2)

    def test_disabled_never_generates(self):
        service, state, context, rsrs = self.make()
        turn = make_turn()
        state.bump_selfnote_turn(service.build_scope(turn))
        result = asyncio.run(service.maybe_auto(turn))
        self.assertEqual(result, {"ran": False, "ok": None, "reason": "auto disabled"})
        self.assertEqual(context.persona_manager.updates, 0)
        self.assertEqual(rsrs.args_for("remember"), [])

    def test_concurrent_calls_run_only_once(self):
        service, state, context, rsrs = self.make(auto_enabled=True, every_sessions=5)
        turn = make_turn()
        self.ready(service, turn)
        calls = {"n": 0}
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_generate(_turn):
            calls["n"] += 1
            started.set()
            await release.wait()
            return {
                "full": "F",
                "brief": "B",
                "source_ids": ["92cfc0da"],
                "evidence_ids": ["92cfc0da"],
            }

        service.generate = slow_generate

        async def go():
            first = asyncio.create_task(service.maybe_auto(turn))
            await started.wait()
            second = await service.maybe_auto(turn)
            release.set()
            return await first, second

        first, second = asyncio.run(go())
        self.assertEqual(first["ran"], True)
        self.assertEqual(first["ok"], True)
        self.assertEqual(second["ran"], False)
        self.assertEqual(second["reason"], "inflight")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(rsrs.args_for("remember")), 1)
        self.assertEqual(context.persona_manager.updates, 1)


if __name__ == "__main__":
    unittest.main()
