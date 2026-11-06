"""Runtime tests with stub AstrBot modules and fake events/requests.

No real AstrBot is imported: the ``astrbot.*`` modules are injected into
``sys.modules`` before ``main`` is imported, and the rsrs CLI is a shell script.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
import unittest

from helpers import TempDirTest

RECALL_PAYLOAD = {
    "command": "recall",
    "status": "ok",
    "summary": {"best_cos": 0.57, "best_pre_floor": 0.348, "query_log_id": "uuid-recall-1", "count": 1},
    "items": [{"name": "92cfc0da", "status": "ok", "value": "0.348 first remembered title"}],
    "details": [{"id": "92cfc0da-1111-2222-3333-444444444444", "title": "first remembered title", "score": 0.348}],
    "errors": [],
}

SHOW_PAYLOAD = {
    "command": "show",
    "status": "ok",
    "summary": {"id": "92cfc0da-2845-4082-8d0b-0a7b8b2335f3", "title": "first remembered title"},
    "items": [{"name": "entry", "status": "ok", "value": "92cfc0da"}],
    "details": {"entry": {"content": "BODY_SHOULD_NOT_LEAK", "title": "first remembered title"}},
    "errors": [],
}

REMEMBER_PAYLOAD = {
    "command": "remember",
    "status": "ok",
    "summary": {"id": "abcd1234-0000-0000-0000-000000000000", "title": "QQ fact: note"},
    "items": [],
    "errors": [],
}

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
    },
    # Must never be echoed by /mem_audit or the memory_audit tool.
    "items": [],
    "details": {
        "hot_untyped": [{"id": "deadbeef", "title": "DETAIL_TITLE_SHOULD_NOT_LEAK"}],
        "hot_untyped_skipped": 0,
    },
    "errors": [],
}

REVIEW_PAYLOAD = {
    "command": "review",
    "status": "ok",
    "summary": {"status": "due", "due": 1},
    "items": [{"name": "92cfc0da", "value": "first remembered title"}],
    "details": [{"id": "92cfc0da-1111-2222-3333-444444444444", "title": "first remembered title"}],
    "errors": [],
}


def _install_astrbot_stubs():
    def _log(*args, **kwargs):
        return None

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = types.SimpleNamespace(warning=_log, info=_log, error=_log, debug=_log)
    api.AstrBotConfig = dict

    event_mod = types.ModuleType("astrbot.api.event")

    class _Filter:
        class PermissionType:
            ADMIN = "admin"

        def command(self, name=None, **kwargs):
            def decorate(func):
                func.__command_name__ = name
                return func

            return decorate

        def permission_type(self, kind):
            def decorate(func):
                func.__permission__ = kind
                return func

            return decorate

        def _hook(self, func):
            return func

        def on_llm_request(self):
            return self._hook

        def on_llm_response(self):
            return self._hook

        def on_astrbot_loaded(self):
            return self._hook

    class AstrMessageEvent:
        pass

    event_mod.filter = _Filter()
    event_mod.AstrMessageEvent = AstrMessageEvent

    star_mod = types.ModuleType("astrbot.api.star")

    class Context:
        pass

    class Star:
        def __init__(self, context):
            self.context = context

    star_mod.Context = Context
    star_mod.Star = Star

    message_mod = types.ModuleType("astrbot.core.agent.message")

    class TextPart:
        def __init__(self, text=""):
            self.text = text
            self.temp = False

        def mark_as_temp(self):
            clone = TextPart(self.text)
            clone.temp = True
            return clone

    message_mod.TextPart = TextPart

    path_mod = types.ModuleType("astrbot.core.utils.astrbot_path")
    holder = {"path": None}

    def get_astrbot_data_path():
        return holder["path"]

    path_mod.get_astrbot_data_path = get_astrbot_data_path
    path_mod.HOLDER = holder

    packages = {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event_mod,
        "astrbot.api.star": star_mod,
        "astrbot.core": types.ModuleType("astrbot.core"),
        "astrbot.core.agent": types.ModuleType("astrbot.core.agent"),
        "astrbot.core.agent.message": message_mod,
        "astrbot.core.utils": types.ModuleType("astrbot.core.utils"),
        "astrbot.core.utils.astrbot_path": path_mod,
    }
    for name, module in packages.items():
        sys.modules.setdefault(name, module)
    return holder


ASTROBOT_PATH = _install_astrbot_stubs()

import main as plugin_main  # noqa: E402  (import after stub injection)


class FakeSender:
    def __init__(self, user_id):
        self.user_id = user_id


class FakeMessageObj:
    def __init__(self, self_id, group_id, user_id, session_id):
        self.self_id = self_id
        self.group_id = group_id
        self.session_id = session_id
        self.sender = FakeSender(user_id)


class FakeEvent:
    def __init__(self, text, *, self_id="10001", group_id="", user_id="5001"):
        self.message_str = text
        self.message_obj = FakeMessageObj(
            self_id, group_id, user_id, f"aiocqhttp:{'Group' if group_id else 'Friend'}Message:{group_id or user_id}"
        )

    def plain_result(self, text):
        return ("plain", text)

    def get_sender_name(self):
        return "tester"

    def get_sender_id(self):
        return str(self.message_obj.sender.user_id)


class FakeReq:
    def __init__(self, system_prompt="sys", with_parts=True):
        self.system_prompt = system_prompt
        self.extra_user_content_parts = [] if with_parts else None


class FakeResp:
    def __init__(self, text):
        self.completion_text = text


class FakePersona:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt


class FakePersonaManager:
    def __init__(self, system_prompt="base persona rules: always verify before answering."):
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
        return types.SimpleNamespace(id="fake-provider")


SELFNOTE_COMPLETION = json.dumps(
    {
        "full": "我说话简短直接，先查证再回答，和主人是长期搭档。",
        "brief": "简短直接，先查证再回答，与主人长期搭档。",
        "source_ids": ["92cfc0da"],
    },
    ensure_ascii=False,
)


class FakeContext:
    """Minimal persona_manager + llm_generate surface used by SelfNoteService."""

    def __init__(self, completion=None, persona_prompt=None, provider=True):
        self.persona_manager = FakePersonaManager(
            persona_prompt
            if persona_prompt is not None
            else "base persona rules: always verify before answering."
        )
        self.completion = completion if completion is not None else SELFNOTE_COMPLETION
        self.llm_calls = []
        self._provider = FakeProvider() if provider else None

    async def llm_generate(self, chat_provider_id=None, prompt=None, system_prompt=None):
        self.llm_calls.append(
            {"provider": chat_provider_id, "prompt": prompt, "system_prompt": system_prompt}
        )
        return FakeResp(self.completion)

    async def get_using_provider_async(self):
        return self._provider


def _collect(agen):
    async def go():
        return [item async for item in agen]

    return asyncio.run(go())


class Harness(TempDirTest):
    learn_available = False
    #: Overridable fake-rsrs payloads so subclasses can flip the metadata gate
    #: or serve a richer ``show`` entry without rewriting the script.
    metadata_payload = {"status": "ok", "summary": {"enabled": True, "mode": "v1"}}
    show_payload = SHOW_PAYLOAD

    def setUp(self):
        super().setUp()
        ASTROBOT_PATH["path"] = str(self.tmp / "astrbot-data")
        (self.tmp / "astrbot-data").mkdir(parents=True, exist_ok=True)
        self.calls = self.tmp / "calls"
        self.calls.mkdir()
        self.binary = self._make_binary()
        self.plugin = self._make_plugin()

    def _make_binary(self):
        recall = json.dumps(json.dumps(RECALL_PAYLOAD))
        show = json.dumps(json.dumps(self.show_payload))
        remember = json.dumps(json.dumps(REMEMBER_PAYLOAD))
        audit = json.dumps(json.dumps(AUDIT_PAYLOAD))
        review = json.dumps(json.dumps(REVIEW_PAYLOAD))
        metadata = json.dumps(json.dumps(self.metadata_payload))
        ok = json.dumps(json.dumps({"command": "ok", "status": "ok", "summary": {}, "items": [], "errors": []}))
        learn_exit = 0 if self.learn_available else 2
        learn_payload = json.dumps(json.dumps({"status": "ok", "summary": {"bge_available": False}}))
        script = f"""#!/bin/sh
printf 'CMD=%s\\n' "$1" > {json.dumps(str(self.calls))}/$$.log
printf 'ARGS=' >> {json.dumps(str(self.calls))}/$$.log
printf '%s ' "$@" | tr '\\n' ' ' >> {json.dumps(str(self.calls))}/$$.log
printf '\\n' >> {json.dumps(str(self.calls))}/$$.log
case "$1" in
  recall) printf '%s' {recall} ;;
  show) printf '%s' {show} ;;
  remember) printf '%s' {remember} ;;
  audit) printf '%s' {audit} ;;
  review) printf '%s' {review} ;;
  metadata) printf '%s' {metadata} ;;
  learn) printf '%s' {learn_payload}; exit {learn_exit} ;;
  *) printf '%s' {ok} ;;
esac
exit 0
"""
        path = self.tmp / "rsrs"
        path.write_text(script, encoding="utf-8")
        os.chmod(path, 0o755)
        return path

    def _make_plugin(self, context=None, **overrides):
        config = {
            "rsrs_bin": str(self.binary),
            "recall_timeout_ms": 5000,
            "cooldown_ms": 0,
            "same_query_cooldown_ms": 0,
            "admin_ids": ["9001"],
            "auto_diary": True,
            "auto_diary_group": False,
            "debug_log": True,
            # Point every job path at the temp dir so tests never touch a real repo.
            "jobs": {
                "enabled": False,
                "respire_repo": str(self.tmp / "no-repo"),
                "granite_model_dir": str(self.tmp / "no-model"),
                "anchor_dir": str(self.tmp / "no-anchor"),
            },
        }
        config.update(overrides)
        return plugin_main.DafeiyuPlugin(
            context=context if context is not None else FakeContext(), config=config
        )

    # -- helpers -----------------------------------------------------------
    def call_lines(self):
        if not self.calls.is_dir():
            return []
        files = sorted(self.calls.glob("*.log"), key=lambda item: (item.stat().st_mtime_ns, item.name))
        lines = []
        for path in files:
            lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line)
        return lines

    def commands(self):
        return [line[len("CMD="):] for line in self.call_lines() if line.startswith("CMD=")]

    def args_for(self, command):
        lines = self.call_lines()
        out = []
        for index, line in enumerate(lines):
            if line == f"CMD={command}" and index + 1 < len(lines):
                out.append(lines[index + 1][len("ARGS="):].strip())
        return out

    async def _drain(self):
        await self._drain_for(self.plugin)

    async def _drain_for(self, plugin):
        for _ in range(50):
            pending = [task for task in plugin._tasks if not task.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    def inject(self, text, **kwargs):
        event = FakeEvent(text, **kwargs)
        req = FakeReq()

        async def go():
            await self.plugin.on_llm_request(event, req)
            await self._drain()

        asyncio.run(go())
        return req

    def assert_standing_format_rule(self, req):
        """The 【】/no-asterisk rule is on EVERY turn, even without a recall.

        Returns the injected parts so a caller can go on to assert that no recall
        block was added (a skipped recall must leave exactly this one part).
        """
        parts = req.extra_user_content_parts
        self.assertTrue(parts, "the standing format rule must always be injected")
        text = parts[-1].text
        self.assertIn("【】", text)
        self.assertIn("禁止星号", text)
        self.assertIn("QQ 不渲染 markdown", text)
        return parts

    def assert_no_recall(self, req):
        """Recall was skipped: only the standing format rule is present."""
        parts = self.assert_standing_format_rule(req)
        self.assertEqual(len(parts), 1, "a skipped recall must add no recall block")
        self.assertNotIn("Recalled memory", parts[0].text)

    def respond(self, text, reply, **kwargs):
        event = FakeEvent(text, **kwargs)
        resp = FakeResp(reply)

        async def go():
            await self.plugin.on_llm_response(event, resp)
            await self._drain()

        asyncio.run(go())


class InjectionTest(Harness):
    def test_temp_part_is_appended(self):
        req = self.inject("why did the napcat bridge crash yesterday")
        # Recall block first, standing format rule second.
        self.assertEqual(len(req.extra_user_content_parts), 2)
        part = req.extra_user_content_parts[0]
        self.assertTrue(part.temp, "the part must be marked temporary to keep prompt caching")
        self.assertIn("<system-reminder>", part.text)
        self.assertIn("- [92cfc0da] [tester/stranger] first remembered title", part.text)
        self.assertIn("memory_feedback(ids, verdict='good')", part.text)
        self.assertEqual(req.system_prompt, "sys")
        self.assertEqual(self.plugin.state.pending_count(), 1)
        # ...and the standing rule rides along on the same turn.
        self.assert_standing_format_rule(req)

    def test_recall_argv_shape(self):
        self.inject("why did the napcat bridge crash yesterday")
        args = self.args_for("recall")
        self.assertTrue(args)
        self.assertIn("--titles", args[0])
        self.assertIn("--limit 20", args[0])
        self.assertIn("--project qq:", args[0])
        # learn is unavailable, so the older build must never receive --lexical.
        self.assertNotIn("--lexical", args[0])

    def test_commands_are_skipped(self):
        req = self.inject("/mem_status")
        self.assert_no_recall(req)
        self.assertNotIn("recall", self.commands())

    def test_greetings_are_skipped(self):
        req = self.inject("你好")
        self.assert_no_recall(req)
        self.assertNotIn("recall", self.commands())

    def test_short_and_punctuation_are_skipped(self):
        req = self.inject("ok")
        self.assert_no_recall(req)
        req = self.inject("!!!???")
        self.assert_no_recall(req)
        self.assertNotIn("recall", self.commands())

    def test_fails_open_on_rsrs_error(self):
        broken = self.tmp / "broken-rsrs"
        broken.write_text("#!/bin/sh\necho boom 1>&2\nexit 3\n", encoding="utf-8")
        os.chmod(broken, 0o755)
        plugin = self._make_plugin(rsrs_bin=str(broken))
        event = FakeEvent("why did the napcat bridge crash yesterday")
        req = FakeReq()

        async def go():
            await plugin.on_llm_request(event, req)

        asyncio.run(go())
        self.assert_no_recall(req)
        self.assertEqual(req.system_prompt, "sys")

    def test_system_prompt_fallback(self):
        event = FakeEvent("why did the napcat bridge crash yesterday")
        req = FakeReq(system_prompt="base", with_parts=False)

        async def go():
            await self.plugin.on_llm_request(event, req)

        asyncio.run(go())
        self.assertTrue(req.system_prompt.startswith("base\n\n<system-reminder>"))

    def test_cooldown_blocks_the_second_injection(self):
        plugin = self._make_plugin(cooldown_ms=60000)
        event = FakeEvent("why did the napcat bridge crash yesterday")
        req1 = FakeReq()
        req2 = FakeReq()

        async def go():
            await plugin.on_llm_request(event, req1)
            await plugin.on_llm_request(FakeEvent("another unrelated question about deploys"), req2)

        asyncio.run(go())
        # Turn 1: recall + standing rule. Turn 2: recall is on cooldown, but the
        # standing rule must still be there.
        self.assertEqual(len(req1.extra_user_content_parts), 2)
        self.assert_no_recall(req2)

    def test_group_scope_is_used_for_the_project(self):
        self.inject("why did the napcat bridge crash yesterday", group_id="777888", user_id="5001")
        args = self.args_for("recall")[0]
        self.assertIn(":group:", args)
        self.assertNotIn("777888", args)


class ExcerptInjectionTest(Harness):
    """P1-2: the top hit carries a verbatim head excerpt; everything fails open to titles."""

    QUERY = "why did the napcat bridge crash yesterday"

    def make_rsrs(self, show_branch):
        """A fake rsrs whose ``show`` behaviour is supplied by ``show_branch``."""
        recall = json.dumps(json.dumps(RECALL_PAYLOAD))
        learn_payload = json.dumps(json.dumps({"status": "ok", "summary": {"bge_available": False}}))
        ok = json.dumps(json.dumps({"command": "ok", "status": "ok", "summary": {}, "items": [], "errors": []}))
        script = f"""#!/bin/sh
printf 'CMD=%s\\n' "$1" > {json.dumps(str(self.calls))}/$$.log
printf 'ARGS=' >> {json.dumps(str(self.calls))}/$$.log
printf '%s ' "$@" | tr '\\n' ' ' >> {json.dumps(str(self.calls))}/$$.log
printf '\\n' >> {json.dumps(str(self.calls))}/$$.log
case "$1" in
  recall) printf '%s' {recall} ;;
  {show_branch}
  learn) printf '%s' {learn_payload}; exit 2 ;;
  *) printf '%s' {ok} ;;
esac
exit 0
"""
        path = self.tmp / f"rsrs-show-{abs(hash(show_branch))}"
        path.write_text(script, encoding="utf-8")
        os.chmod(path, 0o755)
        return path

    def inject_with(self, plugin, text):
        event = FakeEvent(text)
        req = FakeReq()

        async def go():
            await plugin.on_llm_request(event, req)
            await self._drain_for(plugin)

        asyncio.run(go())
        return req

    def test_default_config_shows_the_top_hit_and_injects_its_head(self):
        req = self.inject(self.QUERY)
        part = req.extra_user_content_parts[0]
        # Exactly one show, for the top hit's full uuid from the recall details.
        shows = self.args_for("show")
        self.assertEqual(len(shows), 1)
        self.assertIn("92cfc0da-1111-2222-3333-444444444444", shows[0])
        self.assertIn("摘要:", part.text)
        self.assertIn("verbatim head excerpt", part.text)
        self.assertNotIn("titles only, not bodies", part.text)
        # The rest of the framing and the feedback ring are untouched.
        self.assertIn("- [92cfc0da] [tester/stranger] first remembered title", part.text)
        self.assertIn("memory_feedback(ids, verdict='good')", part.text)
        self.assertEqual(req.system_prompt, "sys")
        self.assertEqual(self.plugin.state.pending_count(), 1)

    def test_excerpt_disabled_skips_the_show_and_keeps_titles_only(self):
        plugin = self._make_plugin(recall_excerpt_chars=0)
        req = self.inject_with(plugin, self.QUERY)
        part = req.extra_user_content_parts[0]
        self.assertNotIn("show", self.commands())
        self.assertNotIn("摘要:", part.text)
        self.assertIn("titles only, not bodies", part.text)
        self.assertNotIn("verbatim head excerpt", part.text)
        self.assertEqual(plugin.state.pending_count(), 1)

    def test_show_failure_fails_open_to_titles(self):
        plugin = self._make_plugin(rsrs_bin=str(self.make_rsrs("show) echo show-boom 1>&2; exit 1 ;;")))
        req = self.inject_with(plugin, self.QUERY)
        part = req.extra_user_content_parts[0]
        self.assertIn("- [92cfc0da] [tester/stranger] first remembered title", part.text)
        self.assertNotIn("摘要:", part.text)
        self.assertIn("titles only, not bodies", part.text)
        self.assertEqual(plugin.state.pending_count(), 1)

    def test_empty_body_fails_open_to_titles(self):
        empty = json.dumps(
            json.dumps(
                {
                    "command": "show",
                    "status": "ok",
                    "summary": {"id": "92cfc0da-1111-2222-3333-444444444444"},
                    "items": [],
                    "details": {"entry": {"content": ""}},
                    "errors": [],
                }
            )
        )
        plugin = self._make_plugin(rsrs_bin=str(self.make_rsrs(f"show) printf '%s' {empty} ;;")))
        req = self.inject_with(plugin, self.QUERY)
        part = req.extra_user_content_parts[0]
        self.assertEqual(len(self.args_for("show")), 1)
        self.assertNotIn("摘要:", part.text)
        self.assertIn("titles only, not bodies", part.text)

    def test_excerpt_is_capped_at_the_configured_budget(self):
        body = "x" * 300
        long_payload = json.dumps(
            json.dumps(
                {
                    "command": "show",
                    "status": "ok",
                    "summary": {"id": "92cfc0da-1111-2222-3333-444444444444"},
                    "items": [],
                    "details": {"entry": {"content": body}},
                    "errors": [],
                }
            )
        )
        plugin = self._make_plugin(rsrs_bin=str(self.make_rsrs(f"show) printf '%s' {long_payload} ;;")))
        req = self.inject_with(plugin, self.QUERY)
        part = req.extra_user_content_parts[0]
        self.assertIn("摘要: " + "x" * 119 + "…", part.text)
        self.assertNotIn(body, part.text)

    def test_only_the_top_hit_is_shown(self):
        self.inject(self.QUERY)
        self.assertEqual(self.commands().count("show"), 1)


class ResponseTest(Harness):
    def test_auto_diary_is_scheduled(self):
        self.respond("how do I restart the bridge", "stop it and start it again")
        args = self.args_for("remember")
        self.assertEqual(len(args), 1)
        self.assertIn("--importance trivial", args[0])
        self.assertIn("--tags qq,diary", args[0])
        self.assertIn("--type Context", args[0])
        self.assertIn("--summary", args[0])  # T1-3: every write path carries one
        self.assertIn("qq diary qq:", args[0])
        self.assertIn("User: how do I restart the bridge", args[0])

    def test_secret_is_never_diaried(self):
        self.respond("my password is hunter2", "noted")
        self.assertEqual(self.args_for("remember"), [])

    def test_group_diary_is_off_by_default(self):
        self.respond("how do I restart the bridge", "stop it and start it again", group_id="777888")
        self.assertEqual(self.args_for("remember"), [])

    def test_group_diary_can_be_enabled(self):
        plugin = self._make_plugin(auto_diary_group=True)
        event = FakeEvent("how do I restart the bridge", group_id="777888")
        resp = FakeResp("stop it and start it again")

        async def go():
            await plugin.on_llm_response(event, resp)
            await self._drain_for(plugin)

        asyncio.run(go())
        self.assertEqual(len(self.args_for("remember")), 1)

    def test_command_turn_is_not_diaried(self):
        self.respond("/mem_status", "memory status")
        self.assertEqual(self.args_for("remember"), [])

    def test_cited_id_marks_good(self):
        self.inject("why did the napcat bridge crash yesterday")
        self.respond("so what happened", "see [92cfc0da] for the earlier fix")
        marks = self.args_for("query-log")
        self.assertTrue(any("--good" in line and "--source model" in line for line in marks), marks)
        self.assertTrue(any("--query-log-id uuid-recall-1" in line for line in marks), marks)
        self.assertEqual(self.plugin.state.pending_count(), 0)

    def test_correction_marks_bad(self):
        self.inject("why did the napcat bridge crash yesterday")
        self.inject("不对，不是这个")
        marks = self.args_for("query-log")
        self.assertTrue(any("--bad" in line for line in marks), marks)


class CommandTest(Harness):
    def run_command(self, handler, event):
        return _collect(handler(event))

    def test_help(self):
        out = self.run_command(self.plugin.mem, FakeEvent("/mem"))
        text = out[0][1]
        self.assertIn("dafeiyu memory", text)
        self.assertIn("mode:", text)
        self.assertIn("/mem_recall", text)

    def test_status(self):
        out = self.run_command(self.plugin.mem_status, FakeEvent("/mem_status"))
        text = out[0][1]
        self.assertIn("memory status", text)
        self.assertIn("scheme:", text)
        self.assertIn("learn: unavailable", text)
        self.assertIn("scope: qq:", text)
        self.assertNotIn("|", text)  # no markdown tables

    def test_recall_command_returns_titles_only(self):
        out = self.run_command(self.plugin.mem_recall, FakeEvent("／mem_recall deploy failed"))
        text = out[0][1]
        self.assertIn("92cfc0da [未知/stranger] first remembered title", text)
        self.assertNotIn("BODY_SHOULD_NOT_LEAK", text)
        self.assertLessEqual(len(text), 1200)

    def test_recall_command_usage(self):
        out = self.run_command(self.plugin.mem_recall, FakeEvent("/mem_recall"))
        self.assertIn("usage", out[0][1])

    def test_show_private(self):
        self.inject("why did the napcat bridge crash yesterday")
        out = self.run_command(self.plugin.mem_show, FakeEvent("/mem_show 92cfc0da"))
        self.assertIn("BODY_SHOULD_NOT_LEAK", out[0][1])

    def test_show_rejects_an_id_outside_the_recent_recall(self):
        self.inject("why did the napcat bridge crash yesterday")
        out = self.run_command(self.plugin.mem_show, FakeEvent("/mem_show deadbeef"))
        self.assertIn("not from this chat's recent recall", out[0][1])
        self.assertNotIn("BODY_SHOULD_NOT_LEAK", out[0][1])

    def test_show_group_requires_admin(self):
        out = self.run_command(self.plugin.mem_show, FakeEvent("/mem_show 92cfc0da", group_id="777888", user_id="5001"))
        self.assertIn("group read denied: sender relation is stranger", out[0][1])
        self.assertNotIn("BODY_SHOULD_NOT_LEAK", out[0][1])

    def test_show_group_admin_allowed(self):
        self.inject("why did the napcat bridge crash yesterday", group_id="777888", user_id="9001")
        out = self.run_command(self.plugin.mem_show, FakeEvent("/mem_show 92cfc0da", group_id="777888", user_id="9001"))
        self.assertIn("BODY_SHOULD_NOT_LEAK", out[0][1])

    def test_remember(self):
        out = self.run_command(self.plugin.mem_remember, FakeEvent("/mem_remember the bridge runs on port 8080"))
        text = out[0][1]
        self.assertIn("stored [abcd1234]", text)
        args = self.args_for("remember")[0]
        self.assertIn("--importance important", args)
        self.assertIn("--tags qq,fact", args)
        self.assertIn("--title QQ fact: the bridge", args)

    def test_forget_is_admin_only(self):
        out = self.run_command(self.plugin.mem_forget, FakeEvent("/mem_forget 92cfc0da"))
        self.assertIn("admin only", out[0][1])
        self.assertEqual(self.args_for("forget"), [])
        self.inject("why did the napcat bridge crash yesterday", user_id="9001")
        out = self.run_command(self.plugin.mem_forget, FakeEvent("/mem_forget 92cfc0da", user_id="9001"))
        self.assertIn("forgot 92cfc0da", out[0][1])
        self.assertEqual(
            self.args_for("forget"),
            ["forget 92cfc0da-1111-2222-3333-444444444444 --json"],
        )

    def test_forget_rejects_an_id_outside_the_recent_recall(self):
        self.inject("why did the napcat bridge crash yesterday", user_id="9001")
        out = self.run_command(self.plugin.mem_forget, FakeEvent("/mem_forget deadbeef", user_id="9001"))
        self.assertIn("not from this chat's recent recall", out[0][1])
        self.assertEqual(self.args_for("forget"), [])

    def test_forget_group_admin_uses_the_group_scope(self):
        self.inject("why did the napcat bridge crash yesterday", group_id="777888", user_id="9001")
        out = self.run_command(
            self.plugin.mem_forget,
            FakeEvent("/mem_forget 92cfc0da", group_id="777888", user_id="9001"),
        )
        self.assertIn("forgot 92cfc0da", out[0][1])
        self.assertEqual(
            self.args_for("forget"),
            ["forget 92cfc0da-1111-2222-3333-444444444444 --json"],
        )

    def test_feedback_requires_pending(self):
        out = self.run_command(self.plugin.mem_good, FakeEvent("/mem_good 92cfc0da"))
        self.assertIn("no recent recall", out[0][1])

    def test_feedback_after_recall(self):
        self.inject("why did the napcat bridge crash yesterday")
        out = self.run_command(self.plugin.mem_good, FakeEvent("/mem_good 92cfc0da"))
        text = out[0][1]
        self.assertIn("good recorded", text)
        self.assertIn("--good", self.args_for("query-log")[0])

    def test_mute_and_list_and_unmute(self):
        out = self.run_command(self.plugin.mem_mute, FakeEvent("/mem_mute deploy failures"))
        self.assertIn("muted:", out[0][1])
        listed = self.run_command(self.plugin.mem_mute, FakeEvent("/mem_mute list"))[0][1]
        self.assertIn("muted topics in this chat: 1", listed)
        unmuted = self.run_command(self.plugin.mem_unmute, FakeEvent("/mem_unmute deploy failures"))[0][1]
        self.assertIn("unmuted", unmuted)
        again = self.run_command(self.plugin.mem_mute, FakeEvent("/mem_mute list"))[0][1]
        self.assertIn("0", again)

    def test_learn_reports_missing_build(self):
        out = self.run_command(self.plugin.mem_learn, FakeEvent("/mem_learn train", user_id="9001"))
        self.assertIn("learn unavailable", out[0][1])
        # Only the cached `learn --help` probe ran; no training was attempted.
        self.assertFalse(any("train" in line for line in self.args_for("learn")))

    def test_admin_commands_fail_closed_when_admin_ids_is_empty(self):
        plugin = self._make_plugin(admin_ids=[])
        out = _collect(plugin.mem_learn(FakeEvent("/mem_learn status", user_id="5001")))
        self.assertIn("admin only", out[0][1])
        # Group body commands also fail closed without admin_ids.
        out = _collect(plugin.mem_show(FakeEvent("/mem_show 92cfc0da", group_id="777888", user_id="5001")))
        self.assertIn("group read denied: sender relation is stranger", out[0][1])

    def test_learn_is_admin_only(self):
        out = self.run_command(self.plugin.mem_learn, FakeEvent("/mem_learn status", user_id="5001"))
        self.assertIn("admin only", out[0][1])

    def test_flywheel_explains_what_is_missing(self):
        out = self.run_command(self.plugin.mem_flywheel, FakeEvent("/mem_flywheel", user_id="9001"))
        text = out[0][1]
        self.assertIn("flywheel unavailable", text)

    def test_flywheel_runs_in_the_background_when_installed(self):
        model = self.tmp / "granite"
        model.mkdir()
        for name in ("model.safetensors", "config.json"):
            (model / name).write_text("x")
        (model / "1_Pooling").mkdir()
        repo = self.tmp / "repo"
        script = repo / "scripts" / "encoder" / "weekly_encoder.sh"
        script.parent.mkdir(parents=True)
        marker = self.tmp / "flywheel-ran"
        capture = self.tmp / "flywheel-env"
        anchor = self.tmp / "anchor"
        script.write_text(
            "#!/bin/sh\n"
            f"printf '%s|%s|%s' \"$RSRS_ENCODER_STATE\" \"$RSRS_ENCODER_BASE\" "
            f"\"${{RSRS_DATA_DIR-unset}}\" > {json.dumps(str(capture))}\n"
            f"printf ran > {json.dumps(str(marker))}\n"
        )
        plugin = self._make_plugin(
            jobs={
                "enabled": False,
                "respire_repo": str(repo),
                "granite_model_dir": str(model),
                "anchor_dir": str(anchor),
            }
        )

        async def go():
            out = [item async for item in plugin.mem_flywheel(FakeEvent("/mem_flywheel", user_id="9001"))]
            await self._drain_for(plugin)
            return out

        out = asyncio.run(go())
        self.assertIn("started in background", out[0][1])
        self.assertTrue(marker.exists(), "the encoder script should have run")
        self.assertEqual(
            capture.read_text(),
            f"{anchor}|{model}|unset",
            "manual flywheel must export the Granite model as BASE and the anchor as STATE",
        )

    def test_clear_scope(self):
        self.inject("why did the napcat bridge crash yesterday")
        out = self.run_command(self.plugin.mem_clear, FakeEvent("/mem_clear"))
        self.assertIn("cleared", out[0][1])
        self.assertEqual(self.plugin.state.pending_count(), 0)

    def test_reflect_requires_admin_and_sets_parent(self):
        out = self.run_command(self.plugin.mem_reflect, FakeEvent("/mem_reflect maybe it was dns", user_id="5001"))
        self.assertIn("admin only", out[0][1])
        out = self.run_command(
            self.plugin.mem_reflect,
            FakeEvent("/mem_reflect maybe it was dns [92cfc0da]", user_id="9001"),
        )
        self.assertIn("unverified", out[0][1])
        args = self.args_for("remember")[0]
        self.assertIn("--type hypothesis", args)
        self.assertIn("--importance important", args)
        self.assertIn("--parent 92cfc0da", args)
        self.assertIn("--valid-until", args)
        self.assertIn("--review-at", args)
        # Defaults: expires in 7 days, reviewed after 3.
        import re as _re
        from datetime import datetime as _dt, timezone as _tz
        valid = args.split("--valid-until ")[1].split(" ")[0]
        review = args.split("--review-at ")[1].split(" ")[0]
        now = _dt.now(_tz.utc)
        self.assertIn((_dt.strptime(valid, "%Y-%m-%dT%H:%M:%S.000Z").replace(tzinfo=_tz.utc) - now).days, (6, 7))
        self.assertIn((_dt.strptime(review, "%Y-%m-%dT%H:%M:%S.000Z").replace(tzinfo=_tz.utc) - now).days, (2, 3))

    def test_unknown_command_is_ignored(self):
        out = self.run_command(self.plugin.mem, FakeEvent("/mem_recall deploy"))
        self.assertEqual(out, [])

    def test_disabled_plugin_still_answers_help(self):
        plugin = self._make_plugin(enabled=False)
        out = _collect(plugin.mem(FakeEvent("/mem")))
        self.assertIn("dafeiyu memory", out[0][1])


class RecallRingTest(Harness):
    """The recent-id ring spans several recalls in one scope."""

    def scope(self):
        return self.plugin._turn(FakeEvent("x")).scope

    def test_memory_show_fails_closed_without_a_recall(self):
        out = asyncio.run(self.plugin.memory_show_tool(FakeEvent("hi"), "92cfc0da"))
        self.assertIn("no recent recall in this chat", out)
        self.assertNotIn("show", self.commands())

    def test_memory_show_accepts_an_earlier_recall(self):
        self.inject("why did the napcat bridge crash yesterday")
        # A newer recall arrives; the first recall's id must stay usable.
        self.plugin.state.record_pending(
            self.scope(),
            ["deadbeef"],
            full_ids=["deadbeef-aaaa-bbbb-cccc-dddddddddddd"],
            id_map={"deadbeef": "deadbeef-aaaa-bbbb-cccc-dddddddddddd"},
        )
        out = asyncio.run(self.plugin.memory_show_tool(FakeEvent("x"), "92cfc0da"))
        self.assertIn("BODY_SHOULD_NOT_LEAK", out)
        self.assertEqual(
            self.plugin._recent_id(self.scope(), "deadbeef"),
            "deadbeef-aaaa-bbbb-cccc-dddddddddddd",
        )
        self.assertEqual(
            self.plugin._pending_id(self.scope(), "92cfc0da"),
            "92cfc0da-1111-2222-3333-444444444444",
        )

    def test_memory_show_rejects_an_foreign_id(self):
        self.inject("why did the napcat bridge crash yesterday")
        out = asyncio.run(self.plugin.memory_show_tool(FakeEvent("x"), "12345678"))
        self.assertIn("not from this chat's recent recall", out)

    def test_recent_id_only_covers_the_same_scope(self):
        self.inject("why did the napcat bridge crash yesterday")
        self.assertEqual(self.plugin._recent_id("qq:other", "92cfc0da"), "")


class AuditReviewTest(Harness):
    def test_mem_audit_private_owner_gets_counts_only(self):
        out = self.run_command(self.plugin.mem_audit, FakeEvent("/mem_audit"))
        text = out[0][1]
        self.assertIn("audit: warn", text)
        self.assertIn("active 4911", text)
        self.assertIn("orphan 0", text)
        self.assertIn("max depth 21", text)
        self.assertIn("hot untyped 870", text)
        self.assertIn("skill gaps 620", text)
        self.assertNotIn("DETAIL_TITLE_SHOULD_NOT_LEAK", text)
        self.assertNotIn("deadbeef", text)
        self.assertLessEqual(len(text), 1200)
        self.assertEqual(self.args_for("audit"), ["audit --json"])

    def test_mem_audit_group_requires_admin(self):
        out = self.run_command(
            self.plugin.mem_audit, FakeEvent("/mem_audit", group_id="777888", user_id="5001")
        )
        self.assertIn("admin only", out[0][1])
        self.assertNotIn("audit", self.commands())
        allowed = self.run_command(
            self.plugin.mem_audit, FakeEvent("/mem_audit", group_id="777888", user_id="9001")
        )
        self.assertIn("audit: warn", allowed[0][1])

    def test_mem_review_lists_due_records(self):
        out = self.run_command(self.plugin.mem_review, FakeEvent("/mem_review"))
        text = out[0][1]
        self.assertIn("review: due | due 1", text)
        self.assertIn("[92cfc0da] first remembered title", text)
        self.assertLessEqual(len(text), 1200)
        self.assertIn("review --due --json --limit", self.args_for("review")[0])

    def test_mem_review_group_requires_admin(self):
        out = self.run_command(
            self.plugin.mem_review, FakeEvent("/mem_review", group_id="777888", user_id="5001")
        )
        self.assertIn("admin only", out[0][1])
        self.assertNotIn("review", self.commands())

    def test_mem_review_failure_is_reported_once(self):
        counter = self.tmp / "review-count"
        broken = self.tmp / "broken-review"
        broken.write_text(
            "#!/bin/sh\n"
            f"printf 'x' >> {json.dumps(str(counter))}\n"
            "printf '%s' '{\"command\":\"review\",\"status\":\"fail\","
            "\"summary\":{\"saved\":false},\"items\":[],"
            "\"errors\":[\"decode review snapshot bdd81d65: nonce length must be 12 bytes\"]}'\n"
            "exit 1\n",
            encoding="utf-8",
        )
        os.chmod(broken, 0o755)
        plugin = self._make_plugin(rsrs_bin=str(broken))
        out = _collect(plugin.mem_review(FakeEvent("/mem_review")))
        text = out[0][1]
        self.assertIn("review failed:", text)
        self.assertIn("decode review snapshot", text)
        self.assertLessEqual(len(text), 260)
        # One attempt, no retry.
        self.assertEqual(counter.read_text(), "x")

    def test_memory_audit_tool_is_registered_and_count_only(self):
        self.assertTrue(callable(getattr(self.plugin, "memory_audit_tool", None)))
        text = asyncio.run(self.plugin.memory_audit_tool(FakeEvent("x")))
        self.assertIn("audit: warn", text)
        self.assertIn("active 4911", text)
        self.assertNotIn("DETAIL_TITLE_SHOULD_NOT_LEAK", text)
        self.assertNotIn("deadbeef", text)
        self.assertLessEqual(len(text), 800)
        self.assertEqual(self.args_for("audit"), ["audit --json"])

    def test_memory_audit_tool_group_requires_admin(self):
        denied = asyncio.run(
            self.plugin.memory_audit_tool(FakeEvent("x", group_id="777888", user_id="5001"))
        )
        self.assertIn("group audit denied: admin only", denied)
        allowed = asyncio.run(
            self.plugin.memory_audit_tool(FakeEvent("x", group_id="777888", user_id="9001"))
        )
        self.assertIn("audit: warn", allowed)

    def test_mem_status_reports_maintenance(self):
        out = self.run_command(self.plugin.mem_status, FakeEvent("/mem_status"))
        self.assertIn("maintenance: never", out[0][1])
        self.plugin.state.write_jobs(
            {
                "maintenance": {
                    "finished_at": 1_700_000_000_000,
                    "status": "partial",
                    "audit": {"status": "ok", "counts": {}},
                    "review": {"status": "error", "counts": {}, "error": "boom"},
                    "spotcheck": {
                        "status": "ok",
                        "counts": {
                            "checked": 6,
                            "reflection_as_evidence": 2,
                            "expired": 0,
                            "source_missing": 1,
                        },
                    },
                }
            }
        )
        out = self.run_command(self.plugin.mem_status, FakeEvent("/mem_status"))
        text = out[0][1]
        self.assertIn("maintenance:", text)
        self.assertIn("audit=ok", text)
        self.assertIn("review=error", text)
        self.assertNotIn("never", text)
        # Nonzero spot-check findings ride along; zero counts stay quiet.
        self.assertIn("spot=ok", text)
        self.assertIn("reflection_as_evidence=2", text)
        self.assertIn("source_missing=1", text)
        self.assertNotIn("expired=0", text)

    def test_help_lists_the_new_commands(self):
        text = self.run_command(self.plugin.mem, FakeEvent("/mem"))[0][1]
        self.assertIn("/mem_audit", text)
        self.assertIn("/mem_review", text)

    def run_command(self, handler, event):
        return _collect(handler(event))


class LifecycleTest(Harness):
    def _offline_jobs(self, **overrides):
        jobs = {
            "enabled": False,
            "respire_repo": str(self.tmp / "no-repo"),
            "granite_model_dir": str(self.tmp / "no-model"),
            "anchor_dir": str(self.tmp / "no-anchor"),
        }
        jobs.update(overrides)
        return jobs

    def test_jobs_do_not_start_when_every_schedule_is_off(self):
        plugin = self._make_plugin(jobs=self._offline_jobs(maintenance_enabled=False))

        async def go():
            await plugin.on_loaded()
            return plugin.jobs.start()

        self.assertFalse(asyncio.run(go()))
        self.assertFalse(plugin.jobs.running)

    def test_maintenance_only_schedule_starts(self):
        plugin = self._make_plugin(jobs=self._offline_jobs(maintenance_enabled=True))

        async def go():
            await plugin.on_loaded()
            running = plugin.jobs.running
            await plugin.jobs.stop()
            return running

        self.assertTrue(asyncio.run(go()))
        self.assertFalse(plugin.jobs.running)

    def test_jobs_start_when_enabled(self):
        plugin = self._make_plugin(
            jobs={
                "enabled": True,
                "daily_hour": 23,
                "daily_minute": 59,
                "respire_repo": str(self.tmp / "no-repo"),
                "granite_model_dir": str(self.tmp / "no-model"),
                "anchor_dir": str(self.tmp / "no-anchor"),
            }
        )

        async def go():
            await plugin.on_loaded()
            running = plugin.jobs.running
            await plugin.jobs.stop()
            return running

        self.assertTrue(asyncio.run(go()))
        self.assertFalse(plugin.jobs.running)

    def test_terminate_cancels_background_tasks(self):
        async def go():
            self.plugin._spawn(asyncio.sleep(30))
            await asyncio.sleep(0)
            await self.plugin.terminate()
            return len(self.plugin._tasks)

        self.assertEqual(asyncio.run(go()), 0)


class LearnAvailableTest(Harness):
    learn_available = True

    def test_learn_status_uses_the_subcommand(self):
        out = _collect(self.plugin.mem_learn(FakeEvent("/mem_learn status", user_id="9001")))
        self.assertIn("learn status", out[0][1])
        self.assertIn("learn", self.commands())

    def test_recall_still_omits_lexical_without_bge(self):
        # learn available + bge unavailable + retrieval=auto => --lexical is passed.
        self.inject("why did the napcat bridge crash yesterday")
        self.assertIn("--lexical", self.args_for("recall")[0])




class SelfNoteAutoTest(Harness):
    """Automatic self-note refresh: turn counting, triggers and read-only status."""

    def auto_plugin(self, **selfnote):
        settings = {
            "auto_enabled": True,
            "every_sessions": 5,
            "min_interval_ms": 3600000,
            "retry_backoff_ms": 600000,
        }
        settings.update(selfnote)
        return self._make_plugin(selfnote=settings)

    def reply(self, plugin, text="how do I restart the bridge", answer="stop it and start it again", **kwargs):
        event = FakeEvent(text, **kwargs)
        resp = FakeResp(answer)

        async def go():
            await plugin.on_llm_response(event, resp)
            await self._drain_for(plugin)

        asyncio.run(go())

    def auto_record(self, plugin, event=None):
        scope = plugin.selfnote.build_scope(plugin._turn(event or FakeEvent("x")))
        return plugin.state.read_selfnote(scope)

    def selfnote_remembers(self):
        return [line for line in self.args_for("remember") if "--tags selfnote,persona" in line]

    def test_private_admin_reply_counts_one_turn(self):
        plugin = self.auto_plugin()
        self.reply(plugin, user_id="9001")
        record = self.auto_record(plugin)
        self.assertEqual(record["auto_turns"], 1)
        self.assertEqual(record["auto_failures"], 0)
        self.assertEqual(record["last_auto_at"], "")
        # No self-note write below the threshold.
        self.assertEqual(self.selfnote_remembers(), [])
        self.assertEqual(plugin.context.persona_manager.updates, 0)

    def test_auto_diary_off_still_counts_real_private_turns(self):
        plugin = self._make_plugin(
            auto_diary=False,
            selfnote={
                "auto_enabled": True,
                "every_sessions": 5,
                "min_interval_ms": 3600000,
                "retry_backoff_ms": 600000,
            },
        )
        self.reply(plugin, user_id="9001")
        record = self.auto_record(plugin)
        self.assertEqual(record["auto_turns"], 1)

    def test_group_other_user_command_and_secret_do_not_count(self):
        plugin = self.auto_plugin()
        self.reply(plugin, user_id="9001", group_id="777888")
        self.assertIsNone(self.auto_record(plugin, FakeEvent("x", group_id="777888", user_id="9001")))
        self.reply(plugin, user_id="5001")
        self.assertIsNone(self.auto_record(plugin))
        self.reply(plugin, text="/mem_status", answer="memory status", user_id="9001")
        self.assertIsNone(self.auto_record(plugin))
        self.reply(plugin, text="my password is hunter2", answer="noted", user_id="9001")
        self.assertIsNone(self.auto_record(plugin))
        self.assertEqual(self.selfnote_remembers(), [])

    def test_threshold_triggers_one_background_refresh(self):
        plugin = self.auto_plugin()
        for _ in range(5):
            self.reply(plugin, user_id="9001")
        pending = [task for task in plugin._tasks if not task.done()]
        self.assertEqual(pending, [], "the refresh must have finished in the background")
        records = self.selfnote_remembers()
        self.assertEqual(len(records), 1)
        self.assertIn("--type self", records[0])
        self.assertIn(":selfnote", records[0])
        record = self.auto_record(plugin)
        self.assertEqual(record["auto_turns"], 0)
        self.assertEqual(record["auto_failures"], 0)
        self.assertTrue(record["last_auto_at"])
        self.assertEqual(record["id"], "abcd1234-0000-0000-0000-000000000000")
        self.assertEqual(plugin.context.persona_manager.updates, 1)
        self.assertIn("简短直接", plugin.context.persona_manager.prompt)

        # The next replies start a fresh window; nothing else fires.
        self.reply(plugin, user_id="9001")
        self.assertIsNotNone(self.auto_record(plugin))
        self.assertEqual(len(self.selfnote_remembers()), 1)

    def test_auto_off_counts_nothing(self):
        for _ in range(6):
            self.respond("how do I restart the bridge", "stop it and start it again", user_id="9001")
        self.assertIsNone(self.auto_record(self.plugin))
        self.assertEqual(self.selfnote_remembers(), [])
        self.assertEqual(self.plugin.context.persona_manager.updates, 0)

    def test_show_auto_and_status_lines(self):
        plugin = self.auto_plugin()
        self.reply(plugin, user_id="9001")
        out = _collect(plugin.mem_selfnote(FakeEvent("/mem_selfnote auto", user_id="9001")))
        self.assertIn("auto: on turns 1/5", out[0][1])
        self.assertIn("failures 0", out[0][1])

        out = _collect(plugin.mem_selfnote(FakeEvent("/mem_selfnote", user_id="9001")))
        self.assertTrue(out[0][1].startswith("selfnote: none"))
        self.assertIn("auto: on turns 1/5", out[0][1])

        out = _collect(plugin.mem_status(FakeEvent("/mem_status", user_id="9001")))
        self.assertIn("auto=on turns=1/5", out[0][1])

    def test_off_status_and_auto_off_line(self):
        out = _collect(self.plugin.mem_selfnote(FakeEvent("/mem_selfnote", user_id="9001")))
        self.assertEqual(out[0][1], "selfnote: none\nauto: off")
        out = _collect(self.plugin.mem_selfnote(FakeEvent("/mem_selfnote auto", user_id="9001")))
        self.assertEqual(out[0][1], "auto: off")
        out = _collect(self.plugin.mem_status(FakeEvent("/mem_status", user_id="9001")))
        self.assertIn("auto=off turns=0/20", out[0][1])

    def test_auto_status_is_read_only_and_admin_only(self):
        plugin = self.auto_plugin()
        out = _collect(plugin.mem_selfnote(FakeEvent("/mem_selfnote auto", user_id="5001")))
        self.assertIn("admin only", out[0][1])
        before = self.auto_record(plugin)
        out = _collect(plugin.mem_selfnote(FakeEvent("/mem_selfnote auto", user_id="9001")))
        self.assertIn("auto: on", out[0][1])
        self.assertEqual(self.auto_record(plugin), before)
        # `auto` is not a write command and never consumes a config change.
        self.assertNotIn("usage", out[0][1])

    def test_group_status_hides_the_id_but_keeps_auto(self):
        plugin = self.auto_plugin()
        _collect(plugin.mem_selfnote(FakeEvent("/mem_selfnote set 温和一点", user_id="9001")))
        self.reply(plugin, user_id="9001")
        # The self-note scope is bot-wide, so the group sees the same record but
        # a non-admin must never see its id.
        out = _collect(
            plugin.mem_status(FakeEvent("/mem_status", group_id="777888", user_id="5001"))
        )
        self.assertIn("selfnote: managed auto=on turns=1/5", out[0][1])
        self.assertNotIn("abcd1234", out[0][1])


class SelfNoteCommandTest(Harness):
    """``/mem_selfnote`` permissions, preview/refresh/set/clear and the tool."""

    def run_command(self, handler, event):
        return _collect(handler(event))

    def prompt(self):
        return self.plugin.context.persona_manager.prompt

    def selfnote_scope(self, event=None):
        return self.plugin.selfnote.build_scope(self.plugin._turn(event or FakeEvent("x")))

    def test_show_reports_none_without_a_block(self):
        out = self.run_command(self.plugin.mem_selfnote, FakeEvent("/mem_selfnote", user_id="9001"))
        self.assertEqual(out[0][1], "selfnote: none\nauto: off")

    def test_preview_generates_but_writes_nothing(self):
        out = self.run_command(
            self.plugin.mem_selfnote, FakeEvent("/mem_selfnote preview", user_id="9001")
        )
        text = out[0][1]
        self.assertIn("selfnote preview", text)
        self.assertIn("简短直接", text)
        self.assertIn("source_ids: 92cfc0da", text)
        # No persona write, no state, no memory write.
        self.assertEqual(self.plugin.context.persona_manager.updates, 0)
        self.assertIsNone(self.plugin.state.read_selfnote(self.selfnote_scope()))
        self.assertEqual(self.args_for("remember"), [])
        self.assertEqual(len(self.plugin.context.llm_calls), 1)
        self.assertEqual(self.plugin.context.llm_calls[0]["provider"], "fake-provider")

    def test_preview_requires_an_admin(self):
        out = self.run_command(
            self.plugin.mem_selfnote, FakeEvent("/mem_selfnote preview", user_id="5001")
        )
        self.assertIn("admin only", out[0][1])
        self.assertEqual(self.plugin.context.llm_calls, [])

    def test_refresh_writes_rsrs_persona_and_state_then_supersedes(self):
        out = self.run_command(
            self.plugin.mem_selfnote, FakeEvent("/mem_selfnote refresh", user_id="9001")
        )
        self.assertIn("selfnote updated [abcd1234]", out[0][1])
        self.assertIn("简短直接", out[0][1])
        first = self.args_for("remember")
        self.assertEqual(len(first), 1)
        self.assertIn("--type self", first[0])
        self.assertIn("--tags selfnote,persona", first[0])
        self.assertIn(":selfnote", first[0])
        self.assertNotIn("--supersedes", first[0])
        self.assertNotIn("--force", first[0])
        prompt = self.prompt()
        self.assertEqual(prompt.count("<!-- dafeiyu:selfnote:start -->"), 1)
        self.assertTrue(prompt.startswith("base persona rules"))
        record = self.plugin.state.read_selfnote(self.selfnote_scope())
        self.assertEqual(record["id"], "abcd1234-0000-0000-0000-000000000000")
        self.assertEqual(record["sources"], ["92cfc0da"])

        # A second refresh replaces the block and supersedes the previous id.
        self.run_command(self.plugin.mem_selfnote, FakeEvent("/mem_selfnote refresh", user_id="9001"))
        second = self.args_for("remember")[1]
        self.assertIn("--supersedes abcd1234-0000-0000-0000-000000000000", second)
        self.assertEqual(self.prompt().count("<!-- dafeiyu:selfnote:start -->"), 1)

    def test_set_and_clear_keep_memories(self):
        out = self.run_command(
            self.plugin.mem_selfnote, FakeEvent("/mem_selfnote set 说话更温和一点", user_id="9001")
        )
        self.assertIn("selfnote set [abcd1234]", out[0][1])
        self.assertIn("说话更温和一点", self.prompt())
        record = self.plugin.state.read_selfnote(self.selfnote_scope())
        self.assertEqual(record["brief"], "说话更温和一点")
        self.assertEqual(record["sources"], [])
        args = self.args_for("remember")[0]
        self.assertIn("主人手动设定", args)

        out = self.run_command(self.plugin.mem_selfnote, FakeEvent("/mem_selfnote clear", user_id="9001"))
        self.assertIn("cleared", out[0][1])
        self.assertIn("memories untouched", out[0][1])
        self.assertNotIn("<!-- dafeiyu:selfnote:start -->", self.prompt())
        self.assertIsNone(self.plugin.state.read_selfnote(self.selfnote_scope()))
        self.assertEqual(self.args_for("forget"), [])

    def test_set_rejects_an_overlong_brief(self):
        out = self.run_command(
            self.plugin.mem_selfnote, FakeEvent("/mem_selfnote set " + "x" * 301, user_id="9001")
        )
        self.assertIn("brief too long", out[0][1])
        self.assertEqual(self.args_for("remember"), [])

    def test_write_commands_require_an_admin(self):
        for command in ("/mem_selfnote refresh", "/mem_selfnote clear"):
            out = self.run_command(self.plugin.mem_selfnote, FakeEvent(command, user_id="5001"))
            self.assertIn("admin only", out[0][1])
        self.assertEqual(self.plugin.context.persona_manager.updates, 0)
        self.assertEqual(self.args_for("remember"), [])

    def test_selfnote_is_private_only_and_admin_only(self):
        out = self.run_command(self.plugin.mem_selfnote, FakeEvent("/mem_selfnote", user_id="5001"))
        self.assertIn("admin only", out[0][1])
        out = self.run_command(
            self.plugin.mem_selfnote, FakeEvent("/mem_selfnote", group_id="777888", user_id="5001")
        )
        self.assertIn("private chat", out[0][1])
        out = self.run_command(
            self.plugin.mem_selfnote, FakeEvent("/mem_selfnote", group_id="777888", user_id="9001")
        )
        self.assertIn("private chat", out[0][1])

    def test_tool_is_admin_only(self):
        text = asyncio.run(
            self.plugin.memory_selfnote_tool(
                FakeEvent("x", user_id="5001"), "内容", "摘要", "92cfc0da"
            )
        )
        self.assertIn("admin only", text)
        self.assertEqual(self.args_for("remember"), [])

    def test_tool_requires_ids_from_the_recent_recall(self):
        text = asyncio.run(
            self.plugin.memory_selfnote_tool(
                FakeEvent("x", user_id="9001"), "内容", "摘要", "92cfc0da"
            )
        )
        self.assertIn("not from this chat's recent recall", text)
        self.assertEqual(self.args_for("remember"), [])

    def test_tool_writes_a_versioned_self_note(self):
        self.inject("why did the napcat bridge crash yesterday", user_id="9001")
        text = asyncio.run(
            self.plugin.memory_selfnote_tool(
                FakeEvent("x", user_id="9001"),
                "我说话简短直接，先查证再回答。",
                "简短直接，先查证再回答。",
                "92cfc0da",
            )
        )
        self.assertIn("selfnote updated [abcd1234]", text)
        args = self.args_for("remember")[0]
        self.assertIn("--type self", args)
        self.assertNotIn("--force", args)
        self.assertIn("简短直接", self.prompt())
        self.assertIsNotNone(self.plugin.state.read_selfnote(self.selfnote_scope()))

    def test_status_reports_the_selfnote_version(self):
        out = self.run_command(self.plugin.mem_status, FakeEvent("/mem_status"))
        self.assertIn("selfnote: none", out[0][1])
        self.run_command(
            self.plugin.mem_selfnote, FakeEvent("/mem_selfnote set 温和一点", user_id="9001")
        )
        out = self.run_command(self.plugin.mem_status, FakeEvent("/mem_status", user_id="9001"))
        text = out[0][1]
        self.assertIn("selfnote: abcd1234 updated", text)
        self.assertNotIn("selfnote: none", text)

    def test_help_lists_the_selfnote_command(self):
        text = self.run_command(self.plugin.mem, FakeEvent("/mem"))[0][1]
        self.assertIn("/mem_selfnote", text)


class TypedWriteTest(Harness):
    """P0-1/P2-3: typed reflect writes, artifact facets and legacy degradation."""

    def gate_off_plugin(self):
        """A second plugin whose fake CLI reports metadata v1 disabled."""
        recall = json.dumps(json.dumps(RECALL_PAYLOAD))
        remember = json.dumps(json.dumps(REMEMBER_PAYLOAD))
        metadata = json.dumps(json.dumps({"status": "ok", "summary": {"enabled": False, "mode": "isolated"}}))
        ok = json.dumps(json.dumps({"command": "ok", "status": "ok", "summary": {}, "items": [], "errors": []}))
        script = f"""#!/bin/sh
printf 'CMD=%s\\n' "$1" > {json.dumps(str(self.calls))}/$$.log
printf 'ARGS=' >> {json.dumps(str(self.calls))}/$$.log
printf '%s ' "$@" | tr '\\n' ' ' >> {json.dumps(str(self.calls))}/$$.log
printf '\\n' >> {json.dumps(str(self.calls))}/$$.log
case "$1" in
  recall) printf '%s' {recall} ;;
  remember) printf '%s' {remember} ;;
  metadata) printf '%s' {metadata} ;;
  *) printf '%s' {ok} ;;
esac
exit 0
"""
        path = self.tmp / "rsrs-gate-off"
        path.write_text(script, encoding="utf-8")
        os.chmod(path, 0o755)
        return self._make_plugin(rsrs_bin=str(path))

    def reflect(self, plugin=None, content="maybe it was dns", source="", days=7):
        target = plugin or self.plugin
        return asyncio.run(
            target.memory_reflect_tool(FakeEvent("x", user_id="9001"), content, source, days)
        )

    def remember(self, plugin=None, **kwargs):
        target = plugin or self.plugin
        payload = {
            "judgment": "the bridge log ships to /tmp/bridge.log",
            "reasoning": "operator said so",
            "memory_type": "knowledge",
            "tags": "",
            "artifact": "",
        }
        payload.update(kwargs)
        return asyncio.run(target.memory_remember_tool(FakeEvent("x"), **payload))

    def test_reflect_tool_passes_typed_flags(self):
        # Regression: the tool once passed camelCase kwargs (validUntil) that
        # remember_args rejects, crashing the tool before any CLI call.
        out = self.reflect()
        self.assertIn("reflection stored [abcd1234]", out)
        self.assertIn("expires in 7d", out)
        args = self.args_for("remember")[0]
        self.assertIn("--type hypothesis", args)
        self.assertIn("--valid-until", args)
        self.assertIn("--review-at", args)
        self.assertIn("--importance important", args)

    def test_reflect_tool_degrades_when_gate_is_off(self):
        plugin = self.gate_off_plugin()
        out = self.reflect(plugin=plugin)
        self.assertIn("metadata v1 off", out)
        args = self.args_for("remember")[0]
        self.assertNotIn("--type hypothesis", args)
        self.assertNotIn("--valid-until", args)
        self.assertIn("--tags qq,hypothesis,unverified", args)

    def test_remember_writes_the_artifact_facet_and_body_line(self):
        out = self.remember(artifact="/tmp/dafeiyu-report.md")
        self.assertIn("stored [abcd1234]", out)
        args = self.args_for("remember")[0]
        self.assertIn("--type knowledge", args)
        self.assertIn("--artifact", args)
        self.assertIn('"/tmp/dafeiyu-report.md"', args)
        self.assertIn("【产物】/tmp/dafeiyu-report.md", args)
        self.assertIn("artifact", args.split("--tags")[1].split("--")[0])

    def test_remember_keeps_typed_kinds_when_the_gate_is_on(self):
        out = self.remember(memory_type="hypothesis")
        self.assertNotIn("v1 off", out)
        args = self.args_for("remember")[0]
        self.assertIn("--type hypothesis", args)

    def test_remember_degrades_typed_kinds_and_artifact_when_gate_is_off(self):
        plugin = self.gate_off_plugin()
        out = self.remember(plugin=plugin, memory_type="hypothesis", artifact="/tmp/x.md")
        self.assertIn("v1 off", out)
        args = self.args_for("remember")[0]
        self.assertNotIn("--type hypothesis", args)
        self.assertNotIn("--artifact", args)
        self.assertIn("qq,hypothesis", args)
        self.assertIn("【产物】/tmp/x.md", args)

    def test_remember_passes_a_write_time_summary_when_the_gate_is_on(self):
        # T1-3: the reason clause (a speaker qualifier) must reach --summary.
        out = self.remember()
        self.assertIn("stored [abcd1234]", out)
        args = self.args_for("remember")[0]
        self.assertIn("--summary", args)
        summary = args.split("--summary ", 1)[1]
        self.assertIn("the bridge log ships to /tmp/bridge.log", summary)
        self.assertIn("operator said so", summary)

    def test_remember_omits_the_summary_when_the_gate_is_off(self):
        # --summary forces contract 1, so a gate-off host must not receive it.
        plugin = self.gate_off_plugin()
        out = self.remember(plugin=plugin)
        self.assertIn("stored [abcd1234]", out)
        args = self.args_for("remember")[0]
        self.assertNotIn("--summary", args)

    def test_update_rewrites_the_summary_on_the_same_id(self):
        self.inject("why did the napcat bridge crash yesterday")
        out = asyncio.run(
            self.plugin.memory_update_tool(
                FakeEvent("x"),
                "92cfc0da",
                "the bridge listens on 8080 now",
                "operator corrected the port yesterday",
                "",
                "",
            )
        )
        self.assertIn("updated [92cfc0da]", out)
        args = self.args_for("update")[0]
        # Same id, one flag: the CLI replaces the summary, never stacks it.
        self.assertEqual(args.count("--summary"), 1)
        summary = args.split("--summary ", 1)[1]
        self.assertIn("8080", summary)
        self.assertIn("yesterday", summary)  # time qualifier from the reason

    def test_update_omits_the_summary_when_the_gate_is_off(self):
        plugin = self.gate_off_plugin()
        # _may_edit resolves an id through THIS instance's id→bucket map, which
        # only a recall on this instance can populate (it is in-memory state).
        asyncio.run(
            plugin.on_llm_request(
                FakeEvent("why did the napcat bridge crash yesterday"), FakeReq()
            )
        )
        out = asyncio.run(
            plugin.memory_update_tool(
                FakeEvent("x"),
                "92cfc0da",
                "the bridge listens on 8080 now",
                "operator corrected the port yesterday",
                "",
                "",
            )
        )
        self.assertIn("updated [92cfc0da]", out)
        self.assertNotIn("--summary", self.args_for("update")[0])

    def test_update_title_only_touch_omits_the_summary(self):
        # No body change: the existing summary stays untouched (we cannot
        # rebuild its qualifiers without the body).
        self.inject("why did the napcat bridge crash yesterday")
        out = asyncio.run(
            self.plugin.memory_update_tool(FakeEvent("x"), "92cfc0da", "", "", "new title", "")
        )
        self.assertIn("updated [92cfc0da]", out)
        self.assertNotIn("--summary", self.args_for("update")[0])


# A top-1 entry that is unverified, expired, has a superseded direct source and
# a stale artifact: every P0-1/P2-1/P2-3 annotation at once. The shown entry is
# a Hypothesis with a parent, so the plugin must fetch the parent separately
# (the live ancestor projection only carries id+title).
STATUS_SHOW_PAYLOAD = {
    "command": "show",
    "status": "ok",
    "summary": {"id": "92cfc0da-1111-2222-3333-444444444444", "title": "first remembered title"},
    "items": [{"name": "entry", "status": "ok", "value": "92cfc0da"}],
    "details": {
        "entry": {
            "content": "STATUS_BODY_HEAD verbatim excerpt body",
            "title": "first remembered title",
            "kind": "Hypothesis",
            "parent_id": "aaaa1111-2222-3333-4444-555555555555",
            "valid_until": "2026-01-01T00:00:00.000Z",
            "review_at": "2026-01-02T00:00:00.000Z",
            "artifact": {"path": "/nonexistent/dafeiyu-probe.txt", "purpose": "probe"},
        },
    },
    "errors": [],
}

PARENT_SHOW_PAYLOAD = {
    "command": "show",
    "status": "ok",
    "summary": {"id": "aaaa1111-2222-3333-4444-555555555555", "title": "source fact"},
    "items": [{"name": "entry", "status": "ok", "value": "aaaa1111"}],
    "details": {
        "entry": {
            "id": "aaaa1111-2222-3333-4444-555555555555",
            "content": "source body",
            "title": "source fact",
            "kind": "Context",
            "superseded_by": "bbbb2222-3333-4444-5555-666666666666",
        },
    },
    "errors": [],
}


class ShowStatusTest(Harness):
    """P0-1/P2-1/P2-3 annotations on show and on the injected top-1 line."""

    def _make_binary(self):
        """Fake rsrs whose ``show`` answers by id: top-1 vs its parent."""
        recall = json.dumps(json.dumps(RECALL_PAYLOAD))
        status_show = json.dumps(json.dumps(STATUS_SHOW_PAYLOAD))
        parent_show = json.dumps(json.dumps(PARENT_SHOW_PAYLOAD))
        remember = json.dumps(json.dumps(REMEMBER_PAYLOAD))
        metadata = json.dumps(json.dumps(self.metadata_payload))
        learn_payload = json.dumps(json.dumps({"status": "ok", "summary": {"bge_available": False}}))
        ok = json.dumps(json.dumps({"command": "ok", "status": "ok", "summary": {}, "items": [], "errors": []}))
        script = f"""#!/bin/sh
printf 'CMD=%s\\n' "$1" > {json.dumps(str(self.calls))}/$$.log
printf 'ARGS=' >> {json.dumps(str(self.calls))}/$$.log
printf '%s ' "$@" | tr '\\n' ' ' >> {json.dumps(str(self.calls))}/$$.log
printf '\\n' >> {json.dumps(str(self.calls))}/$$.log
case "$1" in
  recall) printf '%s' {recall} ;;
  show)
    if [ "$2" = "92cfc0da-1111-2222-3333-444444444444" ]; then
      printf '%s' {status_show}
    else
      printf '%s' {parent_show}
    fi ;;
  remember) printf '%s' {remember} ;;
  metadata) printf '%s' {metadata} ;;
  learn) printf '%s' {learn_payload}; exit 2 ;;
  *) printf '%s' {ok} ;;
esac
exit 0
"""
        path = self.tmp / "rsrs-status"
        path.write_text(script, encoding="utf-8")
        os.chmod(path, 0o755)
        return path

    def test_show_tool_prints_every_status_flag(self):
        self.inject("why did the napcat bridge crash yesterday")
        out = asyncio.run(self.plugin.memory_show_tool(FakeEvent("x"), "92cfc0da"))
        self.assertIn("状态: ", out)
        self.assertIn("未验证", out)
        self.assertIn("已过期 2026-01-01", out)
        self.assertIn("复核日期 2026-01-02", out)
        self.assertIn("来源 aaaa1111 已被 bbbb2222 替代", out)
        self.assertIn("派生结论待复核", out)
        self.assertIn("stale", out)
        self.assertIn("STATUS_BODY_HEAD", out)
        # inject (top-1 + parent) then the explicit show (record + parent).
        self.assertEqual(self.commands().count("show"), 4)

    def test_show_command_prints_the_status_line(self):
        self.inject("why did the napcat bridge crash yesterday")
        out = asyncio.run(self.plugin._show_text(FakeEvent("/mem_show 92cfc0da"), "92cfc0da"))
        self.assertIn("状态: ", out)
        self.assertIn("未验证", out)
        self.assertIn("派生结论待复核", out)

    def test_injected_top1_line_carries_the_status(self):
        req = self.inject("why did the napcat bridge crash yesterday")
        part = req.extra_user_content_parts[0]
        self.assertIn("\n  状态: ", part.text)
        self.assertIn("已过期 2026-01-01", part.text)
        self.assertIn("派生结论待复核", part.text)
        # The excerpt still renders; both lines share the single top-1 show
        # plus the bounded parent read.
        self.assertIn("摘要: STATUS_BODY_HEAD", part.text)
        self.assertEqual(self.commands().count("show"), 2)


if __name__ == "__main__":
    unittest.main()
