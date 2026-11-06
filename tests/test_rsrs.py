"""rsrs adapter tests: argv, discovery, parsing, timeouts, probes."""

from __future__ import annotations

import asyncio
import json
import os
import unittest

from helpers import TempDirTest, make_fake_rsrs

from dafeiyu_rsrs import (
    EnvironmentProbe,
    Rsrs,
    RsrsError,
    assert_id,
    audit_args,
    discover_binary,
    forget_args,
    is_valid_id,
    learn_lexical_args,
    learn_probe_args,
    learn_reset_args,
    learn_status_args,
    learn_train_args,
    parse_envelope,
    query_log_args,
    query_log_mark_args,
    readonly_error,
    recall_args,
    remember_args,
    resolve_query_log_id,
    review_due_args,
    run_process,
    show_args,
    spotcheck_list_args,
    status_args,
    update_args,
)


class ArgvTest(unittest.TestCase):
    def test_recall_argv(self):
        self.assertEqual(recall_args("deploy failed", titles=True, limit=5, project="qq:g"), [
            "recall", "deploy failed", "--json", "--titles", "--limit", "5", "--project", "qq:g",
        ])
        self.assertIn("--lexical", recall_args("q", lexical=True))
        self.assertNotIn("--lexical", recall_args("q", lexical=False))
        args = recall_args("q", type="hypothesis", intent="fact", brief=True)
        self.assertEqual(args[-4:], ["--intent", "fact", "--type", "hypothesis"])
        self.assertIn("--brief", args)

    def test_remember_argv(self):
        args = remember_args(
            "body", title="t", importance="important", type="Context", tags="qq,fact",
            project="qq:p", valid_until="2026-01-01T00:00:00.000Z", review_at="2026-01-01T00:00:00.000Z",
            parent="92cfc0da",
        )
        self.assertEqual(args[:2], ["remember", "body"])
        self.assertIn("--title", args)
        self.assertEqual(args[args.index("--importance") + 1], "important")
        self.assertEqual(args[args.index("--parent") + 1], "92cfc0da")
        self.assertEqual(args[args.index("--tags") + 1], "qq,fact")
        self.assertIn("--valid-until", args)
        self.assertIn("--review-at", args)

    def test_remember_supersedes_argv(self):
        args = remember_args("body", supersedes="92cfc0da-1111-2222-3333-444444444444")
        self.assertEqual(args[args.index("--supersedes") + 1], "92cfc0da-1111-2222-3333-444444444444")
        self.assertNotIn("--force", args)
        self.assertNotIn("--supersedes", remember_args("body"))
        with self.assertRaises(RsrsError):
            remember_args("body", supersedes="not-an-id")

    def test_remember_summary_argv(self):
        # T1-3: --summary rides remember and update; empty values are omitted.
        args = remember_args("body", title="t", summary="标题；首句；限定")
        self.assertEqual(args[args.index("--summary") + 1], "标题；首句；限定")
        self.assertNotIn("--summary", remember_args("body", title="t"))
        args = update_args("92cfc0da", content="c", summary="s")
        self.assertEqual(args[args.index("--summary") + 1], "s")
        self.assertNotIn("--summary", update_args("92cfc0da", content="c"))
        self.assertNotIn("--summary", update_args("92cfc0da", title="t"))

    def test_remember_artifact_argv(self):
        # A dict is encoded compactly; a pre-serialized string passes through.
        args = remember_args("body", artifact={"path": "/tmp/x.md", "purpose": "p"})
        payload = args[args.index("--artifact") + 1]
        self.assertEqual(payload, '{"path":"/tmp/x.md","purpose":"p"}')
        self.assertEqual(
            remember_args("body", artifact='{"path":"/tmp/y.md"}')[
                remember_args("body", artifact='{"path":"/tmp/y.md"}').index("--artifact") + 1
            ],
            '{"path":"/tmp/y.md"}',
        )
        self.assertNotIn("--artifact", remember_args("body"))

    def test_other_argv(self):
        self.assertEqual(status_args(), ["status", "--json"])
        self.assertEqual(learn_status_args(), ["learn", "status", "--json"])
        self.assertEqual(learn_probe_args(), ["learn", "--help"])
        self.assertEqual(
            learn_train_args(limit=10, epochs=1),
            ["learn", "train", "--from-labels", "--json", "--limit", "10", "--epochs", "1"],
        )
        self.assertEqual(learn_reset_args(), ["learn", "reset", "--json"])
        self.assertEqual(learn_lexical_args(True), ["learn", "lexical", "--json"])
        self.assertEqual(learn_lexical_args(False), ["learn", "lexical", "--off", "--json"])
        self.assertEqual(show_args("92cfc0da"), ["show", "92cfc0da", "--json"])
        self.assertEqual(forget_args("92cfc0da"), ["forget", "92cfc0da", "--json"])
        self.assertEqual(query_log_args(3), ["query-log", "--limit", "3", "--json"])

    def test_audit_and_review_argv(self):
        self.assertEqual(audit_args(), ["audit", "--json"])
        self.assertEqual(review_due_args(10), ["review", "--due", "--json", "--limit", "10"])
        self.assertEqual(review_due_args()[-1], "20")
        self.assertEqual(review_due_args("7")[-1], "7")
        self.assertEqual(review_due_args("nonsense")[-1], "20")
        # Clamp to 1..100.
        self.assertEqual(review_due_args(0)[-1], "1")
        self.assertEqual(review_due_args(-5)[-1], "1")
        self.assertEqual(review_due_args(10**6)[-1], "100")
        # Read-only argv: no mutating flags or subcommands.
        for args in (audit_args(), review_due_args(5)):
            for value in args:
                self.assertNotIn(value, ("forget", "purge", "delete", "update", "--force"))

    def test_spotcheck_list_argv(self):
        self.assertEqual(spotcheck_list_args(), ["list", "--limit", "10000", "--json"])
        self.assertEqual(
            spotcheck_list_args(500, metadata=True),
            ["list", "--metadata", "--limit", "500", "--json"],
        )
        # Clamp to 1..50000; junk falls back to the default.
        self.assertEqual(spotcheck_list_args(0)[2], "1")
        self.assertEqual(spotcheck_list_args(10**9)[2], "50000")
        self.assertEqual(spotcheck_list_args("nonsense")[2], "10000")
        for args in (spotcheck_list_args(), spotcheck_list_args(metadata=True)):
            for value in args:
                self.assertNotIn(value, ("forget", "purge", "delete", "update", "--force"))

    def test_mark_argv(self):
        args = query_log_mark_args(["92cfc0da", "b43746e5"], "good", query_log_id="uuid-1", source="human")
        self.assertEqual(args[:5], ["query-log", "--json", "mark", "92cfc0da,b43746e5", "--good"])
        self.assertEqual(args[args.index("--query-log-id") + 1], "uuid-1")
        self.assertEqual(args[args.index("--source") + 1], "human")
        self.assertIn("--bad", query_log_mark_args(["92cfc0da"], "bad"))
        with self.assertRaises(RsrsError):
            query_log_mark_args([], "good")

    def test_id_validation(self):
        self.assertTrue(is_valid_id("92cfc0da"))
        self.assertFalse(is_valid_id("../../etc/passwd"))
        self.assertFalse(is_valid_id("short"))
        with self.assertRaises(RsrsError):
            assert_id("nope")
        with self.assertRaises(RsrsError):
            show_args("--help")


class DiscoveryTest(TempDirTest):
    def test_order(self):
        home = self.tmp / "home"
        (home / ".local" / "bin").mkdir(parents=True)
        (home / ".rsrs" / "bin").mkdir(parents=True)
        path_dir = self.tmp / "pathbin"
        path_dir.mkdir()
        env = {"HOME": str(home), "PATH": str(path_dir)}

        # Nothing installed.
        found = discover_binary({}, env, str(home))
        self.assertEqual(found, {"bin": None, "source": "unresolved"})

        on_path = make_fake_rsrs(path_dir)
        self.assertEqual(discover_binary({}, env, str(home)), {"bin": on_path, "source": "path"})

        conventional = home / ".rsrs" / "bin" / "rsrs"
        conventional.write_text("#!/bin/sh\nexit 0\n")
        os.chmod(conventional, 0o755)
        self.assertEqual(discover_binary({}, env, str(home))["source"], "home")

        managed = home / ".local" / "bin" / "rsrs-dafeiyu"
        managed.write_text("#!/bin/sh\nexit 0\n")
        os.chmod(managed, 0o755)
        self.assertEqual(discover_binary({}, env, str(home))["source"], "managed")

        env_with_bin = dict(env, RSRS_BIN=str(on_path))
        self.assertEqual(discover_binary({}, env_with_bin, str(home))["source"], "env")

        self.assertEqual(
            discover_binary({"rsrs_bin": str(conventional)}, env_with_bin, str(home)),
            {"bin": str(conventional), "source": "config"},
        )

    def test_configured_but_missing_is_reported(self):
        found = discover_binary({"rsrs_bin": str(self.tmp / "nope")}, {"HOME": str(self.tmp)}, str(self.tmp))
        self.assertEqual(found["source"], "config-unverified")
        self.assertEqual(found["bin"], str(self.tmp / "nope"))

    def test_tilde_expansion(self):
        home = self.tmp / "home"
        (home / ".local" / "bin").mkdir(parents=True)
        managed = home / ".local" / "bin" / "rsrs-dafeiyu"
        managed.write_text("#!/bin/sh\nexit 0\n")
        os.chmod(managed, 0o755)
        found = discover_binary({}, {"HOME": str(home)}, str(home))
        self.assertEqual(found["source"], "managed")


class ParseEnvelopeTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_envelope('{"status":"ok"}'), {"status": "ok"})

    def test_empty_and_invalid(self):
        with self.assertRaises(RsrsError) as caught:
            parse_envelope("")
        self.assertEqual(caught.exception.code, "EMPTY_OUTPUT")
        with self.assertRaises(RsrsError) as caught:
            parse_envelope("not json")
        self.assertEqual(caught.exception.code, "INVALID_JSON")
        with self.assertRaises(RsrsError) as caught:
            parse_envelope("[1,2]")
        self.assertEqual(caught.exception.code, "INVALID_ENVELOPE")


class ReadonlyErrorTest(unittest.TestCase):
    def test_warn_with_nonzero_exit_is_not_a_failure(self):
        # rsrs 1.0.19 returns exit 2 with a complete "warn" audit envelope.
        result = {
            "envelope": {"status": "warn", "summary": {"hot_untyped": 870}, "errors": []},
            "code": 2,
            "stderr": "",
        }
        self.assertEqual(readonly_error(result), "")
        self.assertEqual(readonly_error({"envelope": {"status": "due"}, "code": 0}), "")
        self.assertEqual(readonly_error({"envelope": {"status": "skip"}, "code": 1}), "")

    def test_error_envelope_is_a_failure(self):
        result = {
            "envelope": {
                "status": "fail",
                "summary": {"saved": False},
                "errors": ["decode review snapshot: nonce length must be 12 bytes"],
            },
            "code": 1,
        }
        self.assertIn("decode review snapshot", readonly_error(result))
        self.assertEqual(readonly_error({"envelope": {"status": "boom"}}), "boom")

    def test_missing_envelope_is_a_failure(self):
        self.assertEqual(readonly_error({"envelope": None, "code": 3, "stderr": "boom"}), "boom")
        self.assertEqual(readonly_error({"envelope": None, "code": 3, "stderr": ""}), "exit 3")
        self.assertEqual(readonly_error(None), "no result")
        self.assertEqual(
            readonly_error({"envelope": {"summary": {}}, "code": 4, "stderr": "bad"}), "bad"
        )


class RunTest(TempDirTest):
    def make(self, **kwargs):
        binary = make_fake_rsrs(self.tmp, **kwargs)
        config = {"rsrs_bin": str(binary), "data_dir": ""}
        return Rsrs(config, env={"PATH": "/usr/bin:/bin"})

    def test_success(self):
        rsrs = self.make(payload={"command": "status", "status": "ok", "summary": {"local_total": 3}})
        result = asyncio.run(rsrs.run(status_args()))
        self.assertEqual(result["envelope"]["summary"]["local_total"], 3)
        self.assertEqual(result["code"], 0)

    def test_nonzero_with_error_envelope_raises(self):
        rsrs = self.make(payload={"command": "recall", "status": "error", "errors": [{"message": "no index"}]})
        with self.assertRaises(RsrsError) as caught:
            asyncio.run(rsrs.run(recall_args("q")))
        self.assertEqual(caught.exception.code, "RSRS_FAILED")
        self.assertIn("no index", str(caught.exception))

    def test_nonzero_without_json_uses_stderr(self):
        rsrs = self.make(stderr="boom", exit_code=3)
        with self.assertRaises(RsrsError) as caught:
            asyncio.run(rsrs.run(status_args()))
        self.assertEqual(caught.exception.code, "RSRS_FAILED")
        self.assertIn("boom", str(caught.exception))

    def test_skip_status_is_not_a_failure(self):
        # A real build returns status "skip" when the scope has no candidates.
        rsrs = self.make(payload={"command": "recall", "status": "skip", "summary": {"count": 0}, "items": []})
        result = asyncio.run(rsrs.run(recall_args("q", project="qq:x")))
        self.assertEqual(result["envelope"]["status"], "skip")
        self.assertEqual(result["envelope"]["items"], [])

    def test_error_status_still_raises(self):
        rsrs = self.make(payload={"command": "recall", "status": "error", "errors": [{"message": "bad"}]})
        with self.assertRaises(RsrsError):
            asyncio.run(rsrs.run(recall_args("q")))

    def test_allow_failure_returns_envelope_none(self):
        rsrs = self.make(stderr="boom", exit_code=3)
        result = asyncio.run(rsrs.run(status_args(), allow_failure=True))
        self.assertIsNone(result["envelope"])
        self.assertEqual(result["code"], 3)

    def test_timeout_kills_the_child(self):
        rsrs = self.make(sleep=2, payload={"status": "ok"})
        with self.assertRaises(RsrsError) as caught:
            asyncio.run(rsrs.run(status_args(), timeout_ms=100))
        self.assertEqual(caught.exception.code, "TIMEOUT")

    def test_missing_binary(self):
        rsrs = Rsrs({"rsrs_bin": str(self.tmp / "absent"), "data_dir": ""}, env={"PATH": ""})
        with self.assertRaises(RsrsError) as caught:
            asyncio.run(rsrs.run(status_args()))
        self.assertEqual(caught.exception.code, "SPAWN_FAILED")

    def test_no_binary_found(self):
        rsrs = Rsrs({"rsrs_bin": "", "data_dir": ""}, env={"PATH": "", "HOME": str(self.tmp / "empty")})
        with self.assertRaises(RsrsError) as caught:
            asyncio.run(rsrs.run(status_args()))
        self.assertEqual(caught.exception.code, "BINARY_NOT_FOUND")

    def test_data_dir_is_exported_to_the_child(self):
        marker = self.tmp / "env.txt"
        script = self.tmp / "rsrs"
        script.write_text(
            "#!/bin/sh\nprintf '%s' \"$RSRS_DATA_DIR\" > " + json.dumps(str(marker)) + "\n"
            "printf '%s' '{\"status\":\"ok\"}'\nexit 0\n"
        )
        os.chmod(script, 0o755)
        rsrs = Rsrs({"rsrs_bin": str(script), "data_dir": "/tmp/rsrs-data"}, env={"PATH": "/usr/bin:/bin"})
        asyncio.run(rsrs.run(status_args()))
        self.assertEqual(marker.read_text(), "/tmp/rsrs-data")

    def test_rpc_port_is_exported_only_with_data_dir(self):
        # A non-default data dir (rsrs space) must pin its own RPC port so the
        # child cannot share the machine's single 15169 runtime (82989fbe:
        # writes would silently land in the real library).
        marker = self.tmp / "env.txt"
        script = self.tmp / "rsrs"
        script.write_text(
            "#!/bin/sh\nprintf 'dir=%s port=%s' \"$RSRS_DATA_DIR\" \"$RSRS_RPC_PORT\" > "
            + json.dumps(str(marker))
            + "\nprintf '%s' '{\"status\":\"ok\"}'\nexit 0\n"
        )
        os.chmod(script, 0o755)

        both = Rsrs(
            {"rsrs_bin": str(script), "data_dir": "/tmp/rsrs-data", "rpc_port": 15270},
            env={"PATH": "/usr/bin:/bin"},
        )
        asyncio.run(both.run(status_args()))
        self.assertEqual(marker.read_text(), "dir=/tmp/rsrs-data port=15270")

        # Empty data_dir: neither variable is injected — the child keeps the
        # machine default profile and runtime.
        default = Rsrs(
            {"rsrs_bin": str(script), "data_dir": "", "rpc_port": 15270},
            env={"PATH": "/usr/bin:/bin"},
        )
        asyncio.run(default.run(status_args()))
        self.assertEqual(marker.read_text(), "dir= port=")

        # data_dir set but rpc_port 0: data dir yes, port no (opt-out).
        optout = Rsrs(
            {"rsrs_bin": str(script), "data_dir": "/tmp/rsrs-data", "rpc_port": 0},
            env={"PATH": "/usr/bin:/bin"},
        )
        asyncio.run(optout.run(status_args()))
        self.assertEqual(marker.read_text(), "dir=/tmp/rsrs-data port=")

    def test_argv_is_not_a_shell_string(self):
        # A query containing shell metacharacters must arrive as one argument.
        capture = self.tmp / "args.txt"
        script = self.tmp / "rsrs"
        script.write_text(
            "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\" >> " + json.dumps(str(capture)) + "; done\n"
            "printf '%s' '{\"status\":\"ok\"}'\nexit 0\n"
        )
        os.chmod(script, 0o755)
        rsrs = Rsrs({"rsrs_bin": str(script), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        asyncio.run(rsrs.run(recall_args("a; rm -rf / && echo $(id)")))
        lines = capture.read_text().splitlines()
        self.assertIn("a; rm -rf / && echo $(id)", lines)
        self.assertNotIn("rm", lines)


class ProbeTest(TempDirTest):
    def test_learn_probe_and_cache(self):
        counter = self.tmp / "count"
        script = self.tmp / "rsrs"
        script.write_text(
            "#!/bin/sh\nprintf 'x' >> " + json.dumps(str(counter)) + "\n"
            "case \"$1\" in\n  learn) exit 0 ;;\n  *) printf '%s' '{\"status\":\"ok\"}'; exit 0 ;;\nesac\n"
        )
        os.chmod(script, 0o755)
        rsrs = Rsrs({"rsrs_bin": str(script), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        probe = asyncio.run(rsrs.probe_learn())
        self.assertTrue(probe["learn_available"])
        asyncio.run(rsrs.probe_learn())
        self.assertEqual(counter.read_text(), "x")  # cached for 5 minutes
        asyncio.run(rsrs.probe_learn(force=True))
        self.assertEqual(counter.read_text(), "xx")

    def test_metadata_status_parses_and_caches_only_known_answers(self):
        counter = self.tmp / "meta-count"
        script = self.tmp / "rsrs"
        script.write_text(
            "#!/bin/sh\nprintf 'x' >> " + json.dumps(str(counter)) + "\n"
            "case \"$1\" in\n"
            "  metadata) printf '%s' '{\"status\":\"ok\",\"summary\":{\"enabled\":true,\"mode\":\"v1\"}}'; exit 0 ;;\n"
            "  *) printf '%s' '{\"status\":\"ok\"}'; exit 0 ;;\n"
            "esac\n"
        )
        os.chmod(script, 0o755)
        rsrs = Rsrs({"rsrs_bin": str(script), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        first = asyncio.run(rsrs.metadata_status())
        self.assertTrue(first["enabled"])
        self.assertTrue(first["known"])
        asyncio.run(rsrs.metadata_status())
        self.assertEqual(counter.read_text(), "x")  # cached

        # A broken probe is unknown, not disabled, and is not cached.
        broken = self.tmp / "broken-rsrs"
        broken.write_text("#!/bin/sh\nprintf 'not-json'; exit 0\n")
        os.chmod(broken, 0o755)
        other = Rsrs({"rsrs_bin": str(broken), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        unknown = asyncio.run(other.metadata_status())
        self.assertFalse(unknown["enabled"])
        self.assertFalse(unknown["known"])

    def test_learn_probe_failure_is_reported(self):
        binary = make_fake_rsrs(self.tmp, stderr="error: unrecognized subcommand 'learn'", exit_code=2)
        rsrs = Rsrs({"rsrs_bin": str(binary), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        probe = asyncio.run(rsrs.probe_learn())
        self.assertFalse(probe["learn_available"])
        self.assertIn("learn", probe["reason"] or "")

    def test_environment_probe_falls_back_to_status(self):
        binary = make_fake_rsrs(self.tmp, stderr="no learn", exit_code=2)
        rsrs = Rsrs({"rsrs_bin": str(binary), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        # learn probe fails, so bge comes from the status string probe (also this script's stderr).
        value = asyncio.run(EnvironmentProbe(rsrs).probe())
        self.assertFalse(value["learn_available"])
        self.assertFalse(value["bge_available"])
        self.assertIsNotNone(value["warning"])

    def test_environment_probe_reports_cli_mode(self):
        script = self.tmp / "rsrs"
        script.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  learn) printf '%s' '{\"status\":\"ok\",\"summary\":{\"bge_available\":true,\"lexical\":true}}'; exit 0 ;;\n"
            "  *) printf '%s' '{\"status\":\"ok\"}'; exit 0 ;;\n"
            "esac\n"
        )
        os.chmod(script, 0o755)
        rsrs = Rsrs({"rsrs_bin": str(script), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        value = asyncio.run(EnvironmentProbe(rsrs).probe())
        self.assertEqual(value["cli_mode"], "lexical")
        self.assertTrue(value["bge_available"])

    def test_environment_probe_reads_bge_from_learn_status(self):
        script = self.tmp / "rsrs"
        script.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  learn) printf '%s' '{\"status\":\"ok\",\"summary\":{\"bge_available\":true}}'; exit 0 ;;\n"
            "  *) printf '%s' '{\"status\":\"ok\"}'; exit 0 ;;\n"
            "esac\n"
        )
        os.chmod(script, 0o755)
        rsrs = Rsrs({"rsrs_bin": str(script), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        value = asyncio.run(EnvironmentProbe(rsrs).probe())
        self.assertTrue(value["learn_available"])
        self.assertTrue(value["bge_available"])
        self.assertIsNone(value["warning"])


class QueryLogTest(TempDirTest):
    def test_direct_query_log_id(self):
        binary = make_fake_rsrs(self.tmp, payload={"status": "ok"})
        rsrs = Rsrs({"rsrs_bin": str(binary), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        envelope = {"status": "ok", "summary": {"query_log_id": "uuid-9"}}
        self.assertEqual(asyncio.run(resolve_query_log_id(rsrs, envelope, ["92cfc0da"])), "uuid-9")

    def test_fallback_requires_candidate_intersection(self):
        payload = {
            "status": "ok",
            "details": [{"request_id": "uuid-1", "candidates": ["92cfc0da-1111"]}],
        }
        binary = make_fake_rsrs(self.tmp, payload=payload)
        rsrs = Rsrs({"rsrs_bin": str(binary), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        envelope = {"status": "ok", "summary": {}}
        self.assertEqual(asyncio.run(resolve_query_log_id(rsrs, envelope, ["92cfc0da"])), "uuid-1")
        self.assertIsNone(asyncio.run(resolve_query_log_id(rsrs, envelope, ["deadbeef"])))

    def test_no_ids_means_no_lookup(self):
        binary = make_fake_rsrs(self.tmp, payload={"status": "ok"})
        rsrs = Rsrs({"rsrs_bin": str(binary), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})
        self.assertIsNone(asyncio.run(resolve_query_log_id(rsrs, {"summary": {}}, [])))


class CancellationTest(TempDirTest):
    def test_cancelled_run_reaps_the_child(self):
        """Regression: a cancelled spawn used to leave a live transport behind."""
        binary = make_fake_rsrs(self.tmp, sleep=5, payload={"status": "ok"})
        rsrs = Rsrs({"rsrs_bin": str(binary), "data_dir": ""}, env={"PATH": "/usr/bin:/bin"})

        async def go():
            task = asyncio.create_task(rsrs.run(status_args()))
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return "cancelled"
            return "finished"

        self.assertEqual(asyncio.run(go()), "cancelled")


class RunProcessTest(TempDirTest):
    def test_captures_streams(self):
        script = make_fake_rsrs(self.tmp, stdout="hello", stderr="warn", exit_code=0)
        result = asyncio.run(run_process(script, [], env={"PATH": "/usr/bin:/bin"}, timeout_ms=5000))
        self.assertEqual(result["stdout"], "hello")
        self.assertEqual(result["stderr"], "warn")
        self.assertEqual(result["code"], 0)


if __name__ == "__main__":
    unittest.main()
