"""rsrs (respire CLI) adapter for the AstrBot adapter.

Binary discovery, self-learning feature probing, argument builders and an async
JSON spawn wrapper.

Every command is spawned with an argument list (``create_subprocess_exec``,
never a shell), so memory text can never reach a shell parser.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import time
from pathlib import Path

try:
    from .dafeiyu_config import expand_path
    from .dafeiyu_policy import detect_bge
except ImportError:  # pragma: no cover - direct module import (tests, scripts)
    from dafeiyu_config import expand_path
    from dafeiyu_policy import detect_bge

#: Recall must never delay a turn.
RECALL_TIMEOUT_MS = 1800
#: Ordinary read/write commands.
ORDINARY_TIMEOUT_MS = 15000
#: ``learn train --from-labels``.
TRAIN_TIMEOUT_MS = 120000
#: Weekly Granite encoder pipeline.
ENCODER_TIMEOUT_MS = 2 * 60 * 60 * 1000
#: Read-only ``audit --json`` on a large store.
AUDIT_TIMEOUT_MS = 120000
#: ``review --due --json`` listing due records.
REVIEW_TIMEOUT_MS = 60000
#: Read-only full-directory ``list`` passes for the nightly spot-check.
SPOTCHECK_TIMEOUT_MS = 60000
#: stderr is truncated to this many characters before it reaches a log.
STDERR_LIMIT = 2000
#: Memory ids are 8-64 hex characters; anything else is never passed to the CLI.
ID_RE = re.compile(
    r"^(?:[0-9a-f]{8,64}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)

#: Detection results are cached for this long.
DETECT_TTL_MS = 5 * 60 * 1000


class RsrsError(Exception):
    """Failure raised when the CLI cannot run or returns an unusable envelope."""

    def __init__(self, message, *, code="RSRS_ERROR", bin_path=None, argv=None, exit_code=None, stderr=""):
        super().__init__(message)
        self.code = code
        self.bin = bin_path
        # Named `argv`, never `args`: Exception.args is the constructor tuple and
        # overwriting it would replace the exception message with a list repr.
        self.argv = list(argv or [])
        self.exit_code = exit_code
        self.stderr = stderr


def is_valid_id(value):
    return isinstance(value, str) and bool(ID_RE.match(value.strip()))


def assert_id(value, label="id"):
    if not is_valid_id(value):
        raise RsrsError(f"invalid {label}: expected 8-64 hex characters", code="INVALID_ID")
    return value.strip()


def short_id(value):
    return str(value or "").strip()[:8]


def truncate(text, limit=STDERR_LIMIT):
    value = text if isinstance(text, str) else ""
    return value if len(value) <= limit else value[:limit] + "…"


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def is_executable(file_path):
    try:
        info = os.stat(file_path)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and os.access(file_path, os.X_OK)


def find_on_path(name, env=None):
    env = env if isinstance(env, dict) else os.environ
    for directory in str(env.get("PATH", "")).split(os.pathsep):
        if directory == "":
            continue
        candidate = os.path.join(directory, name)
        if is_executable(candidate):
            return candidate
    return None


def discover_binary(config=None, env=None, home=None, check=is_executable):
    """Resolve the CLI in the documented order.

    config ``rsrs_bin`` > ``$RSRS_BIN`` > ``~/.local/bin/rsrs-dafeiyu`` (managed
    self-learning build) > ``~/.rsrs/bin/rsrs`` > ``rsrs`` on PATH.

    Returns ``{"bin": str|None, "source": str}``.
    """
    env = dict(os.environ) if env is None else dict(env)
    config = config if isinstance(config, dict) else {}
    home_dir = str(home or env.get("HOME") or Path.home())

    configured = expand_path(config.get("rsrs_bin", "") or "", env)
    if configured:
        return {"bin": configured, "source": "config" if check(configured) else "config-unverified"}

    from_env = expand_path(env.get("RSRS_BIN", "") or "", env)
    if from_env and check(from_env):
        return {"bin": from_env, "source": "env"}

    # Managed copy: the npm `rsrs` shim re-materializes ~/.rsrs/bin/rsrs and
    # would otherwise silently revert a self-learning build.
    managed = os.path.join(home_dir, ".local", "bin", "rsrs-dafeiyu")
    if check(managed):
        return {"bin": managed, "source": "managed"}

    conventional = os.path.join(home_dir, ".rsrs", "bin", "rsrs")
    if check(conventional):
        return {"bin": conventional, "source": "home"}

    on_path = find_on_path("rsrs", env)
    if on_path:
        return {"bin": on_path, "source": "path"}
    return {"bin": None, "source": "unresolved"}


# --------------------------------------------------------------------------- #
# Argument builders
# --------------------------------------------------------------------------- #


def status_args():
    return ["status", "--json"]


def learn_status_args():
    return ["learn", "status", "--json"]


def learn_probe_args():
    """The self-learning feature probe: only that build accepts ``learn``."""
    return ["learn", "--help"]


def learn_train_args(limit=5000, epochs=1):
    return ["learn", "train", "--from-labels", "--json", "--limit", str(limit), "--epochs", str(epochs)]


def learn_reset_args():
    return ["learn", "reset", "--json"]


def learn_lexical_args(enabled=True):
    return ["learn", "lexical", "--json"] if enabled else ["learn", "lexical", "--off", "--json"]


def recall_args(query, *, titles=False, brief=False, limit=None, project=None, intent=None, type=None, lexical=False):
    args = ["recall", str(query or ""), "--json"]
    if titles:
        args.append("--titles")
    if brief:
        args.append("--brief")
    if limit is not None:
        args.extend(["--limit", str(limit)])
    if project:
        args.extend(["--project", str(project)])
    if intent:
        args.extend(["--intent", str(intent)])
    if type:
        args.extend(["--type", str(type)])
    if lexical:
        args.append("--lexical")
    return args


def remember_args(content, *, title=None, importance=None, type=None, parent=None, tags=None, project=None,
                  source_refs=None, valid_until=None, review_at=None, artifact=None, force=False, supersedes=None,
                  summary=None):
    args = ["remember", str(content or ""), "--json"]
    if title:
        args.extend(["--title", str(title)])
    if summary:
        # Contract-1 facet: callers pass this only when metadata v1 is on.
        args.extend(["--summary", str(summary)])
    if importance:
        args.extend(["--importance", str(importance)])
    if type:
        args.extend(["--type", str(type)])
    if parent:
        args.extend(["--parent", assert_id(parent, "parent")])
    if tags:
        args.extend(["--tags", str(tags)])
    if project:
        args.extend(["--project", str(project)])
    if source_refs is not None:
        args.extend(["--source-refs", json.dumps(source_refs, separators=(",", ":"))])
    if valid_until:
        args.extend(["--valid-until", str(valid_until)])
    if review_at:
        args.extend(["--review-at", str(review_at)])
    if artifact is not None:
        # A pre-serialized JSON object is passed through untouched so callers
        # control escaping; dicts are encoded here for convenience.
        payload = artifact if isinstance(artifact, str) else json.dumps(artifact, separators=(",", ":"))
        args.extend(["--artifact", payload])
    if supersedes:
        args.extend(["--supersedes", assert_id(supersedes, "supersedes")])
    if force:
        args.append("--force")
    return args


def show_args(memory_id):
    return ["show", assert_id(memory_id), "--json"]


def forget_args(memory_id):
    return ["forget", assert_id(memory_id), "--json"]


def chain_args(memory_id, depth=3):
    return ["chain", assert_id(memory_id), "--depth", str(max(0, int(depth))), "--json"]


def attach_args(child_id, parent_id):
    return ["attach", assert_id(child_id), "--parent", assert_id(parent_id), "--json"]


def update_args(memory_id, *, content=None, title=None, importance=None, tags=None, kind=None, summary=None):
    args = ["update", assert_id(memory_id), "--json"]
    if content is not None:
        args.extend(["--content", str(content)])
    if title is not None:
        args.extend(["--title", str(title)])
    if summary:
        # The CLI assigns the new summary on the same id (replace, never stack).
        args.extend(["--summary", str(summary)])
    if importance is not None:
        args.extend(["--importance", str(importance)])
    if tags is not None:
        args.extend(["--tags", str(tags)])
    if kind is not None:
        args.extend(["--kind", str(kind)])
    return args


def audit_args():
    """Read-only store audit. Never deletes, updates or forgets anything."""
    return ["audit", "--json"]


def review_due_args(limit=20):
    """Read-only due-review listing, with ``limit`` clamped to 1-100."""
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 20
    value = max(1, min(100, value))
    return ["review", "--due", "--json", "--limit", str(value)]


def list_args(limit=200):
    """Recent entries, used only to reconcile a stale self-note pointer."""
    return ["list", "--limit", str(max(1, min(500, int(limit)))), "--json"]


def spotcheck_list_args(limit=10000, metadata=False):
    """Read-only directory scan for the nightly spot-check.

    ``metadata=True`` selects the ``--metadata`` projection (fields such as
    ``valid_until`` / ``superseded_by`` without bodies); the default projection
    carries bodies for the P2-2 source check. The limit is clamped to 1-50000
    — one scan must stay a single bounded local read.
    """
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 10000
    value = max(1, min(50000, value))
    args = ["list"]
    if metadata:
        args.append("--metadata")
    return args + ["--limit", str(value), "--json"]


def metadata_status_args():
    """Read-only metadata-gate probe (typed writes need v1)."""
    return ["metadata", "status", "--json"]


def query_log_args(limit=20):
    return ["query-log", "--limit", str(limit), "--json"]


def query_log_mark_args(ids, verdict, *, query_log_id=None, source=None):
    items = [ids] if isinstance(ids, str) else list(ids or [])
    if not items:
        raise RsrsError("feedback requires at least one id", code="INVALID_ARGS")
    # `query-log mark` has no --json flag of its own; the parent query-log
    # command must see it before the mark subcommand.
    args = [
        "query-log",
        "--json",
        "mark",
        ",".join(assert_id(item) for item in items),
        "--bad" if verdict == "bad" else "--good",
    ]
    if query_log_id:
        args.extend(["--query-log-id", str(query_log_id)])
    if source:
        args.extend(["--source", str(source)])
    return args


# --------------------------------------------------------------------------- #
# Spawn wrapper
# --------------------------------------------------------------------------- #


async def _kill_and_reap(process, readers=()):
    """Kill a child, wait for it and cancel its pipe readers.

    Without the explicit wait the transport stays open and an event loop that is
    closing (plugin reload, test teardown) blocks in ``select`` or logs
    "Event loop is closed" from the child watcher.
    """
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), 5)
    except (asyncio.TimeoutError, ProcessLookupError, asyncio.CancelledError):
        pass
    for reader in readers:
        if not reader.done():
            reader.cancel()
    for reader in readers:
        try:
            await reader
        except (asyncio.CancelledError, Exception):
            pass


async def run_process(bin_path, args, *, env=None, cwd=None, timeout_ms=ORDINARY_TIMEOUT_MS):
    """Run one command with an argument list and capture its output.

    The child is killed on timeout or cancellation and always reaped. Never
    raises for a nonzero exit; the caller inspects ``code``.
    """
    argv = [str(bin_path)] + [str(item) for item in args]
    # `create_subprocess_exec` leaks a live transport when it is cancelled while
    # spawning, so the spawn itself runs in a shielded task: on cancellation we
    # let it finish, then kill and reap the child.
    spawn = asyncio.ensure_future(
        asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
    )
    try:
        process = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        try:
            spawned = await spawn
        except BaseException:
            spawned = None
        if spawned is not None:
            await _kill_and_reap(spawned)
        raise
    except FileNotFoundError as exc:
        raise RsrsError(f"failed to spawn {bin_path}: {exc}", code="SPAWN_FAILED", bin_path=bin_path, argv=args) from exc
    except OSError as exc:
        raise RsrsError(f"failed to spawn {bin_path}: {exc}", code="SPAWN_FAILED", bin_path=bin_path, argv=args) from exc

    stdout_reader = asyncio.ensure_future(process.stdout.read())
    stderr_reader = asyncio.ensure_future(process.stderr.read())
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.gather(stdout_reader, stderr_reader), max(1, timeout_ms) / 1000.0
        )
    except asyncio.TimeoutError:
        await _kill_and_reap(process, (stdout_reader, stderr_reader))
        raise RsrsError(
            f"rsrs timed out after {timeout_ms} ms",
            code="TIMEOUT",
            bin_path=bin_path,
            argv=args,
        ) from None
    except asyncio.CancelledError:
        await _kill_and_reap(process, (stdout_reader, stderr_reader))
        raise

    # Pipe EOF can arrive before the child watcher records the exit status, so
    # wait for the process itself before reading `returncode`.
    try:
        await asyncio.wait_for(process.wait(), 5)
    except (asyncio.TimeoutError, ProcessLookupError):
        pass

    return {
        "code": process.returncode,
        "stdout": (stdout or b"").decode("utf-8", "replace"),
        "stderr": (stderr or b"").decode("utf-8", "replace"),
        "timed_out": False,
    }


def parse_envelope(stdout, *, bin_path=None, argv=None):
    text = str(stdout or "").strip()
    if text == "":
        raise RsrsError("rsrs returned no output", code="EMPTY_OUTPUT", bin_path=bin_path, argv=argv)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RsrsError(
            f"rsrs returned invalid JSON: {exc}",
            code="INVALID_JSON",
            bin_path=bin_path,
            argv=argv,
            stderr=truncate(text),
        ) from exc
    if not isinstance(parsed, dict):
        raise RsrsError("rsrs returned a non-object envelope", code="INVALID_ENVELOPE", bin_path=bin_path, argv=argv)
    return parsed


def envelope_error(envelope):
    errors = envelope.get("errors") if isinstance(envelope, dict) else None
    messages = []
    if isinstance(errors, list):
        for entry in errors:
            if isinstance(entry, str):
                messages.append(entry)
            elif isinstance(entry, dict):
                value = entry.get("message") or entry.get("reason")
                if value:
                    messages.append(str(value))
    if messages:
        return "; ".join(messages)
    summary = envelope.get("summary") if isinstance(envelope, dict) else None
    if isinstance(summary, dict) and isinstance(summary.get("reason"), str):
        return summary["reason"]
    return ""


#: Statuses a read-only ``audit`` / ``review`` may legitimately report.
READONLY_STATUSES = ("ok", "skip", "warn", "due")


def readonly_error(result, ok_statuses=READONLY_STATUSES):
    """Return a short failure reason for a read-only result, or ``""``.

    Read-only commands may answer with ``warn`` (audit findings) or ``due``
    (pending reviews); those are normal results, not failures. ``rsrs audit``
    also exits non-zero (2 with findings on 1.0.19) while returning a complete
    envelope, so a nonzero exit is only a failure when the envelope itself is
    unusable: missing, carrying ``errors``, or reporting an unknown status.
    """
    if not isinstance(result, dict):
        return "no result"
    envelope = result.get("envelope")
    if not isinstance(envelope, dict):
        return truncate(result.get("stderr") or f"exit {result.get('code')}", 200)
    reason = envelope_error(envelope)
    if reason:
        return truncate(reason, 200)
    status = envelope.get("status")
    if isinstance(status, str) and status != "":
        if status not in ok_statuses:
            return truncate(status, 200)
        return ""
    code = result.get("code")
    if isinstance(code, int) and code != 0:
        return truncate(result.get("stderr") or f"exit {code}", 200)
    return ""


class Rsrs:
    """Cached handle around the rsrs CLI."""

    def __init__(self, config, env=None, now=None):
        self.config = config if isinstance(config, dict) else {}
        base_env = dict(os.environ)
        if env:
            base_env.update({str(key): str(value) for key, value in env.items()})
        # The data dir travels through the child env instead of a flag.
        data_dir = self.config.get("data_dir") or ""
        if data_dir:
            base_env["RSRS_DATA_DIR"] = data_dir
            # A non-default data dir (rsrs space profile) must not share the
            # machine's single 15169 runtime: that daemon serves whatever
            # client.json said when it booted, so writes would silently land in
            # the real library (82989fbe). Give the child its own RPC port so
            # it boots a dedicated runtime bound to this data dir.
            rpc_port = self.config.get("rpc_port")
            if rpc_port is not None and int(str(rpc_port).strip() or 0) > 0:
                base_env["RSRS_RPC_PORT"] = str(rpc_port)
        self.env = base_env
        self._now = now or (lambda: time.monotonic() * 1000.0)
        self._cache = None
        self._metadata_cache = None

    # -- discovery ---------------------------------------------------------
    def resolve(self, force=False):
        stamp = self._now()
        if not force and self._cache and stamp - self._cache["at"] < DETECT_TTL_MS:
            return self._cache
        found = discover_binary(self.config, self.env)
        self._cache = {
            "at": stamp,
            "bin": found["bin"],
            "source": found["source"],
            "learn_available": False,
            "learn_probed_at": None,
            "learn_reason": None,
        }
        return self._cache

    @property
    def bin(self):
        return self.resolve()["bin"]

    @property
    def source(self):
        return self.resolve()["source"]

    @property
    def learn_available(self):
        return bool(self._cache and self._cache.get("learn_available"))

    def invalidate(self):
        self._cache = None
        self._metadata_cache = None

    async def metadata_status(self, force=False):
        """Return ``{"enabled": bool, "mode": str}``; never raises.

        Typed writes (``--type``/``--supersedes``) need metadata v1. Each host
        may legitimately run with it off, so callers must degrade to tags and a
        state pointer instead of assuming it is on.
        """
        stamp = self._now()
        cached = self._metadata_cache
        if not force and cached and stamp - cached["at"] < DETECT_TTL_MS:
            return {"enabled": cached["enabled"], "mode": cached["mode"], "known": True}
        try:
            result = await self.run(metadata_status_args(), timeout_ms=ORDINARY_TIMEOUT_MS, allow_failure=True)
            envelope = result.get("envelope") if isinstance(result, dict) else None
            summary = envelope.get("summary") if isinstance(envelope, dict) else None
            if not isinstance(summary, dict):
                raise ValueError("no metadata summary")
            enabled = summary.get("enabled") is True
            mode = str(summary.get("mode") or "unknown")
        except Exception:
            # Unknown is not disabled: do not cache a transient probe failure,
            # otherwise a v1 host would silently fall back to legacy writes for
            # the whole cache window.
            return {"enabled": False, "mode": "unknown", "known": False}
        self._metadata_cache = {"at": stamp, "enabled": enabled, "mode": mode}
        return {"enabled": enabled, "mode": mode, "known": True}

    # -- probe -------------------------------------------------------------
    async def probe_learn(self, force=False):
        current = self.resolve()
        stamp = self._now()
        probed_at = current.get("learn_probed_at")
        if not force and probed_at is not None and stamp - probed_at < DETECT_TTL_MS:
            return {
                "learn_available": current["learn_available"],
                "bin": current["bin"],
                "source": current["source"],
                "reason": current.get("learn_reason"),
            }
        if not current["bin"]:
            current["learn_available"] = False
            current["learn_probed_at"] = stamp
            current["learn_reason"] = "no rsrs binary found"
            return {
                "learn_available": False,
                "bin": None,
                "source": current["source"],
                "reason": current["learn_reason"],
            }
        try:
            result = await run_process(current["bin"], learn_probe_args(), env=self.env, timeout_ms=ORDINARY_TIMEOUT_MS)
            ok = result["code"] == 0
            reason = None if ok else truncate(result["stderr"] or "learn --help failed")
        except RsrsError as exc:
            ok = False
            reason = truncate(exc.stderr or str(exc))
        current["learn_available"] = ok
        current["learn_probed_at"] = self._now()
        current["learn_reason"] = reason
        return {"learn_available": ok, "bin": current["bin"], "source": current["source"], "reason": reason}

    # -- execution ---------------------------------------------------------
    async def run(self, args, *, timeout_ms=ORDINARY_TIMEOUT_MS, allow_failure=False):
        """Run one command and return ``{"envelope", "code", "stdout", "stderr"}``."""
        current = self.resolve()
        if not current["bin"]:
            raise RsrsError(
                "rsrs binary not found (set rsrs_bin, RSRS_BIN, or install ~/.local/bin/rsrs-dafeiyu)",
                code="BINARY_NOT_FOUND",
                argv=args,
            )
        result = await run_process(current["bin"], args, env=self.env, timeout_ms=timeout_ms)
        try:
            envelope = parse_envelope(result["stdout"], bin_path=current["bin"], argv=args)
        except RsrsError as exc:
            if allow_failure:
                return {"envelope": None, "error": exc, **result}
            if result["code"] != 0:
                fallback = truncate(result["stderr"]) or "exit {}".format(result["code"])
                raise RsrsError(
                    "rsrs {} failed: {}".format(args[0], fallback),
                    code="RSRS_FAILED",
                    bin_path=current["bin"],
                    argv=args,
                    exit_code=result["code"],
                    stderr=truncate(result["stderr"]),
                ) from exc
            raise
        # `skip` is a normal empty result (no candidates in scope), not a failure.
        status_value = envelope.get("status")
        failed = result["code"] != 0 or (isinstance(status_value, str) and status_value not in ("ok", "skip"))
        if failed and not allow_failure:
            detail = envelope_error(envelope) or truncate(result["stderr"]) or f"exit {result['code']}"
            raise RsrsError(
                f"rsrs {args[0]} failed: {detail}",
                code="RSRS_FAILED",
                bin_path=current["bin"],
                argv=args,
                exit_code=result["code"],
                stderr=truncate(result["stderr"]),
            )
        return {"envelope": envelope, **result}


class EnvironmentProbe:
    """Cached ``{learn_available, bge_available}`` probe shared by hooks/commands."""

    def __init__(self, rsrs, now=None, ttl_ms=60000):
        self._rsrs = rsrs
        self._now = now or (lambda: time.monotonic() * 1000.0)
        self._ttl_ms = ttl_ms
        self._cache = None

    async def probe(self, force=False):
        stamp = self._now()
        if not force and self._cache and stamp - self._cache["at"] < self._ttl_ms:
            return self._cache["value"]
        learn = {"learn_available": False, "bin": None, "source": None, "reason": "probe failed"}
        try:
            learn = await self._rsrs.probe_learn()
        except Exception as exc:  # never let a probe break a turn
            learn = {
                "learn_available": False,
                "bin": self._rsrs.bin,
                "source": self._rsrs.source,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        bge_available = None
        cli_mode = None
        # The self-learning build reports both authoritatively in `learn status`.
        if learn.get("learn_available"):
            try:
                result = await self._rsrs.run(learn_status_args(), timeout_ms=RECALL_TIMEOUT_MS, allow_failure=True)
                envelope = result.get("envelope") or {}
                summary = envelope.get("summary") if isinstance(envelope, dict) else None
                if isinstance(summary, dict) and isinstance(summary.get("bge_available"), bool):
                    bge_available = summary["bge_available"]
                if isinstance(summary, dict) and isinstance(summary.get("lexical"), bool):
                    cli_mode = "lexical" if summary["lexical"] else "hybrid"
            except Exception:
                bge_available = None
        if bge_available is None:
            try:
                result = await self._rsrs.run(status_args(), timeout_ms=RECALL_TIMEOUT_MS, allow_failure=True)
                bge_available = detect_bge(result.get("envelope"))
            except Exception:
                bge_available = False
        value = {
            "learn_available": bool(learn.get("learn_available")),
            "bge_available": bool(bge_available),
            "cli_mode": cli_mode,
            "bin": learn.get("bin") or self._rsrs.bin,
            "source": learn.get("source") or self._rsrs.source,
            "warning": None
            if learn.get("learn_available")
            else f"self-learning build unavailable ({learn.get('reason') or 'unknown'})",
        }
        self._cache = {"at": stamp, "value": value}
        return value

    def invalidate(self):
        self._cache = None


async def resolve_query_log_id(rsrs, envelope, ids, *, timeout_ms=RECALL_TIMEOUT_MS):
    """Resolve the query-log id for a recall.

    Prefers ``summary.query_log_id``. When absent (older builds), reads the
    newest ``query-log`` row and accepts it only when its candidates intersect
    the recall ids, so feedback is never attached to an unrelated query.
    """
    summary = envelope.get("summary") if isinstance(envelope, dict) else None
    direct = summary.get("query_log_id") if isinstance(summary, dict) else None
    if isinstance(direct, str) and direct.strip() != "":
        return direct.strip()
    if not ids:
        return None
    try:
        result = await rsrs.run(query_log_args(1), timeout_ms=timeout_ms, allow_failure=True)
        log = result.get("envelope") or {}
        details = log.get("details") if isinstance(log, dict) else None
        row = details[0] if isinstance(details, list) and details else None
        if not isinstance(row, dict):
            return None
        candidates = {str(item)[:8] for item in (row.get("candidates") or [])}
        wanted = {str(item)[:8] for item in ids}
        if candidates & wanted and isinstance(row.get("request_id"), str):
            return row["request_id"]
        return None
    except Exception:
        return None
