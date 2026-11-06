"""Self-learning schedule for the AstrBot adapter.

Daily: ``rsrs learn train --from-labels`` so the locally learned residual
reranker keeps improving from query-log labels.
Weekly (Sunday 02:00 local): the respire Granite encoder pipeline
(prepare -> train -> reembed -> gate -> swap), run as an external script.
Nightly (read-only): ``rsrs audit --json``, ``rsrs review --due --json`` and two
``rsrs list`` directory scans for the plan spot-check (reflection-as-evidence,
expired-not-reviewed, missing source pointers — counts only, never bodies).
Maintenance only reads counters; it never deletes, updates or forgets a memory,
and it keeps running even when ``jobs.enabled`` is false.

Jobs are disabled by default in AstrBot because the DSH plugin may already own
the schedule. Manual ``/mem_learn train`` and ``/mem_flywheel`` still work when
the scheduler is off. Every job is guarded by one lock file under the plugin data
directory and records its last run.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging

try:
    from .dafeiyu_granite import encoder_status
    from .dafeiyu_policy import spotcheck_metadata, spotcheck_sources
    from .dafeiyu_rsrs import (
        AUDIT_TIMEOUT_MS,
        ENCODER_TIMEOUT_MS,
        REVIEW_TIMEOUT_MS,
        SPOTCHECK_TIMEOUT_MS,
        TRAIN_TIMEOUT_MS,
        audit_args,
        learn_train_args,
        readonly_error,
        review_due_args,
        run_process,
        spotcheck_list_args,
        truncate,
    )
    from .dafeiyu_state import append_log
except ImportError:  # pragma: no cover - direct module import (tests, scripts)
    from dafeiyu_granite import encoder_status
    from dafeiyu_policy import spotcheck_metadata, spotcheck_sources
    from dafeiyu_rsrs import (
        AUDIT_TIMEOUT_MS,
        ENCODER_TIMEOUT_MS,
        REVIEW_TIMEOUT_MS,
        SPOTCHECK_TIMEOUT_MS,
        TRAIN_TIMEOUT_MS,
        audit_args,
        learn_train_args,
        readonly_error,
        review_due_args,
        run_process,
        spotcheck_list_args,
        truncate,
    )
    from dafeiyu_state import append_log

LOGGER = logging.getLogger("astrbot_plugin_dafeiyu")

#: Catch-up runs start within this window after load.
CATCHUP_MS = 60 * 1000
#: A daily run is due again after this long.
DAILY_INTERVAL_MS = 24 * 60 * 60 * 1000
#: The read-only maintenance pass runs at most once per this window.
MAINTENANCE_INTERVAL_MS = 24 * 60 * 60 * 1000
#: Weekly encoder slot: Sunday 02:00 local time (``datetime.weekday`` Sunday = 6).
WEEKLY_WEEKDAY = 6
WEEKLY_HOUR = 2

#: Only these scalars from ``audit`` may ever reach ``jobs.json``.
AUDIT_SCALARS = (
    "status",
    "total_active",
    "orphans",
    "invalid_importance",
    "same_title_groups",
    "truncated_titles",
    "max_depth",
    "hot_untyped",
    "failing_skills",
    "skill_template_gaps",
    "hot_untyped_skipped",
    "skill_template_gaps_skipped",
)
#: Summary keys that may carry memory text; never persisted.
TEXT_KEYS = frozenset(
    {"title", "content", "body", "text", "desc", "description", "snippet", "preview", "excerpt", "titles"}
)


def ms_until_next(hour, minute, now_ms):
    """Milliseconds until the next local ``hour:minute``."""
    now = dt.datetime.fromtimestamp(now_ms / 1000.0)
    target = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return int((target - now).total_seconds() * 1000)


def ms_until_next_weekly(now_ms):
    """Milliseconds until the next Sunday 02:00 local time."""
    now = dt.datetime.fromtimestamp(now_ms / 1000.0)
    target = now.replace(hour=WEEKLY_HOUR, minute=0, second=0, microsecond=0)
    delta_days = (WEEKLY_WEEKDAY - target.weekday()) % 7
    target += dt.timedelta(days=delta_days)
    if target <= now:
        target += dt.timedelta(days=7)
    return int((target - now).total_seconds() * 1000)


def summarize_learn(envelope):
    """Keep only short scalar fields from a learn envelope — never memory text."""
    summary = envelope.get("summary") if isinstance(envelope, dict) else None
    if not isinstance(summary, dict):
        return {}
    out = {}
    for key, value in summary.items():
        if isinstance(value, bool) or isinstance(value, (int, float)):
            out[str(key)] = value
        elif isinstance(value, str) and len(value) <= 80:
            out[str(key)] = value
    return out


def _scalar(value):
    return isinstance(value, bool) or isinstance(value, (int, float))


def summarize_audit(envelope):
    """Whitelisted ``audit`` counters only.

    ``details`` (which contains hot-untyped ids and titles) is never copied, so
    the record that lands in ``jobs.json`` cannot leak memory text. The real
    ``rsrs audit`` puts the counters in ``summary`` and the verdict (``ok`` /
    ``warn``) at the envelope top level, so both places are read.
    """
    if not isinstance(envelope, dict):
        return {}
    source = envelope.get("summary")
    if not isinstance(source, dict):
        source = envelope
    out = {}
    for key in AUDIT_SCALARS:
        if key not in source:
            continue
        value = source[key]
        if _scalar(value):
            out[key] = value
        elif isinstance(value, str) and len(value) <= 40 and key == "status":
            out[key] = value
    if "status" not in out:
        top = envelope.get("status")
        if isinstance(top, str) and len(top) <= 40:
            out["status"] = top
    return out


def summarize_review(envelope):
    """Scalar ``review`` summary fields plus the item count — never titles."""
    if not isinstance(envelope, dict):
        return {}
    summary = envelope.get("summary")
    out = {}
    if isinstance(summary, dict):
        for key, value in summary.items():
            name = str(key)
            if name.lower() in TEXT_KEYS:
                continue
            if _scalar(value):
                out[name] = value
            elif isinstance(value, str) and len(value) <= 40:
                out[name] = value
    if "status" not in out:
        top = envelope.get("status")
        if isinstance(top, str) and len(top) <= 40:
            out["status"] = top
    items = envelope.get("items")
    out["items"] = len(items) if isinstance(items, list) else 0
    if "due" not in out:
        details = envelope.get("details")
        if isinstance(details, list):
            out["due"] = len(details)
    return out


def _details(result):
    """The ``details`` directory rows of one envelope, or ``[]``."""
    envelope = result.get("envelope") if isinstance(result, dict) else None
    details = envelope.get("details") if isinstance(envelope, dict) else None
    return details if isinstance(details, list) else []


class Jobs:
    """Daily reranker training, the weekly Granite pipeline and nightly audit."""

    def __init__(self, config, rsrs, state, logger=None, now=None):
        self._config = config if isinstance(config, dict) else {}
        self._jobs = self._config.get("jobs", {}) if isinstance(self._config.get("jobs"), dict) else {}
        self._rsrs = rsrs
        self._state = state
        self._logger = logger or LOGGER
        self._now = now or (lambda: int(dt.datetime.now().timestamp() * 1000))
        self._task = None

    @property
    def enabled(self):
        return bool(self._jobs.get("enabled"))

    @property
    def maintenance_enabled(self):
        return bool(self._jobs.get("maintenance_enabled", True))

    @property
    def running(self):
        return self._task is not None and not self._task.done()

    def _warn(self, message):
        try:
            self._logger.warning("dafeiyu: %s", message)
        except Exception:
            pass
        self._state.log(f"warn {message}")

    # -- daily training ----------------------------------------------------
    async def run_daily_learn(self, force=False):
        """Run the residual-reranker training; never raises."""
        probe = await self._rsrs.probe_learn()
        if not probe.get("learn_available"):
            self._state.log("daily learn skipped: learn subcommand unavailable")
            return {"ran": False, "reason": "learn-unavailable"}
        release = self._state.acquire_job_lock()
        if release is None:
            self._state.log("daily learn skipped: lock held")
            return {"ran": False, "reason": "locked"}
        started_at = self._now()
        try:
            last = self._state.read_jobs().get("daily") or {}
            finished = last.get("finished_at") if isinstance(last, dict) else None
            if not force and isinstance(finished, (int, float)) and started_at - finished < DAILY_INTERVAL_MS:
                return {"ran": False, "reason": "not-due"}
            args = learn_train_args(limit=int(self._jobs.get("daily_limit", 5000)), epochs=1)
            result = await self._rsrs.run(args, timeout_ms=TRAIN_TIMEOUT_MS)
            record = {
                "started_at": started_at,
                "finished_at": self._now(),
                "status": "ok",
                "counts": summarize_learn(result.get("envelope")),
            }
            self._state.write_jobs({"daily": record})
            self._state.log(f"daily learn ok {record['counts']}")
            return {"ran": True, **record}
        except Exception as exc:
            record = {
                "started_at": started_at,
                "finished_at": self._now(),
                "status": "error",
                "error": truncate(f"{type(exc).__name__}: {exc}", 500),
            }
            self._state.write_jobs({"daily": record})
            self._warn(f"daily learn failed: {record['error']}")
            return {"ran": False, **record}
        finally:
            release()

    # -- weekly encoder ----------------------------------------------------
    async def run_weekly_encoder(self):
        """Run the respire Granite encoder script; never raises."""
        if not self._jobs.get("weekly_granite", True):
            return {"ran": False, "reason": "disabled"}
        status = encoder_status(
            granite_model_dir=self._jobs.get("granite_model_dir", ""),
            respire_repo=self._jobs.get("respire_repo", ""),
        )
        if not status["ready"]:
            self._state.log(f"weekly encoder skipped: {status.get('reason')}")
            return {"ran": False, "reason": status.get("reason") or "not-ready"}
        release = self._state.acquire_job_lock()
        if release is None:
            self._state.log("weekly encoder skipped: lock held")
            return {"ran": False, "reason": "locked"}
        started_at = self._now()
        log_path = self._state.files["encoder_log"]
        append_log(log_path, f"--- {dt.datetime.fromtimestamp(started_at / 1000).isoformat()} start {status['script']['script']}")
        env = dict(self._rsrs.env)
        env.update(
            {
                "RSRS_BIN": self._rsrs.bin or "",
                "RSRS_ENCODER_BASE": self._jobs.get("granite_model_dir", ""),
                "RSRS_ENCODER_STATE": self._jobs.get("anchor_dir", ""),
            }
        )
        if self._config.get("data_dir"):
            env["RSRS_DATA_DIR"] = self._config["data_dir"]
        else:
            env.pop("RSRS_DATA_DIR", None)
        try:
            result = await run_process(
                "bash",
                [status["script"]["script"]],
                env=env,
                timeout_ms=ENCODER_TIMEOUT_MS,
            )
            append_log(log_path, truncate(result["stdout"], 20000))
            if result["stderr"]:
                append_log(log_path, truncate(result["stderr"], 20000))
            record = {
                "started_at": started_at,
                "finished_at": self._now(),
                "exit_code": result["code"],
                "status": "ok" if result["code"] == 0 else "error",
            }
            self._state.write_jobs({"weekly": record})
            append_log(log_path, f"--- {dt.datetime.fromtimestamp(record['finished_at'] / 1000).isoformat()} exit {result['code']}")
            return {"ran": True, **record}
        except Exception as exc:
            record = {
                "started_at": started_at,
                "finished_at": self._now(),
                "status": "error",
                "error": truncate(f"{type(exc).__name__}: {exc}", 500),
            }
            self._state.write_jobs({"weekly": record})
            append_log(log_path, f"--- error {record['error']}")
            self._warn(f"weekly encoder failed: {record['error']}")
            return {"ran": False, **record}
        finally:
            release()

    # -- nightly read-only maintenance -------------------------------------
    async def _run_audit(self):
        try:
            result = await self._rsrs.run(audit_args(), timeout_ms=AUDIT_TIMEOUT_MS, allow_failure=True)
            reason = readonly_error(result)
            if reason:
                return {"status": "error", "counts": {}, "error": truncate(reason, 300)}
            counts = summarize_audit(result.get("envelope"))
            status = counts.get("status")
            if not isinstance(status, str) or not status:
                status = "ok"
            return {"status": status, "counts": counts}
        except Exception as exc:
            detail = truncate(f"{type(exc).__name__}: {exc}", 300)
            self._warn(f"maintenance audit failed: {detail}")
            return {"status": "error", "counts": {}, "error": detail}

    async def _run_review(self):
        limit = self._jobs.get("review_limit", 20)
        try:
            result = await self._rsrs.run(
                review_due_args(limit), timeout_ms=REVIEW_TIMEOUT_MS, allow_failure=True
            )
            reason = readonly_error(result)
            if reason:
                return {"status": "error", "counts": {}, "error": truncate(reason, 300)}
            counts = summarize_review(result.get("envelope"))
            status = counts.get("status")
            if not isinstance(status, str) or not status:
                status = "ok"
            return {"status": status, "counts": counts}
        except Exception as exc:
            detail = truncate(f"{type(exc).__name__}: {exc}", 300)
            self._warn(f"maintenance review failed: {detail}")
            return {"status": "error", "counts": {}, "error": detail}

    async def _run_spotcheck(self):
        """Plan rule-layer spot-check over two read-only ``list`` scans.

        The metadata projection feeds the P0-1 checks (reflection treated as
        evidence, expired / past-review records, facts standing on a
        superseded cause); the body projection feeds the P2-2 missing-source
        count. Only integer counts are persisted — never ids, titles or bodies.
        """
        limit = self._jobs.get("spotcheck_limit", 10000)
        window = self._jobs.get("spotcheck_window_days", 14)
        now = dt.datetime.fromtimestamp(self._now() / 1000.0, tz=dt.timezone.utc)
        try:
            meta = await self._rsrs.run(
                spotcheck_list_args(limit, metadata=True),
                timeout_ms=SPOTCHECK_TIMEOUT_MS,
                allow_failure=True,
            )
            reason = readonly_error(meta)
            if reason:
                return {"status": "error", "counts": {}, "error": truncate(reason, 300)}
            counts = spotcheck_metadata(_details(meta), now=now)
            full = await self._rsrs.run(
                spotcheck_list_args(limit),
                timeout_ms=SPOTCHECK_TIMEOUT_MS,
                allow_failure=True,
            )
            reason = readonly_error(full)
            if reason:
                return {"status": "error", "counts": counts, "error": truncate(reason, 300)}
            counts.update(
                spotcheck_sources(_details(full), now=now, window_days=window)
            )
            scalars = {key: value for key, value in counts.items() if _scalar(value)}
            return {"status": "ok", "counts": scalars}
        except Exception as exc:
            detail = truncate(f"{type(exc).__name__}: {exc}", 300)
            self._warn(f"maintenance spotcheck failed: {detail}")
            return {"status": "error", "counts": {}, "error": detail}

    async def run_maintenance(self, force=False):
        """Read-only ``audit`` + ``review --due`` + spot-check. Never mutates memory, never raises."""
        release = self._state.acquire_job_lock()
        if release is None:
            self._state.log("maintenance skipped: lock held")
            return {"ran": False, "reason": "locked"}
        started_at = self._now()
        try:
            last = self._state.read_jobs().get("maintenance") or {}
            finished = last.get("finished_at") if isinstance(last, dict) else None
            if not force and isinstance(finished, (int, float)) and started_at - finished < MAINTENANCE_INTERVAL_MS:
                return {"ran": False, "reason": "not-due"}
            # Independent guards: a broken audit must not skip the review.
            audit = await self._run_audit()
            review = await self._run_review()
            spotcheck = await self._run_spotcheck()
            failed = [
                entry.get("status") == "error" for entry in (audit, review, spotcheck)
            ]
            if all(failed):
                status = "error"
            elif any(failed):
                status = "partial"
            else:
                status = "ok"
            record = {
                "started_at": started_at,
                "finished_at": self._now(),
                "status": status,
                "audit": audit,
                "review": review,
                "spotcheck": spotcheck,
            }
            try:
                self._state.write_jobs({"maintenance": record})
            except Exception as exc:
                self._warn(f"maintenance record not saved: {type(exc).__name__}: {exc}")
            self._state.log(
                "maintenance {} audit={} review={} spot={}".format(
                    status,
                    audit.get("status"),
                    review.get("status"),
                    spotcheck.get("status"),
                )
            )
            return {"ran": True, **record}
        finally:
            release()

    # -- scheduler ---------------------------------------------------------
    def start(self):
        """Start the background loop when any schedule is on."""
        if not (self.enabled or self.maintenance_enabled) or self.running:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())
        self._state.log("jobs started")
        return True

    async def stop(self):
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def _maintenance_deadline(self, stamp):
        """Next maintenance slot, or ``None`` when the switch is off."""
        if not self.maintenance_enabled:
            return None
        last = self._state.read_jobs().get("maintenance") or {}
        finished = last.get("finished_at") if isinstance(last, dict) else None
        overdue = not isinstance(finished, (int, float)) or stamp - finished >= MAINTENANCE_INTERVAL_MS
        if overdue:
            return stamp + CATCHUP_MS
        return stamp + ms_until_next(
            self._jobs.get("maintenance_hour", 4), self._jobs.get("maintenance_minute", 0), stamp
        )

    def _initial_deadlines(self):
        stamp = self._now()
        last = self._state.read_jobs().get("daily") or {}
        finished = last.get("finished_at") if isinstance(last, dict) else None
        overdue = not isinstance(finished, (int, float)) or stamp - finished >= DAILY_INTERVAL_MS
        daily_at = stamp + CATCHUP_MS if overdue else stamp + ms_until_next(
            self._jobs.get("daily_hour", 3), self._jobs.get("daily_minute", 30), stamp
        )
        weekly_at = stamp + ms_until_next_weekly(stamp)
        return daily_at, weekly_at, self._maintenance_deadline(stamp)

    async def _loop(self):
        daily_at, weekly_at, maintenance_at = self._initial_deadlines()
        while True:
            try:
                deadlines = [daily_at, weekly_at]
                if maintenance_at is not None:
                    deadlines.append(maintenance_at)
                delay = max(1.0, (min(deadlines) - self._now()) / 1000.0)
                await asyncio.sleep(delay)
                stamp = self._now()
                if maintenance_at is not None and stamp >= maintenance_at:
                    try:
                        await self.run_maintenance()
                    except Exception as exc:  # defensive: maintenance must never kill the loop
                        self._warn(f"maintenance job crashed: {type(exc).__name__}: {exc}")
                    maintenance_at = self._maintenance_deadline(stamp)
                # Training and the encoder are opt-in: jobs.enabled gates them.
                if weekly_at is not None and stamp >= weekly_at:
                    if self.enabled:
                        try:
                            await self.run_weekly_encoder()
                        except Exception as exc:
                            self._warn(f"weekly job crashed: {type(exc).__name__}: {exc}")
                    weekly_at = stamp + ms_until_next_weekly(stamp)
                if stamp >= daily_at:
                    if self.enabled:
                        try:
                            await self.run_daily_learn()
                        except Exception as exc:
                            self._warn(f"daily job crashed: {type(exc).__name__}: {exc}")
                    daily_at = stamp + ms_until_next(
                        self._jobs.get("daily_hour", 3), self._jobs.get("daily_minute", 30), stamp
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._warn(f"job loop error: {type(exc).__name__}: {exc}")
                await asyncio.sleep(60)
