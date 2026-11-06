"""AstrBot adapter: durable memory for a small mobile-QQ account (NapCat/aiocqhttp).

The plugin is a thin host shell. All decisions live in the sibling modules so
they can be tested without AstrBot:

  dafeiyu_config   config normalization and path expansion
  dafeiyu_policy   query extraction, hashed scopes, framing, selection, parsing
  dafeiyu_rsrs     rsrs discovery, argv builders, async JSON execution
  dafeiyu_state    state dir (0700), atomic files (0600), mutes, pending, locks
  dafeiyu_selfnote versioned, bounded self-note block inside the bot persona
  dafeiyu_granite  Granite checkpoint and weekly encoder script detection
  dafeiyu_jobs     daily ``learn train`` and weekly encoder schedule

Memory itself is never stored here: ``rsrs`` (respire) is the only authority
store, shared with the DSH plugin.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star

try:  # optional on older AstrBot builds
    from astrbot.api import AstrBotConfig
except ImportError:  # pragma: no cover
    AstrBotConfig = None  # type: ignore[assignment]

try:  # AstrBot >= 4.24.0 ships the temp-part marker
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover - older builds
    TextPart = None  # type: ignore[assignment]

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except Exception:  # pragma: no cover - older builds
    get_astrbot_data_path = None  # type: ignore[assignment]

try:  # package import when loaded as a plugin package
    from .dafeiyu_config import resolve_config
    from .dafeiyu_granite import encoder_status, read_index_meta
    from .dafeiyu_jobs import Jobs, summarize_audit, summarize_learn, summarize_review
    from .dafeiyu_policy import (
        QUERY_MAX_CHARS,
        ProactiveGate,
        cited_ids,
        clip,
        diary_decision,
        entry_status,
        evaluate_query,
        excerpt_text,
        format_recall_message,
        format_recall_reply,
        group_ids_from_projects,
        group_recall_scopes,
        hash_query,
        hashed_id,
        is_correction,
        item_id,
        item_score,
        item_title,
        legacy_group_scope,
        make_summary,
        merge_recall_envelopes,
        parse_command,
        parse_ids,
        plan_recall,
        select_items,
        sender_tag,
        short_id,
        speaker_header,
        topic_key,
        turn_from_event,
        with_past_names,
    )
    from .dafeiyu_rsrs import (
        AUDIT_TIMEOUT_MS,
        ORDINARY_TIMEOUT_MS,
        RECALL_TIMEOUT_MS,
        REVIEW_TIMEOUT_MS,
        TRAIN_TIMEOUT_MS,
        ENCODER_TIMEOUT_MS,
        EnvironmentProbe,
        Rsrs,
        RsrsError,
        audit_args,
        learn_lexical_args,
        learn_reset_args,
        learn_status_args,
        learn_train_args,
        query_log_mark_args,
        recall_args,
        readonly_error,
        remember_args,
        review_due_args,
        resolve_query_log_id,
        run_process,
        show_args,
        status_args,
        forget_args,
        chain_args,
        attach_args,
        update_args,
        truncate,
    )
    from .dafeiyu_state import (
        PERSON_RELATIONS,
        StateStore,
        append_log,
        now_ms,
        person_hash,
        person_is_admin,
    )
    from .dafeiyu_selfnote import MAX_BRIEF, MAX_FULL, SelfNoteError, SelfNoteService
except ImportError:  # direct module import (tests, scripts)
    from dafeiyu_config import resolve_config
    from dafeiyu_granite import encoder_status, read_index_meta
    from dafeiyu_jobs import Jobs, summarize_audit, summarize_learn, summarize_review
    from dafeiyu_policy import (
        QUERY_MAX_CHARS,
        ProactiveGate,
        cited_ids,
        clip,
        diary_decision,
        entry_status,
        evaluate_query,
        excerpt_text,
        format_recall_message,
        format_recall_reply,
        group_ids_from_projects,
        group_recall_scopes,
        hash_query,
        hashed_id,
        is_correction,
        item_id,
        item_score,
        item_title,
        legacy_group_scope,
        make_summary,
        merge_recall_envelopes,
        parse_command,
        parse_ids,
        plan_recall,
        select_items,
        sender_tag,
        short_id,
        speaker_header,
        topic_key,
        turn_from_event,
        with_past_names,
    )
    from dafeiyu_rsrs import (
        AUDIT_TIMEOUT_MS,
        ORDINARY_TIMEOUT_MS,
        RECALL_TIMEOUT_MS,
        REVIEW_TIMEOUT_MS,
        TRAIN_TIMEOUT_MS,
        ENCODER_TIMEOUT_MS,
        EnvironmentProbe,
        Rsrs,
        RsrsError,
        audit_args,
        learn_lexical_args,
        learn_reset_args,
        learn_status_args,
        learn_train_args,
        query_log_mark_args,
        recall_args,
        readonly_error,
        remember_args,
        review_due_args,
        resolve_query_log_id,
        run_process,
        show_args,
        status_args,
        forget_args,
        chain_args,
        attach_args,
        update_args,
        truncate,
    )
    from dafeiyu_state import (
        PERSON_RELATIONS,
        StateStore,
        append_log,
        now_ms,
        person_hash,
        person_is_admin,
    )
    from dafeiyu_selfnote import MAX_BRIEF, MAX_FULL, SelfNoteError, SelfNoteService

LOGGER = logging.getLogger("astrbot_plugin_dafeiyu")

#: Reply caps keep mobile QQ messages short.
RECALL_REPLY_LIMIT = 1200
SHOW_REPLY_LIMIT = 3000
STATUS_REPLY_LIMIT = 1500
#: ``learn train --from-labels`` is debounced to at most once per this window.
TRAIN_DEBOUNCE_MS = 10 * 60 * 1000
#: Default mute duration for ``/mem_mute``.
MUTE_HOURS = 720

#: Standing output-format rule, appended on EVERY LLM request (group and
#: private). The fish renders names with markdown asterisks, which QQ shows
#: literally, so the rule must be present even on turns where recall is skipped
#: for being too short, on cooldown, or because rsrs failed.
FORMAT_INSTRUCTION = (
    "<system-reminder>\n"
    "回复格式硬规则（QQ 不渲染 markdown）：\n"
    "1. 提到真人姓名、昵称、群名等称呼时一律用【】包裹，例如【百年江湖 随意沉浮】；\n"
    "2. 禁止星号：不要用 ** 或 * 包裹任何内容，QQ 会把星号原样显示出来；\n"
    "3. 不要用 markdown 的 #、>、` 等符号做排版。\n"
    "4. 昵称会变：#括号内『曾用』名与现名指同一人；昵称不同但属同一档案/曾用名关联者视为同一人，以注入档案为准，不以昵称为准。\n"
    "（本条是给模型的常驻格式指令，不是记忆内容。）\n"
    "</system-reminder>"
)

COMMANDS = (
    "mem", "mem_status", "mem_recall", "mem_show", "mem_remember", "mem_forget",
    "mem_good", "mem_bad", "mem_mute", "mem_unmute", "mem_reflect", "mem_learn",
    "mem_flywheel", "mem_clear", "mem_audit", "mem_review", "mem_selfnote",
    "mem_person",
)

_ID_IN_TEXT_RE = re.compile(r"\[([0-9a-fA-F]{8,64})\]")


_llm_tool_decorator = getattr(filter, "llm_tool", None)


def _llm_tool(name):
    """Register a function-calling tool when the host exposes it."""
    if _llm_tool_decorator is None:
        return lambda func: func
    return _llm_tool_decorator(name=name)


def _admin_only(func):
    """Stack AstrBot's ADMIN permission when the host exposes it."""
    permission_type = getattr(filter, "PermissionType", None)
    if permission_type is None or not hasattr(filter, "permission_type"):
        return func
    try:
        return filter.permission_type(permission_type.ADMIN)(func)
    except Exception:  # pragma: no cover - defensive
        return func


def _rfc3339(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _person_stamp(value):
    """Render a persons.json millisecond stamp as a date, or ``-``."""
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError, OverflowError):
        return "-"


#: Trailing ``:group:<gid16>:<uid16>`` of a per-member group bucket.
_MEMBER_SCOPE_SUFFIX = re.compile(r":group:[0-9a-f]{16}:([0-9a-f]{16})$")


def scope_person_key(scope, turn):
    """Person hash a recalled record belongs to, or ``''`` when unknown.

    A per-member group bucket carries the hash as its last segment. The legacy
    member-less bucket carries none, so its author is genuinely unknown and must
    not be guessed. A private bucket has no member segment either, but there the
    current sender *is* the author.
    """
    match = _MEMBER_SCOPE_SUFFIX.search(str(scope or ""))
    if match:
        return match.group(1)
    if not getattr(turn, "is_group", False):
        return person_hash(turn.user_id)
    return ""


def speaker_label(person_key, resolve_person):
    """``[昵称/relation]`` for one record, degraded when there is no 档案.

    ``resolve_person`` returns the person record or ``None``; a missing record
    must degrade to ``[未知/stranger]`` rather than invent an attribution.
    """
    person = resolve_person(person_key) if person_key else None
    if not isinstance(person, dict):
        return "[未知/stranger]"
    name = str(person.get("current_nickname") or "").strip() or "未知"
    label = with_past_names(name, person.get("nicknames"))
    relation = str(person.get("relation") or "stranger")
    return f"[{label}/{relation}]"


def _recall_id_map(envelope, items):
    """Short display ids plus the full UUIDs from the aligned ``details`` rows.

    ``query-log mark`` rejects short ids when an explicit ``--query-log-id`` is
    supplied, so feedback must resolve through this map.
    """
    all_items = envelope.get("items") if isinstance(envelope.get("items"), list) else []
    details = envelope.get("details") if isinstance(envelope.get("details"), list) else []
    full_by_short = {}
    for index, item in enumerate(all_items):
        short = item_id(item)
        if not short:
            continue
        detail = details[index] if index < len(details) else None
        full = detail.get("id") if isinstance(detail, dict) and isinstance(detail.get("id"), str) else short
        full_by_short[short[:8]] = full
    ids, full_ids, id_map = [], [], {}
    for item in items:
        short = item_id(item)
        if not short:
            continue
        full = full_by_short.get(short[:8], short)
        ids.append(short)
        full_ids.append(full)
        id_map[short[:8]] = full
    return ids, full_ids, id_map


def _format_ms(stamp):
    """Wall-clock ms to a short local timestamp; empty when unusable."""
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or stamp <= 0:
        return ""
    try:
        return datetime.fromtimestamp(int(stamp) / 1000.0).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return ""


def _review_lines(envelope, limit=10):
    """``[id] title`` rows from a review envelope; titles are only printed, never stored."""
    if not isinstance(envelope, dict):
        return []
    items = envelope.get("items") if isinstance(envelope.get("items"), list) else []
    details = envelope.get("details") if isinstance(envelope.get("details"), list) else []
    rows = []
    for index, row in enumerate(details):
        if len(rows) >= limit:
            break
        if not isinstance(row, dict):
            continue
        rid = row.get("id") if isinstance(row.get("id"), str) else ""
        title = row.get("title") if isinstance(row.get("title"), str) else ""
        if not rid or not title:
            fallback = items[index] if index < len(items) else None
            if isinstance(fallback, dict):
                if not rid and isinstance(fallback.get("name"), str):
                    rid = fallback["name"]
                if not title and isinstance(fallback.get("value"), str):
                    title = fallback["value"]
        if rid or title:
            head = f"[{short_id(rid)}] " if rid else ""
            rows.append(clip((head + title).strip(), 100))
    if rows:
        return rows
    for item in items:
        if len(rows) >= limit:
            break
        if not isinstance(item, dict):
            continue
        rid = item.get("name") if isinstance(item.get("name"), str) else ""
        title = item.get("value") if isinstance(item.get("value"), str) else ""
        if rid or title:
            head = f"[{short_id(rid)}] " if rid else ""
            rows.append(clip((head + title).strip(), 100))
    return rows


class DafeiyuPlugin(Star):
    """Mobile-QQ memory adapter backed by the shared rsrs store."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        data_root = None
        if get_astrbot_data_path is not None:
            try:
                data_root = get_astrbot_data_path()
            except Exception:
                data_root = None
        self.config = resolve_config(config, data_root=data_root)
        self.state = StateStore(self.config["state_dir"], debug=self.config["debug_log"])
        self.rsrs = Rsrs(self.config)
        self.probe = EnvironmentProbe(self.rsrs)
        self.gate = ProactiveGate(self.config, self.state, now=now_ms)
        self.jobs = Jobs(self.config, self.rsrs, self.state, logger=logger)
        self.selfnote = SelfNoteService(
            self.config, self.rsrs, self.state, context, logger=logger, now=now_ms
        )
        self._admins = set(self.config["admin_ids"])
        self._inflight = set()
        #: group_id -> {person_hash: last nickname}, for the 本群成员 line.
        self._group_speakers = {}
        #: short recall id -> bucket it came from, so memory_show can attribute.
        self._id_scope = {}
        self._tasks = set()
        self.state.log(f"init state_dir={self.config['state_dir']} scope_mode={self.config['scope_mode']}")

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    def _turn(self, event):
        return turn_from_event(event, self.config)

    def _rest(self, event, expected):
        """Text after ``/expected``; ``None`` when this handler does not own it."""
        text = getattr(event, "message_str", "") or ""
        name, rest = parse_command(text, known=COMMANDS)
        if name != expected:
            return None
        return rest

    def _relation_of(self, user_id):
        """Person relation for one sender, or ``None`` when there is no 档案.

        An unreadable or missing record yields ``None``, which every gate treats
        as ``stranger`` — a brand-new sender never gains rights by being unknown.
        """
        try:
            person = self.state.read_person(person_hash(user_id))
        except Exception:  # fail closed: an unreadable 档案 grants nothing
            return None
        if not isinstance(person, dict):
            return None
        relation = str(person.get("relation") or "").strip()
        return relation or None

    def _is_admin(self, user_id):
        """``admin_ids`` OR a relation of ``master`` (master == admin)."""
        if str(user_id) in self._admins:
            return True
        return self._relation_of(user_id) == "master"

    def _admin_allowed(self, user_id):
        """Fail closed for destructive commands.

        ``admin_ids`` is the plugin's own authorization list. When it is empty
        the command is refused even if an older host does not expose the ADMIN
        permission decorator; configure at least one admin id.

        A ``master`` is equivalent to an admin. That cannot widen the fail-closed
        default: with an empty ``admin_ids`` no admin exists, and
        ``set_person_relation`` is itself admin-only, so no ``master`` can ever
        be created to begin with.
        """
        return self._is_admin(user_id)

    def _may_reveal(self, turn):
        """Read gate.

        Private chats always pass. In a group an admin (hence a ``master``)
        passes, and so do ``friend`` and ``other_fish`` — that is the read
        relaxation the relation layer adds. ``stranger`` keeps the old behaviour
        and still needs an admin.
        """
        if not turn.is_group:
            return True
        if self._is_admin(turn.user_id):
            return True
        return self._relation_of(turn.user_id) in ("friend", "other_fish")

    def _may_write(self, turn):
        """Create gate: open to EVERY sender, private or group.

        用户裁决 2026-11-05 放开群写入 — a group member (stranger included) may
        add a memory. Opening the door does not let anyone write into somebody
        else's bucket: the write still lands in the SENDER'S OWN member bucket
        and carries their 【发言人】 attribution, so the global view keeps the
        authorship intact.

        This gate covers CREATING only. Editing an existing memory is gated by
        :meth:`_may_edit`, because a global-view recall hands every member ids
        that belong to other people.
        """
        return True

    def _may_edit(self, turn, target_id):
        """Whether this sender may EDIT an existing memory.

        An admin (or a ``master``) may edit anything.

        A private chat has exactly ONE bucket — the sender's own — so the recall
        ring (``_recent_id``) already proves the id came from this conversation.
        Requiring the in-memory origin map here would only break editing after a
        restart, and would answer with a group-flavoured message inside a private
        chat: the same class of misleading text this change set set out to remove.

        A GROUP is the opposite case: the global view recalls other members'
        buckets, so an id the origin map cannot place is refused rather than
        guessed — "unknown" must never mean "allowed".
        """
        if not turn.is_group:
            return True
        if self._is_admin(turn.user_id):
            return True
        origin = self._id_scope.get(short_id(target_id), "")
        if not origin:
            return False
        return origin == turn.scope

    def _read_denied(self, turn):
        """Rejection text that names the SENDER's relation, not the group config.

        The previous wording blamed the group's settings, which made the LLM
        restate a permission problem as "this group has write permission off" —
        wrong, and impossible for the user to act on.
        """
        relation = self._relation_of(turn.user_id) or "stranger"
        return f"group read denied: sender relation is {relation} (需要 friend/master/管理员)"

    def _edit_denied(self, turn, reason):
        return f"group edit denied: {reason}"

    @staticmethod
    def _write_denied(turn):
        """Create denial, kept only as a policy seam (the gate now always passes).

        Deliberately does not blame the group's configuration — that is what made
        the LLM restate a permission problem as "this group has write off".
        """
        return "write denied: sender not permitted"

    @staticmethod
    def _audit_denied():
        return "group audit denied: admin only (master/管理员)"

    def _may_admin(self, turn):
        """Read-only admin commands: the private owner, or an admin in a group.

        Unlike ``_admin_allowed`` this does not fail closed for private chats,
        because ``mem_audit`` / ``mem_review`` / ``memory_audit`` only read
        aggregate counters and never mutate a memory.

        Deliberately NOT relaxed for ``friend`` / ``other_fish``: audit stays
        admin-equivalent, only content reads were widened.
        """
        return (not turn.is_group) or self._is_admin(turn.user_id)

    def _spawn(self, coro):
        """Schedule background work without ever blocking the reply path."""
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            return None
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task):
        self._tasks.discard(task)
        try:
            error = task.exception()
        except (asyncio.CancelledError, Exception):
            return
        if error is not None:
            logger.warning("dafeiyu: background task failed: %s: %s", type(error).__name__, truncate(str(error), 300))

    def _append_part(self, req, text):
        """Attach per-turn context as a temp user part; fall back loudly."""
        parts = getattr(req, "extra_user_content_parts", None)
        if parts is not None and TextPart is not None:
            try:
                part = TextPart(text=text)
                marker = getattr(part, "mark_as_temp", None)
                if callable(marker):
                    try:
                        part = marker()
                    except Exception:
                        pass
                parts.append(part)
                return True
            except Exception as exc:
                logger.warning("dafeiyu: TextPart failed (%s); falling back to system_prompt", exc)
        prompt = getattr(req, "system_prompt", None)
        if prompt is None:
            logger.warning("dafeiyu: no extra_user_content_parts and no system_prompt; recall dropped")
            return False
        logger.warning("dafeiyu: appending recall to system_prompt; this invalidates prompt caching")
        req.system_prompt = f"{prompt}\n\n{text}" if prompt else text
        return True

    # ------------------------------------------------------------------ #
    # Recall injection
    # ------------------------------------------------------------------ #
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        try:
            self._record_speaker(event)
        except Exception as exc:  # fail open: profiling must never block a turn
            logger.warning("dafeiyu: upsert_person failed: %s: %s", type(exc).__name__, truncate(str(exc), 300))
        try:
            await self._prepare_recall(event, req)
        except Exception as exc:  # fail open: memory must never block a turn
            logger.warning("dafeiyu: recall failed open: %s: %s", type(exc).__name__, truncate(str(exc), 300))
        # Standing output rule. Deliberately OUTSIDE _prepare_recall: a turn whose
        # recall is skipped (too short, on cooldown, rsrs down) still has to come
        # back with names in 【】. Appended after recall so the recall block keeps
        # index 0 and a skipped recall is still detectable by its absence.
        try:
            self._append_part(req, FORMAT_INSTRUCTION)
        except Exception as exc:  # fail open: formatting must never block a turn
            logger.warning(
                "dafeiyu: format instruction failed open: %s: %s",
                type(exc).__name__,
                truncate(str(exc), 300),
            )

    def _record_speaker(self, event):
        """Refresh this sender's person 档案 and remember them for the group.

        Runs on every message, before any recall work, so a first-time speaker
        already has a record by the time an injection names them. The group hash
        is recorded too — that is what lets a later group read find every member
        and merge all of their buckets.
        """
        turn = self._turn(event)
        key = person_hash(turn.user_id)
        self.state.upsert_person(
            key,
            turn.sender_name,
            admin=person_is_admin(turn.user_id, self.config.get("admin_ids") or []),
            group_hash=hashed_id(turn.group_id) if turn.is_group else None,
        )
        if turn.is_group:
            # Fallback source for the "本群成员" line; the person档案 is primary.
            seen = self._group_speakers.setdefault(turn.group_id, {})
            seen.setdefault(key, turn.sender_name or "")

    def _group_member_hashes(self, turn):
        """Person hashes of this group's members, most recently active first.

        Built from the ``groups`` field written by :meth:`_record_speaker`, so
        only people who actually spoke here are included. The current sender is
        always kept (their bucket must never be dropped), and the list is capped
        at 8 so a busy group cannot fan out into an unbounded number of recalls.
        """
        if not getattr(turn, "is_group", False):
            return []
        group = hashed_id(turn.group_id)
        current = person_hash(turn.user_id)
        seen = []
        for key, entry in self.state.list_persons().items():
            record = entry if isinstance(entry, dict) else {}
            groups = record.get("groups") or []
            if group not in [str(item) for item in groups]:
                continue
            last_seen = int(record.get("last_seen") or 0)
            seen.append((last_seen, key))
        # Newest first; the hash breaks ties so the order is deterministic.
        seen.sort(key=lambda pair: (-pair[0], pair[1]))

        members = [current]
        for _last_seen, key in seen:
            if key in members or len(members) >= 8:
                continue
            members.append(key)
        return members

    #: How long a store-wide group-scope scan stays fresh for a private read.
    _GROUP_SCAN_TTL_S = 600.0

    async def _private_group_ids(self, turn):
        """Group hashes a private read must fan out over (三裁决③).

        Union of two sources, because either alone can be stale: the sender's
        persons.json ``groups`` (fresh for a brand-new group, recorded on the
        first message) and a cached scan of the store's own metadata (covers
        groups whose person record was lost to a persons.json reset). A group
        turn, a failed scan or a broken profile contributes nothing — the read
        fails open onto fewer scopes, never an error.
        """
        if getattr(turn, "is_group", False):
            return []
        groups = []
        try:
            record = self.state.read_person(person_hash(turn.user_id)) or {}
            if isinstance(record, dict):
                groups.extend(
                    str(item) for item in (record.get("groups") or []) if str(item)
                )
        except Exception:  # noqa: BLE001 - fail open: a broken profile must not kill recall
            pass
        now = time.monotonic()
        cached = getattr(self, "_group_scan", None)
        if not cached or now - cached[0] >= self._GROUP_SCAN_TTL_S:
            scanned = None
            try:
                result = await self.rsrs.run(
                    ["list", "--limit", "50000", "--metadata", "--json"],
                    timeout_ms=self.config["recall_timeout_ms"],
                    allow_failure=True,
                )
                envelope = result.get("envelope")
                if isinstance(envelope, dict) and envelope.get("status") in ("ok", "skip"):
                    projects = [
                        item.get("project")
                        for item in (envelope.get("details") or [])
                        if isinstance(item, dict)
                    ]
                    scanned = group_ids_from_projects(projects)
            except Exception:  # noqa: BLE001 - fail open: skip this scan, retry on the next read
                scanned = None
            if scanned is not None:
                # Only a confirmed scan is cached; failures retry next read.
                self._group_scan = (now, scanned)
                cached = (now, scanned)
        for gid in cached[1] if cached else []:
            if gid not in groups:
                groups.append(gid)
        return groups

    async def _recall_scopes(self, turn):
        """Scope fan-out for one read: whole group, or the private sender's own buckets."""
        if getattr(turn, "is_group", False):
            return group_recall_scopes(turn, self._group_member_hashes(turn))
        return group_recall_scopes(turn, user_groups=await self._private_group_ids(turn))

    def _annotate_speakers(self, items, turn):
        """Prefix every recalled title with ``[昵称/relation]`` in place.

        The bucket a record came from travels on the item as ``_scope`` (set by
        ``_recall_fanout``), which is what makes a legacy-bucket hit degrade to
        ``[未知/stranger]`` instead of being credited to the current speaker.
        """
        if len(self._id_scope) > 1000:
            self._id_scope.clear()
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            title = item_title(item)
            if not title:
                continue
            key = scope_person_key(item.get("_scope"), turn)
            self._id_scope[short_id(item_id(item))] = str(item.get("_scope") or "")
            item["title"] = f"{speaker_label(key, self.state.read_person)} {title}"
        return items

    def _group_members_line(self, turn):
        """``本群成员：…`` for the injected block, or ``None`` to omit it.

        There is no group-membership table, so this lists the senders seen in
        this group during the session, each with its档案 relation.
        """
        if not getattr(turn, "is_group", False):
            return None
        seen = self._group_speakers.get(turn.group_id) or {}
        names = []
        for key, fallback in seen.items():
            person = self.state.read_person(key)
            name = str((person or {}).get("current_nickname") or fallback or "").strip()
            if not name:
                continue
            relation = str((person or {}).get("relation") or "stranger")
            label = with_past_names(name, (person or {}).get("nicknames"))
            names.append(f"{label}/{relation}")
        if not names:
            return None
        return f"本群成员：{'、'.join(names)}"

    def _recall_allowed(self, scope, query_hash):
        if scope in self._inflight:
            return False, "in-flight"
        stamp = now_ms()
        entry = self.state.get_cooldown(scope)
        last_inject = int(entry.get("last_inject_ms", 0) or 0)
        if last_inject and stamp - last_inject < self.config["cooldown_ms"]:
            return False, "cooldown"
        if entry.get("query_hash") == query_hash:
            last_query = int(entry.get("query_ms", 0) or 0)
            if last_query and stamp - last_query < self.config["same_query_cooldown_ms"]:
                return False, "same-query"
        return True, "ok"

    def _speaker_past(self, turn):
        """Nicknames of the speaker for attribution headers (persons.json).

        Read-only and fail-open: a missing or broken record yields no
        history, i.e. the byte-identical pre-task-15 header.
        """
        try:
            record = self.state.read_person(person_hash(turn.user_id)) or {}
        except Exception:  # noqa: BLE001 - attribution must never break a turn
            return None
        return record.get("nicknames") if isinstance(record, dict) else None

    async def _recall_fanout(self, query, scopes, limit, lexical):
        """Run one recall per scope and merge, deduplicating by short id.

        A single scope keeps the original one-shot path. For a group the legs
        split the total budget, so the merged result never exceeds ``limit``.
        """
        if len(scopes) == 1:
            args = recall_args(query, titles=True, limit=limit, project=scopes[0], lexical=lexical)
            try:
                result = await self.rsrs.run(args, timeout_ms=self.config["recall_timeout_ms"])
            except RsrsError as exc:
                # Same fail-open contract as the multi-leg path: one broken
                # recall must not abort the turn. Every caller already handles an
                # empty item list, so an empty envelope degrades cleanly.
                self.state.log(f"recall failed ({scopes[0]}): {exc.code}: {truncate(str(exc), 200)}")
                return {"items": [], "details": []}
            return self._tag_scopes(result["envelope"], scopes[0])

        per = max(1, limit // len(scopes))
        envelopes = []
        for scope in scopes:
            args = recall_args(query, titles=True, limit=per, project=scope, lexical=lexical)
            try:
                result = await self.rsrs.run(args, timeout_ms=self.config["recall_timeout_ms"])
            except RsrsError as exc:
                self.state.log(f"recall failed ({scope}): {exc.code}: {truncate(str(exc), 200)}")
                continue
            envelopes.append(self._tag_scopes(result["envelope"], scope))
        return merge_recall_envelopes(envelopes, limit)

    @staticmethod
    def _tag_scopes(envelope, scope):
        """Record which bucket each item came from, for speaker attribution.

        Kept out of the item payload proper: it is a read-side annotation that
        ``merge_recall_envelopes`` carries through untouched.
        """
        items = envelope.get("items") if isinstance(envelope, dict) else None
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item["_scope"] = scope
        return envelope

    async def _prepare_recall(self, event, req):
        if not self.config["enabled"] or not self.config["auto_recall"]:
            return
        turn = self._turn(event)
        raw = getattr(event, "message_str", "") or ""
        # Bounded auto-feedback: a correction labels the previous recall bad.
        if is_correction(raw):
            self._mark_correction(turn)
        ok, query, reason = evaluate_query(raw)
        if not ok:
            self.state.log(f"recall skip: {reason}")
            return
        query_hash = hash_query(query)
        allowed, why = self._recall_allowed(turn.scope, query_hash)
        if not allowed:
            self.state.log(f"recall skip: {why}")
            return
        self._inflight.add(turn.scope)
        try:
            environment = await self.probe.probe()
            plan = plan_recall(self.config["retrieval"], environment["learn_available"], environment["bge_available"])
            envelope = await self._recall_fanout(
                query,
                await self._recall_scopes(turn),
                self.config["recall_limit"],
                plan["use_lexical"],
            )
            items = select_items(
                envelope, plan["use_lexical"], self.config["min_score_hybrid"], self.config["recall_limit"]
            )
            ids, full_ids, id_map = _recall_id_map(envelope, items)
            # P1-2 representation layering: the top hit may carry a verbatim
            # head excerpt plus its status flags; everything else stays a
            # title. Fail-open to titles.
            snippets, statuses = await self._top_excerpt(items, id_map)
            reflections = await self._reflections(turn, query)
            self._annotate_speakers(items, turn)
            text = format_recall_message(
                items, reflections, self.config["recall_budget_chars"], snippets, statuses
            )
            if text is None:
                self.state.log("recall produced nothing within budget")
                return
            members = self._group_members_line(turn)
            if members:
                text = text.replace("</system-reminder>", f"{members}\n</system-reminder>")
            if not self._append_part(req, text):
                return
            request_id = await resolve_query_log_id(self.rsrs, envelope, ids)
            stamp = now_ms()
            self.state.record_pending(
                turn.scope,
                ids,
                full_ids=full_ids,
                id_map=id_map,
                request_id=request_id,
                query_hash=query_hash,
                ts=stamp,
            )
            self.state.set_cooldown(
                turn.scope, last_inject_ms=stamp, query_hash=query_hash, query_ms=stamp
            )
            self.state.log(f"recall injected scope={turn.scope} ids={','.join(short_id(i) for i in ids)}")
        finally:
            self._inflight.discard(turn.scope)

    async def _source_entries(self, entry):
        """P0-1 source-invalidation reads; returns source entry dicts.

        The show/chain ancestor projections only carry id+title, so a derived
        record's sources must be read individually. Bounded by design: only a
        hypothesis/reflection (the derived kinds) follows its direct parent and
        only actual ``source_refs`` entries are followed (max 4 total); a
        tombstoned source becomes ``{"id": ..., "deleted": True}`` and a
        transient read failure is skipped (fail-open, never claims deletion).
        """
        kind = str(entry.get("kind") or "")
        targets = []
        if kind in ("Hypothesis", "Reflection") and entry.get("parent_id"):
            targets.append(str(entry["parent_id"]))
        refs = entry.get("source_refs")
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict) and isinstance(ref.get("id"), str) and ref.get("id"):
                    targets.append(ref["id"])
        sources = []
        seen = set()
        for full in targets[:4]:
            if full in seen:
                continue
            seen.add(full)
            try:
                result = await self.rsrs.run(show_args(full), timeout_ms=ORDINARY_TIMEOUT_MS)
            except RsrsError as exc:
                if "not found" in str(exc).lower():
                    sources.append({"id": full, "deleted": True})
                else:
                    self.state.log(f"source check failed: {exc.code}")
                continue
            envelope = result.get("envelope") if isinstance(result.get("envelope"), dict) else {}
            details = envelope.get("details") if isinstance(envelope.get("details"), dict) else {}
            source = details.get("entry") if isinstance(details.get("entry"), dict) else {}
            if source:
                sources.append(source)
        return sources

    async def _annotated_status(self, entry):
        """Status row for one entry (P0-1 typing + source invalidation, P2-1
        expiry, P2-3 artifact); never raises and never blocks the reply."""
        try:
            sources = await self._source_entries(entry)
            return "；".join(entry_status(entry, sources=sources, path_exists=os.path.exists))
        except Exception as exc:
            self.state.log(f"status annotation failed: {type(exc).__name__}")
            return ""

    async def _top_excerpt(self, items, id_map):
        """Top-hit annotation: ``(snippets, statuses)`` as ``{short_id: text}``.

        The snippet is the verbatim head excerpt (P1-2); the status carries the
        P0-1/P2-1/P2-3 flags (unverified kind, expiry, superseded source, stale
        artifact) read from the same single ``show``. Any failure (disabled
        budget, no hit, missing id, show error) returns ``({}, {})`` so the
        turn still injects plain titles.
        """
        try:
            limit = int(self.config.get("recall_excerpt_chars", 0) or 0)
        except (TypeError, ValueError):
            limit = 0
        if limit <= 0 or not items:
            return {}, {}
        top = items[0]
        short = short_id(item_id(top))
        full = id_map.get(short) or item_id(top)
        if not short or not full:
            return {}, {}
        try:
            result = await self.rsrs.run(show_args(full), timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            self.state.log(f"excerpt show failed: {exc.code}")
            return {}, {}
        except Exception as exc:  # never block the turn on an excerpt
            self.state.log(f"excerpt show failed: {type(exc).__name__}")
            return {}, {}
        envelope = result.get("envelope") if isinstance(result.get("envelope"), dict) else {}
        details = envelope.get("details") if isinstance(envelope.get("details"), dict) else {}
        entry = details.get("entry") if isinstance(details.get("entry"), dict) else {}
        status = await self._annotated_status(entry)
        if status:
            self.state.log(f"status annotated id={short} status={status}")
        # The status rides the same single top-1 show as the excerpt: the
        # recall_excerpt_chars=0 knob still means "no extra show at all".
        body = entry.get("content") if isinstance(entry.get("content"), str) else ""
        excerpt = excerpt_text(body, limit)
        if not excerpt and not status:
            return {}, {}
        if excerpt:
            self.state.log(f"excerpt injected id={short} chars={len(excerpt)}")
        snippets = {short: excerpt} if excerpt else {}
        return snippets, ({short: status} if status else {})

    @_llm_tool("memory_show")
    async def memory_show_tool(self, event: AstrMessageEvent, memory_id: str) -> str:
        '''Read one recalled memory body by its 8-character id.

        Use this after a recalled-memory title looks relevant. Do not read
        data/memories or use shell/grep; this tool reads the shared rsrs store.
        A 状态 line (unverified / expired / superseded / stale artifact) is
        binding context: never quote an expired or superseded record as current.

        Args:
            memory_id(string): The 8-character id shown in the recalled-memory block.
        '''
        turn = self._turn(event)
        if not self._may_reveal(turn):
            return self._read_denied(turn)
        ids = parse_ids(memory_id)
        if not ids:
            return "usage: memory_show(memory_id=<8-char id>)"
        short = short_id(ids[0])
        recent = self.state.recent_pending(turn.scope)
        if not recent:
            return "no recent recall in this chat; call memory_recall first"
        full = self._recent_id(turn.scope, short)
        if not full:
            return "that id is not from this chat's recent recall"
        try:
            result = await self.rsrs.run(show_args(full), timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            return f"show failed: {clip(str(exc), 200)}"
        envelope = result.get("envelope") or {}
        details = envelope.get("details") if isinstance(envelope.get("details"), dict) else {}
        entry = details.get("entry") if isinstance(details.get("entry"), dict) else {}
        title = entry.get("title") if isinstance(entry.get("title"), str) else ""
        body = entry.get("content") if isinstance(entry.get("content"), str) else ""
        status = await self._annotated_status(entry)
        scope = self._id_scope.get(short, "")
        head = (
            f"{speaker_label(scope_person_key(scope, turn), self.state.read_person)} "
            f"[{short}] {title}"
            + (f"\n状态: {status}" if status else "")
        )
        if not body:
            return head if status else f"no body for {short}"
        return clip(f"{head}\n{body}", 3000)

    @_llm_tool("memory_recall")
    async def memory_recall_tool(self, event: AstrMessageEvent, query: str) -> str:
        '''Search the shared memory store with a query.

        Use this when the automatic recalled-memory block is not enough, then
        call memory_show for the record body that looks relevant.

        Args:
            query(string): Search keywords, names, dates or a short question.
        '''
        turn = self._turn(event)
        text = query.strip()
        if not text:
            return "query required"
        environment = await self.probe.probe()
        plan = plan_recall(
            self.config["retrieval"], environment["learn_available"], environment["bge_available"]
        )
        try:
            envelope = await self._recall_fanout(
                text, await self._recall_scopes(turn), 20, plan["use_lexical"]
            )
        except RsrsError as exc:
            return f"recall failed: {clip(str(exc), 200)}"
        items = select_items(
            envelope, plan["use_lexical"], self.config["min_score_hybrid"], 20
        )
        if not items:
            return "no matching memories in this chat"
        ids, full_ids, id_map = _recall_id_map(envelope, items)
        self.state.record_pending(
            turn.scope,
            ids,
            full_ids=full_ids,
            id_map=id_map,
            query_hash=hash_query(text),
        )
        self._annotate_speakers(items, turn)
        return format_recall_reply(items, 1500)

    @_llm_tool("memory_remember")
    async def memory_remember_tool(
        self,
        event: AstrMessageEvent,
        judgment: str,
        reasoning: str,
        memory_type: str,
        tags: str,
        artifact: str = "",
    ) -> str:
        '''Store one durable fact, preference, event or task in shared memory.

        Use this when the user states something worth keeping. The judgment must
        be a complete statement; reasoning records why it matters. Hypothesis
        and reflection records are unverified guesses: never store a decision or
        knowledge record whose only basis is one — cite what the user said, a
        file, a run or an observation instead.

        Args:
            judgment(string): The core fact as a complete statement.
            reasoning(string): Why it matters, evidence, or context; may be empty.
            memory_type(string): knowledge, skill, emotion, event, task, profile, decision, preference, hypothesis or reflection.
            tags(string): Comma-separated tags, for example "宠物,偏好".
            artifact(string): Path of a file this fact produced (report, script, output); may be empty.
        '''
        turn = self._turn(event)
        # 用户裁决 2026-11-05 放开群写入: creating is open to every sender, so
        # this gate now passes for groups too. Kept as the policy seam — the
        # write still lands in this sender's own bucket with their attribution.
        if not self._may_write(turn):
            return self._write_denied(turn)
        fact = judgment.strip()
        if not fact:
            return "judgment required"
        # rsrs accepts only its strict kind names; "event" is stored as context
        # and "profile" as self (SelfNote), not as invented aliases.
        kind_map = {
            "knowledge": "knowledge",
            "skill": "skill",
            "emotion": "emotion",
            "event": "context",
            "task": "task",
            "profile": "self",
            "decision": "decision",
            "preference": "preference",
            "hypothesis": "hypothesis",
            "reflection": "reflection",
            "context": "context",
            "self": "self",
        }
        kind_name = memory_type.strip().lower()
        kind = kind_map.get(kind_name, "context")
        body = f"{speaker_header(turn, self._speaker_past(turn))}\n【判断】{fact}\n【理由】{reasoning.strip() or '（未提供）'}"
        tag_parts = ["qq", kind_name, sender_tag(turn)]
        tag_parts.extend(part.strip() for part in tags.split(",") if part.strip())
        # metadata v1 gate: hypothesis/reflection/self and the artifact facet
        # are contract-1 writes. A host with the gate off degrades to tags and
        # a body line instead of failing the store (legacy contract).
        meta = await self.rsrs.metadata_status()
        typed_ok = bool(meta.get("enabled"))
        notes = []
        send_type = kind
        if not typed_ok and kind in ("hypothesis", "reflection", "self"):
            send_type = None
            notes.append("v1 off: stored untyped, kind kept in tags")
        artifact_path = (artifact or "").strip()
        artifact_json = None
        if artifact_path:
            # The body line keeps the path lexically searchable (titles-only
            # recall never returns metadata); the facet adds the machine field.
            body += f"\n【产物】{artifact_path}"
            tag_parts.append("artifact")
            if typed_ok:
                artifact_json = json.dumps(
                    {"path": artifact_path, "purpose": clip(fact, 100)}, separators=(",", ":")
                )
            else:
                notes.append("v1 off: artifact kept in body only")
        try:
            result = await self.rsrs.run(
                remember_args(
                    body,
                    title=clip(fact, 60),
                    # T1-3: write-time summary. --summary forces contract 1,
                    # so it rides the same typed gate as the artifact facet.
                    summary=make_summary(clip(fact, 60), body) if typed_ok else None,
                    type=send_type,
                    importance="important",
                    tags=",".join(tag_parts),
                    project=turn.scope,
                    artifact=artifact_json,
                    force=True,
                ),
                timeout_ms=ORDINARY_TIMEOUT_MS,
            )
        except RsrsError as exc:
            return f"remember failed: {clip(str(exc), 200)}"
        envelope = result.get("envelope") or {}
        summary = envelope.get("summary") if isinstance(envelope.get("summary"), dict) else {}
        stored_id = summary.get("id") if isinstance(summary.get("id"), str) else ""
        title = summary.get("title") if isinstance(summary.get("title"), str) else clip(fact, 60)
        if stored_id:
            reply = f"stored [{short_id(stored_id)}] {title}"
            if notes:
                reply += " (" + "; ".join(notes) + ")"
            return reply
        return f"not stored: {clip(str(summary.get('reason') or envelope.get('status') or 'unknown'), 200)}"

    def _recent_id(self, scope, raw):
        """Resolve a short id against every live pending recall in this scope."""
        ids = parse_ids(raw)
        if not ids:
            return ""
        short = short_id(ids[0])
        for pending in self.state.recent_pending(scope):
            mapping = pending.get("id_map") if isinstance(pending.get("id_map"), dict) else {}
            full = mapping.get(short)
            if isinstance(full, str) and full:
                return full
            for value in pending.get("full_ids") or []:
                if short_id(value) == short:
                    return str(value)
        return ""

    def _pending_id(self, scope, raw):
        """Compatibility name; the pending ring now spans several recalls."""
        return self._recent_id(scope, raw)

    @_llm_tool("memory_update")
    async def memory_update_tool(
        self,
        event: AstrMessageEvent,
        memory_id: str,
        judgment: str,
        reasoning: str,
        title: str,
        tags: str,
    ) -> str:
        '''Update one recalled memory body, title or tags.

        Use this when the user corrects or refines an existing record.

        Args:
            memory_id(string): The 8-character id from the recalled-memory block.
            judgment(string): The corrected core statement; may be empty to keep the body.
            reasoning(string): The supporting reason; may be empty.
            title(string): New title; empty keeps the current title.
            tags(string): New comma-separated tags; empty keeps the current tags.
        '''
        turn = self._turn(event)
        # Editing is NOT part of the 2026-11-05 create opening: a global-view
        # recall hands every member ids from other buckets, so an unguarded
        # update would let anyone rewrite a teammate's memory.
        if not self._may_edit(turn, memory_id):
            origin = self._id_scope.get(short_id(memory_id), "")
            reason = (
                "target memory belongs to another member (admin only)"
                if origin
                else "memory origin unknown (admin only)"
            )
            return self._edit_denied(turn, reason)
        memory_full = self._recent_id(turn.scope, memory_id)
        if not memory_full:
            return "id not from this chat's recent recall"
        fact = judgment.strip()
        reason = reasoning.strip()
        content = None
        if fact or reason:
            content = f"【判断】{fact or '（未提供）'}\n【理由】{reason or '（未提供）'}"
        new_title = title.strip() or (clip(fact, 60) if fact else None)
        # T1-3: a body change regenerates the summary on the same id — the CLI
        # assigns it (replace, never stack). Same typed gate as remember; a
        # content-free touch keeps the existing summary untouched.
        summary_text = None
        if content is not None and (await self.rsrs.metadata_status()).get("enabled"):
            summary_text = make_summary(new_title or "", content)
        try:
            result = await self.rsrs.run(
                update_args(
                    memory_full,
                    content=content,
                    title=new_title,
                    tags=tags.strip() or None,
                    summary=summary_text,
                ),
                timeout_ms=ORDINARY_TIMEOUT_MS,
            )
        except RsrsError as exc:
            return f"update failed: {clip(str(exc), 200)}"
        return f"updated [{short_id(memory_full)}] {result.get('envelope', {}).get('status', 'ok')}"

    @_llm_tool("memory_forget")
    async def memory_forget_tool(self, event: AstrMessageEvent, memory_id: str) -> str:
        '''Tombstone one recalled memory.

        Args:
            memory_id(string): The 8-character id from the recalled-memory block.
        '''
        turn = self._turn(event)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        memory_full = self._recent_id(turn.scope, memory_id)
        if not memory_full:
            return "id not from this chat's recent recall"
        try:
            await self.rsrs.run(forget_args(memory_full), timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            return f"forget failed: {clip(str(exc), 200)}"
        return f"forgot [{short_id(memory_full)}]"

    @_llm_tool("memory_chain")
    async def memory_chain_tool(self, event: AstrMessageEvent, memory_id: str, depth: int) -> str:
        '''Read a memory causal chain: ancestors, the record and descendants.

        Args:
            memory_id(string): The 8-character id from the recalled-memory block.
            depth(number): Descendant depth, 0-3; use 3 for the full chain.
        '''
        turn = self._turn(event)
        if not self._may_reveal(turn):
            return self._read_denied(turn)
        memory_full = self._recent_id(turn.scope, memory_id)
        if not memory_full:
            return "id not from this chat's recent recall"
        try:
            result = await self.rsrs.run(
                chain_args(memory_full, max(0, min(3, int(depth)))), timeout_ms=ORDINARY_TIMEOUT_MS
            )
        except RsrsError as exc:
            return f"chain failed: {clip(str(exc), 200)}"
        details = (result.get("envelope") or {}).get("details")
        return clip(json.dumps(details, ensure_ascii=False), 3000)

    async def _reflect_store(self, turn, text, parent, ttl_days, title):
        """Build the ``remember`` args for one unverified hypothesis (P0-1).

        A metadata-v1 host gets the typed write (``kind=hypothesis`` plus
        ``valid_until``/``review_at`` so recall/review can flag expiry). A
        legacy host with the gate off degrades to a tagged untyped record
        instead of failing the write, mirroring the selfNote fallback.
        Returns ``(args, typed)``.
        """
        now = datetime.now(timezone.utc)
        meta = await self.rsrs.metadata_status()
        kwargs = dict(
            title=title,
            importance="important",
            parent=parent or None,
            project=turn.scope,
        )
        if meta.get("enabled"):
            kwargs.update(
                type="hypothesis",
                valid_until=_rfc3339(now + timedelta(days=ttl_days)),
                review_at=_rfc3339(now + timedelta(days=max(1, ttl_days - 4))),
                # T1-3 on the typed path only: the untyped fallback must stay
                # contract-0 so it survives a gate-off host.
                summary=make_summary(title, text),
            )
            return remember_args(text, **kwargs), True
        kwargs.update(tags="qq,hypothesis,unverified")
        return remember_args(text, **kwargs), False

    @_llm_tool("memory_reflect")
    async def memory_reflect_tool(
        self,
        event: AstrMessageEvent,
        content: str,
        source_id: str,
        valid_days: int,
    ) -> str:
        '''Store an unverified hypothesis linked to a source memory.

        Args:
            content(string): The hypothesis, clearly worded as unverified.
            source_id(string): 8-character id of the source memory; may be empty.
            valid_days(number): Validity in days, 1-30; use 7 by default.
        '''
        turn = self._turn(event)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        text = content.strip()
        if not text:
            return "content required"
        ttl = max(1, min(30, int(valid_days)))
        parent = self._recent_id(turn.scope, source_id) if source_id.strip() else ""
        try:
            args, typed = await self._reflect_store(
                turn, text, parent, ttl, f"Hypothesis: {clip(text, 60)}"
            )
            result = await self.rsrs.run(args, timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            return f"reflect failed: {clip(str(exc), 200)}"
        summary = (result.get("envelope") or {}).get("summary") or {}
        stored_id = summary.get("id") if isinstance(summary.get("id"), str) else ""
        head = f"reflection stored [{short_id(stored_id)}]" if stored_id else "reflection submitted"
        if not typed:
            return clip(f"{head} - unverified hypothesis (metadata v1 off: stored untyped, no expiry)", 400)
        return clip(f"{head} - unverified hypothesis, expires in {ttl}d, review in {max(1, ttl - 4)}d", 400)

    @_llm_tool("memory_feedback")
    async def memory_feedback_tool(self, event: AstrMessageEvent, ids: str, verdict: str) -> str:
        '''Label recalled records good or bad so the local reranker learns.

        Args:
            ids(string): Comma-separated 8-character ids from the recalled block.
            verdict(string): good or bad.
        '''
        turn = self._turn(event)
        wanted = [part.strip() for part in ids.split(",") if part.strip()]
        if not wanted:
            return "ids required"
        label = "bad" if verdict.strip().lower() == "bad" else "good"
        pending = self.state.latest_pending(turn.scope)
        request_id = pending.get("request_id") if pending else None
        id_map = pending.get("id_map") if pending else None
        marked = await self._mark(turn.scope, wanted, label, request_id, "model", id_map)
        return f"{label} recorded" if marked else "feedback not recorded"

    @_llm_tool("memory_status")
    async def memory_status_tool(self, event: AstrMessageEvent) -> str:
        '''Read memory health: scheme, store size, learning state and Granite index.'''
        turn = self._turn(event)
        environment = await self.probe.probe()
        plan = plan_recall(
            self.config["retrieval"], environment["learn_available"], environment["bge_available"]
        )
        lines = [
            f"scheme={'lexical' if plan['use_lexical'] else 'hybrid'}",
            f"learn={'on' if environment['learn_available'] else 'off'}",
            f"bge={'yes' if environment['bge_available'] else 'no'}",
            f"scope={turn.scope}",
            f"pending={self.state.pending_count(turn.scope)}",
        ]
        try:
            result = await self.rsrs.run(status_args(), timeout_ms=RECALL_TIMEOUT_MS, allow_failure=True)
            summary = (result.get("envelope") or {}).get("summary") or {}
            if isinstance(summary.get("local_alive"), int):
                lines.append(f"store_alive={summary['local_alive']}")
        except Exception:
            lines.append("store=unavailable")
        if environment["learn_available"]:
            try:
                result = await self.rsrs.run(
                    learn_status_args(), timeout_ms=RECALL_TIMEOUT_MS, allow_failure=True
                )
                summary = (result.get("envelope") or {}).get("summary") or {}
                if isinstance(summary.get("updates"), int):
                    lines.append(f"learn_updates={summary['updates']}")
            except Exception:
                pass
        encoder = encoder_status(
            granite_model_dir=self.config["jobs"]["granite_model_dir"],
            respire_repo=self.config["jobs"]["respire_repo"],
        )
        lines.append(f"granite={'ready' if encoder.get('ready') else 'missing'}")
        return clip("\n".join(lines), 800)

    @_llm_tool("memory_audit")
    async def memory_audit_tool(self, event: AstrMessageEvent) -> str:
        '''Read aggregate memory health counters (read-only).

        Returns counts only: active records, orphans, bad importance, duplicate
        and truncated titles, chain depth, hot untyped, failing skills and skill
        template gaps. It never returns ids, titles or bodies and never changes a
        memory.
        '''
        turn = self._turn(event)
        # Audit is an admin-equivalent read, matching the /mem_audit command:
        # friend/other_fish were widened for CONTENT reads only.
        if not self._may_admin(turn):
            return self._audit_denied()
        return clip(await self._audit_view(800), 800)

    #: ``audit`` counters rendered on the first line, in this order.
    _AUDIT_FIRST = (
        ("total_active", "active"),
        ("orphans", "orphan"),
        ("invalid_importance", "bad importance"),
        ("same_title_groups", "dup titles"),
        ("truncated_titles", "trunc titles"),
        ("max_depth", "max depth"),
    )
    #: Counters rendered on the second line.
    _AUDIT_SECOND = (
        ("hot_untyped", "hot untyped"),
        ("failing_skills", "failing skills"),
        ("skill_template_gaps", "skill gaps"),
    )

    def _audit_text(self, envelope, limit=1200):
        """Compact count-only audit summary; details lists are never printed."""
        counts = summarize_audit(envelope)
        status = counts.get("status")
        if not isinstance(status, str) or not status:
            status = "ok"
        first = [
            f"{label} {counts[key]}"
            for key, label in self._AUDIT_FIRST
            if isinstance(counts.get(key), (int, float))
        ]
        second = [
            f"{label} {counts[key]}"
            for key, label in self._AUDIT_SECOND
            if isinstance(counts.get(key), (int, float))
        ]
        lines = [f"audit: {status}"]
        if first:
            lines.append(" | ".join(first))
        if second:
            lines.append(" | ".join(second))
        return clip("\n".join(lines), limit)

    async def _audit_view(self, limit=1200):
        """Run the read-only audit and render it, or a short failure line."""
        try:
            result = await self.rsrs.run(audit_args(), timeout_ms=AUDIT_TIMEOUT_MS, allow_failure=True)
        except RsrsError as exc:
            return f"audit failed: {clip(str(exc), 200)}"
        reason = readonly_error(result)
        if reason:
            return f"audit failed: {clip(reason, 200)}"
        return self._audit_text(result.get("envelope"), limit)

    @_llm_tool("memory_attach")
    async def memory_attach_tool(
        self, event: AstrMessageEvent, child_id: str, parent_id: str
    ) -> str:
        '''Attach one recalled memory under another as cause/effect.

        Args:
            child_id(string): 8-character id of the effect record.
            parent_id(string): 8-character id of the cause record.
        '''
        turn = self._turn(event)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        child = self._recent_id(turn.scope, child_id)
        parent = self._recent_id(turn.scope, parent_id)
        if not child or not parent:
            return "both ids must come from this chat's recent recall"
        try:
            await self.rsrs.run(attach_args(child, parent), timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            return f"attach failed: {clip(str(exc), 200)}"
        return f"attached {short_id(child)} under {short_id(parent)}"

    @_llm_tool("memory_selfnote")
    async def memory_selfnote_tool(
        self,
        event: AstrMessageEvent,
        content: str,
        brief: str,
        source_ids: str,
    ) -> str:
        '''Rewrite the bot's own persona self-note from recalled evidence.

        The base persona rules are never changed; only the managed self-note
        block is replaced, and the previous version is superseded in memory.

        Args:
            content(string): Full self-note text, at most 1200 characters. Style, tone, relationship and recent self-knowledge only; no private facts.
            brief(string): One-line summary stored in the persona block, at most 300 characters.
            source_ids(string): Comma-separated 8-character ids from this chat's recent recall.
        '''
        turn = self._turn(event)
        if turn.is_group:
            return "selfnote is managed in a private chat"
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        if not self.selfnote.config.get("enabled"):
            return "selfnote disabled"
        body = str(content or "").strip()
        summary = str(brief or "").strip()
        if not body:
            return "content required"
        if len(body) > self.selfnote.max_full:
            return f"content too long (max {self.selfnote.max_full})"
        if not summary:
            return "brief required"
        if len(summary) > self.selfnote.max_brief:
            return f"brief too long (max {self.selfnote.max_brief})"
        wanted = [
            part.strip()
            for part in str(source_ids or "").replace("，", ",").split(",")
            if part.strip()
        ]
        if not wanted:
            return "source_ids required"
        resolved = []
        for raw in wanted:
            full = self._recent_id(turn.scope, raw)
            if not full:
                return f"id {clip(raw, 12)} is not from this chat's recent recall"
            resolved.append(full)
        try:
            result = await self.selfnote.apply(
                turn,
                {"full": body, "brief": summary, "source_ids": resolved},
                manual=False,
            )
        except SelfNoteError as exc:
            return f"selfnote failed: {clip(str(exc), 200)}"
        except Exception as exc:  # fail open: never break the reply path
            logger.warning("dafeiyu: selfnote tool failed: %s", type(exc).__name__)
            return "selfnote failed"
        suffix = "" if result.get("state_saved", True) else " (state pointer pending)"
        return clip(f"selfnote updated [{short_id(result['id'])}]{suffix} {result['brief']}", 400)

    async def _reflections(self, turn, query):
        """Optional single unverified hypothesis; off by default."""
        if not self.gate.enabled:
            return []
        key = topic_key(query)
        allowed, why = self.gate.permit(turn.scope, key, None)
        if not allowed:
            self.state.log(f"reflection skip: {why}")
            return []
        max_chars = self.config["proactive"]["reflection_max_chars"]
        environment = await self.probe.probe()
        plan = plan_recall(
            self.config["retrieval"], environment["learn_available"], environment["bge_available"]
        )
        args = recall_args(
            query,
            titles=True,
            limit=max(1, self.config["proactive"]["max_items"]),
            project=turn.scope,
            type="hypothesis",
            lexical=plan["use_lexical"],
        )
        try:
            result = await self.rsrs.run(args, timeout_ms=self.config["recall_timeout_ms"])
        except RsrsError as exc:
            self.state.log(f"reflection recall failed: {exc.code}")
            return []
        items = select_items(
            result["envelope"], plan["use_lexical"], self.config["min_score_hybrid"], 1
        )
        items = [item for item in items if len(item_title(item)) <= max_chars]
        if not items:
            return []
        self.gate.record(turn.scope, key)
        return items

    # ------------------------------------------------------------------ #
    # Response handling: auto-feedback and auto-diary
    # ------------------------------------------------------------------ #
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        try:
            self._after_response(event, resp)
        except Exception as exc:
            logger.warning("dafeiyu: response hook failed open: %s: %s", type(exc).__name__, truncate(str(exc), 300))

    def _after_response(self, event, resp):
        turn = self._turn(event)
        reply = getattr(resp, "completion_text", "") or ""
        pending = self.state.latest_pending(turn.scope)
        if pending and reply:
            hits = cited_ids(reply, pending.get("ids") or [])
            if hits:
                self.state.take_pending(turn.scope, hits)
                self._spawn(
                    self._mark(
                        turn.scope, hits, "good", pending.get("request_id"), "model", pending.get("id_map")
                    )
                )
        decision = diary_decision(
            text=getattr(event, "message_str", "") or "",
            reply=reply,
            scope=turn.scope,
            is_group=turn.is_group,
            config=self.config,
        )
        if decision["ok"]:
            self._spawn(self._store_diary(turn, decision))
        else:
            self.state.log(f"diary skip: {decision['reason']}")
        # Automatic self-note refresh: only a real private-owner reply counts.
        # Count independently of whether the diary itself is stored: a user who
        # turns auto_diary off still wants the persona to grow. The command,
        # secret and empty checks are the same ones used by the diary decision.
        count_ok = decision["ok"]
        if not count_ok and decision["reason"] == "disabled":
            count_ok = diary_decision(
                text=getattr(event, "message_str", "") or "",
                reply=reply,
                scope=turn.scope,
                is_group=turn.is_group,
                config={**self.config, "auto_diary": True},
            )["ok"]
        if (
            count_ok
            and not turn.is_group
            and self._is_admin(turn.user_id)
            and self.selfnote.auto_enabled
        ):
            try:
                self.selfnote.note_turn(turn)
                due, _why = self.selfnote.auto_due(turn)
            except Exception as exc:  # fail open: never block a reply
                due = False
                logger.warning("dafeiyu: selfnote turn count failed open: %s", type(exc).__name__)
            if due:
                self._spawn(self.selfnote.maybe_auto(turn))

    def _mark_correction(self, turn):
        pending = self.state.latest_pending(turn.scope)
        if not pending:
            return
        taken = self.state.take_pending(turn.scope)
        ids = (taken or {}).get("ids") or []
        if not ids:
            return
        self.state.log(f"correction: marking {len(ids)} ids bad")
        self._spawn(
            self._mark(turn.scope, ids, "bad", taken.get("request_id"), "human", taken.get("id_map"))
        )

    async def _mark(self, scope, ids, verdict, request_id, source, id_map=None):
        if not request_id:
            # Legacy grading without --query-log-id accepts only --source model.
            # Refuse a human/task label rather than misattributing provenance.
            self.state.log("feedback skipped: no query_log_id")
            return False
        mapping = id_map if isinstance(id_map, dict) else {}
        resolved = [mapping.get(str(value)[:8], str(value)) for value in ids]
        if request_id and any(len(str(value)) < 32 for value in resolved):
            self.state.log("feedback rejected: short ids without a full-id map")
            logger.warning("dafeiyu: feedback needs full ids when a query-log id is used")
            return False
        try:
            args = query_log_mark_args(resolved, verdict, query_log_id=request_id, source=source)
        except RsrsError as exc:
            self.state.log(f"feedback rejected: {exc}")
            return False
        try:
            await self.rsrs.run(args, timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            self.state.log(f"feedback failed: {exc.code}: {truncate(str(exc), 200)}")
            return False
        self.state.log(f"feedback {verdict} {len(ids)} scope={scope}")
        self._queue_train()
        return True

    def _queue_train(self):
        stamp = now_ms()
        last = int(self.state.read_jobs().get("last_train_ms", 0) or 0)
        if stamp - last < TRAIN_DEBOUNCE_MS:
            return False
        self.state.write_jobs({"last_train_ms": stamp})
        self._spawn(self._train_now())
        return True

    async def _train_now(self, limit=None):
        probe = await self.rsrs.probe_learn()
        if not probe["learn_available"]:
            return {"ok": False, "reason": "learn unavailable"}
        try:
            args = learn_train_args(limit=int(limit or self.config["jobs"]["daily_limit"]), epochs=1)
            result = await self.rsrs.run(args, timeout_ms=TRAIN_TIMEOUT_MS)
        except RsrsError as exc:
            self.state.log(f"train failed: {exc.code}")
            return {"ok": False, "reason": truncate(str(exc), 200)}
        counts = summarize_learn(result["envelope"])
        self.state.log(f"train ok {counts}")
        return {"ok": True, "counts": counts}

    async def _store_diary(self, turn, decision):
        # T1-3: summaries are generated on every write path; the probe is
        # cached, and a gate-off host stays on the contract-0 write.
        typed_ok = bool((await self.rsrs.metadata_status()).get("enabled"))
        content = f"{speaker_header(turn, self._speaker_past(turn))}\n{decision['content']}"
        args = remember_args(
            content,
            title=decision["title"],
            summary=make_summary(decision["title"], content) if typed_ok else None,
            type="Context",
            importance="trivial",
            tags=f"qq,diary,{sender_tag(turn)}",
            project=turn.scope,
            force=True,  # diary rows are intentionally repetitive; skip dedup
        )
        try:
            await self.rsrs.run(args, timeout_ms=ORDINARY_TIMEOUT_MS)
            self.state.log(f"diary stored scope={turn.scope}")
        except RsrsError as exc:
            self.state.log(f"diary failed: {exc.code}: {truncate(str(exc), 200)}")
            logger.warning("dafeiyu: diary store failed: %s", truncate(str(exc), 200))

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
    @filter.command("mem")
    async def mem(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem")
        if rest is None:
            return
        yield event.plain_result(self._help_text())

    def _help_text(self):
        mode = self.config["retrieval"]
        lines = [
            "dafeiyu memory",
            f"mode: {mode} | scope: {self.config['scope_mode']}",
            "commands:",
            "/mem_status - health",
            "/mem_recall <q> - search this chat",
            "/mem_show <id> - body",
            "/mem_remember <text> - store fact",
            "/mem_good <ids> | /mem_bad <ids> - label",
            "/mem_mute <topic> | /mem_mute list | /mem_unmute <topic>",
            "/mem_forget <id> - delete (admin)",
            "/mem_reflect <text> - hypothesis (admin)",
            "/mem_learn [status|train|reset|lexical-on|lexical-off] (admin)",
            "/mem_flywheel - weekly encoder (admin)",
            "/mem_audit - read-only store audit (admin)",
            "/mem_review - due reviews, read-only (admin)",
            "/mem_selfnote [show|auto|preview|refresh|set <brief>|clear] (admin)",
            "/mem_clear - reset this chat's recall state",
        ]
        return clip("\n".join(lines), STATUS_REPLY_LIMIT)

    @filter.command("mem_status")
    async def mem_status(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_status")
        if rest is None:
            return
        yield event.plain_result(await self._status_text(event))

    async def _status_text(self, event):
        turn = self._turn(event)
        lines = ["memory status"]
        environment = await self.probe.probe()
        planned = plan_recall(
            self.config["retrieval"], environment["learn_available"], environment["bge_available"]
        )
        # The CLI's persisted lexical/hybrid mode wins over our plan; report both.
        scheme = environment.get("cli_mode") or ("lexical" if planned["use_lexical"] else "hybrid")
        lines.append(f"scheme: {scheme} (retrieval={self.config['retrieval']})")
        lines.append(f"rsrs: {environment['bin'] or 'not found'} ({environment['source']})")
        if environment["learn_available"]:
            lines.append("learn: available")
        else:
            lines.append(f"learn: unavailable - {clip(environment['warning'] or 'unknown', 120)}")
        lines.append(f"bge: {'yes' if environment['bge_available'] else 'no'}")
        encoder = encoder_status(
            granite_model_dir=self.config["jobs"]["granite_model_dir"],
            respire_repo=self.config["jobs"]["respire_repo"],
        )
        if encoder["model"]["installed"]:
            index = read_index_meta(self.config["jobs"]["anchor_dir"])
            bits = [f"granite: installed {index['version'] or 'no-index'}"]
            if index["docs"]:
                bits.append(f"docs={index['docs']}")
            if index["dim"]:
                bits.append(f"dim={index['dim']}")
            for key in ("hit20", "mrr20"):
                value = index["metrics"].get(key)
                if isinstance(value, (int, float)):
                    bits.append(f"{key}={value:.4g}")
            lines.append(" ".join(bits))
            if not encoder["ready"]:
                lines.append(f"encoder: {clip(encoder.get('reason') or 'not ready', 120)}")
        else:
            lines.append(f"granite: missing {', '.join(encoder['model']['missing'])}")
        jobs_record = self.state.read_jobs()
        daily = jobs_record.get("daily") if isinstance(jobs_record.get("daily"), dict) else {}
        if daily.get("finished_at"):
            lines.append(f"jobs: daily={daily.get('status')} scheduler={'on' if self.config['jobs']['enabled'] else 'off'}")
        else:
            lines.append(f"jobs: scheduler={'on' if self.config['jobs']['enabled'] else 'off'}")
        maintenance = jobs_record.get("maintenance") if isinstance(jobs_record.get("maintenance"), dict) else {}
        if maintenance:
            when = _format_ms(maintenance.get("finished_at")) or "unknown"
            audit = maintenance.get("audit") if isinstance(maintenance.get("audit"), dict) else {}
            review = maintenance.get("review") if isinstance(maintenance.get("review"), dict) else {}
            bits = [
                "maintenance: {}".format(when),
                "audit={}".format(audit.get("status") or "-"),
                "review={}".format(review.get("status") or "-"),
            ]
            spot = maintenance.get("spotcheck") if isinstance(maintenance.get("spotcheck"), dict) else {}
            if spot:
                # Nonzero findings ride along so a problem is visible without
                # digging into jobs.json; counts only, never record text.
                flagged = ""
                counts = spot.get("counts") if isinstance(spot.get("counts"), dict) else {}
                findings = [
                    f"{key}={counts[key]}"
                    for key in (
                        "reflection_as_evidence",
                        "expired",
                        "review_due",
                        "derived_from_superseded",
                        "source_missing",
                    )
                    if isinstance(counts.get(key), int) and counts.get(key)
                ]
                if findings:
                    flagged = " " + ",".join(findings)
                bits.append("spot={}{}".format(spot.get("status") or "-", flagged))
            lines.append(" ".join(bits))
        else:
            lines.append("maintenance: never")
        try:
            result = await self.rsrs.run(status_args(), timeout_ms=RECALL_TIMEOUT_MS, allow_failure=True)
            summary = (result.get("envelope") or {}).get("summary") or {}
            if isinstance(summary, dict) and isinstance(summary.get("local_total"), int):
                lines.append(f"store: {summary.get('local_total')} entries")
        except Exception:
            lines.append("store: status unavailable")
        lines.append(
            f"pending: {self.state.pending_count(turn.scope)} mutes: {self.state.count_mutes(turn.scope)}"
        )
        try:
            note = self.state.read_selfnote(self.selfnote.build_scope(turn))
        except Exception:
            note = None
        auto_status = self.selfnote.auto_status(turn)
        auto_bits = "auto={} turns={}/{}".format(
            "on" if auto_status.get("enabled") else "off",
            auto_status.get("turns", 0),
            auto_status.get("every_sessions", self.selfnote.every_sessions),
        )
        if isinstance(note, dict) and note.get("id"):
            if turn.is_group and not self._is_admin(turn.user_id):
                lines.append(f"selfnote: managed {auto_bits}")
            else:
                lines.append(
                    f"selfnote: {short_id(note['id'])} updated {note.get('updated_at') or 'unknown'} {auto_bits}"
                )
        else:
            lines.append(f"selfnote: none {auto_bits}")
        lines.append(f"scope: {turn.scope}")
        return clip("\n".join(lines), STATUS_REPLY_LIMIT)

    @filter.command("mem_recall")
    async def mem_recall(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_recall")
        if rest is None:
            return
        yield event.plain_result(await self._recall_text(event, rest))

    async def _recall_text(self, event, rest):
        if rest.strip() == "":
            return "usage: /mem_recall <query>"
        turn = self._turn(event)
        query = rest.strip()[:QUERY_MAX_CHARS]
        environment = await self.probe.probe()
        plan = plan_recall(self.config["retrieval"], environment["learn_available"], environment["bge_available"])
        try:
            envelope = await self._recall_fanout(
                query,
                await self._recall_scopes(turn),
                self.config["recall_limit"],
                plan["use_lexical"],
            )
        except RsrsError as exc:
            return f"recall failed: {clip(str(exc), 200)}"
        items = select_items(envelope, plan["use_lexical"], self.config["min_score_hybrid"], self.config["recall_limit"])
        if not items:
            return "no matches in this chat"
        ids, full_ids, id_map = _recall_id_map(envelope, items)
        request_id = await resolve_query_log_id(self.rsrs, envelope, ids)
        self.state.record_pending(
            turn.scope,
            ids,
            full_ids=full_ids,
            id_map=id_map,
            request_id=request_id,
            query_hash=hash_query(query),
        )
        self._annotate_speakers(items, turn)
        return format_recall_reply(items, RECALL_REPLY_LIMIT)

    @filter.command("mem_show")
    async def mem_show(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_show")
        if rest is None:
            return
        yield event.plain_result(await self._show_text(event, rest))

    async def _show_text(self, event, rest):
        turn = self._turn(event)
        if not self._may_reveal(turn):
            return self._read_denied(turn)
        ids = parse_ids(rest)
        if not ids:
            return "usage: /mem_show <id>"
        full = self._recent_id(turn.scope, ids[0])
        if not full:
            return "that id is not from this chat's recent recall"
        try:
            result = await self.rsrs.run(show_args(full), timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            return f"show failed: {clip(str(exc), 200)}"
        envelope = result["envelope"]
        details = envelope.get("details") if isinstance(envelope.get("details"), dict) else {}
        entry = details.get("entry") if isinstance(details.get("entry"), dict) else {}
        summary = envelope.get("summary") if isinstance(envelope.get("summary"), dict) else {}
        title = summary.get("title") or entry.get("title") or ""
        body = entry.get("content") or ""
        status = await self._annotated_status(entry)
        head = f"[{short_id(full)}] {title}" + (f"\n状态: {status}" if status else "")
        if not body:
            return head if status else f"no body for {short_id(full)}"
        return clip(f"{head}\n{body}", SHOW_REPLY_LIMIT)

    @filter.command("mem_remember")
    async def mem_remember(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_remember")
        if rest is None:
            return
        yield event.plain_result(await self._remember_text(event, rest))

    async def _remember_text(self, event, rest):
        turn = self._turn(event)
        # 用户裁决 2026-11-05 放开群写入: open to every sender, written into
        # this sender's own member bucket with their 【发言人】 attribution.
        if not self._may_write(turn):
            return self._write_denied(turn)
        content = rest.strip()
        if content == "":
            return "usage: /mem_remember <text>"
        title = "QQ fact: " + clip(content, 60)
        # T1-3 summary is contract-1: only send it when the v1 gate is on.
        typed_ok = bool((await self.rsrs.metadata_status()).get("enabled"))
        args = remember_args(
            content,
            title=title,
            summary=make_summary(title, content) if typed_ok else None,
            type="Context",
            importance="important",
            tags="qq,fact",
            project=turn.scope,
        )
        try:
            result = await self.rsrs.run(args, timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            return f"remember failed: {clip(str(exc), 200)}"
        envelope = result["envelope"]
        summary = envelope.get("summary") if isinstance(envelope.get("summary"), dict) else {}
        stored_id = summary.get("id") if isinstance(summary.get("id"), str) else ""
        if stored_id:
            stored_title = summary.get("title") if isinstance(summary.get("title"), str) else title
            return clip(f"stored [{short_id(stored_id)}] {stored_title}", 400)
        suggestion = summary.get("reason") or envelope.get("status") or "not stored"
        return clip(f"not stored: {suggestion}", 400)

    @filter.command("mem_person")
    @_admin_only
    async def mem_person(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_person")
        if rest is None:
            return
        yield event.plain_result(await self._person_text(event, rest))

    async def _person_text(self, event, rest):
        """``/mem_person set <hash|昵称> <relation>`` and ``/mem_person list``.

        Admin only and fail-closed: an empty ``admin_ids`` refuses both, an
        unknown relation is rejected before any write, and an ambiguous or
        unknown nickname resolves to nothing rather than guessing.
        """
        turn = self._turn(event)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        parts = rest.split()
        if not parts:
            return "usage: /mem_person set <hash|昵称> <relation>  |  /mem_person list"
        action = parts[0].lower()
        if action == "list":
            return self._person_list_text()
        if action == "set":
            if len(parts) < 3:
                return "usage: /mem_person set <hash|昵称> <relation>"
            return self._person_set_text(turn, parts[1], parts[2])
        return "usage: /mem_person set <hash|昵称> <relation>  |  /mem_person list"

    def _person_list_text(self):
        persons = self.state.list_persons()
        if not persons:
            return "no person profiles yet"
        lines = [f"{len(persons)} person profile(s):"]
        for key in sorted(persons):
            entry = persons[key] if isinstance(persons[key], dict) else {}
            lines.append(
                f"{key}  {entry.get('current_nickname') or '-'}  {entry.get('relation') or 'stranger'}"
                f"  first={_person_stamp(entry.get('first_seen'))}"
                f" last={_person_stamp(entry.get('last_seen'))}"
            )
        return "\n".join(lines)

    def _resolve_person_key(self, who):
        """A 16-hex hash, else a nickname that matches exactly one profile.

        Zero or several nickname matches resolve to ``None`` so a typo can never
        rewrite somebody else's relation.
        """
        token = str(who or "").strip()
        if re.fullmatch(r"[0-9a-fA-F]{16}", token):
            return token.lower()
        persons = self.state.list_persons()
        matches = [
            key
            for key, entry in persons.items()
            if str((entry or {}).get("current_nickname") or "").strip() == token
        ]
        return matches[0] if len(matches) == 1 else None

    def _person_set_text(self, turn, who, relation):
        if relation not in PERSON_RELATIONS:
            return f"relation must be one of {'|'.join(PERSON_RELATIONS)}, got: {relation}"
        key = self._resolve_person_key(who)
        if not key:
            return f"no single person matches {who}; pass the 16-hex hash"
        entry = self.state.set_person_relation(
            key, relation, set_by=f"user:{person_hash(turn.user_id)}"
        )
        return (
            f"relation set: {key} {entry.get('current_nickname') or '-'} -> "
            f"{entry.get('relation')} (by {entry.get('set_by')})"
        )

    @filter.command("mem_forget")
    @_admin_only
    async def mem_forget(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_forget")
        if rest is None:
            return
        yield event.plain_result(await self._forget_text(event, rest))

    async def _forget_text(self, event, rest):
        turn = self._turn(event)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        ids = parse_ids(rest)
        if not ids:
            return "usage: /mem_forget <id>"
        full = self._recent_id(turn.scope, ids[0])
        if not full:
            return "that id is not from this chat's recent recall"
        try:
            await self.rsrs.run(forget_args(full), timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            return f"forget failed: {clip(str(exc), 200)}"
        return f"forgot {short_id(full)}"

    @filter.command("mem_good")
    async def mem_good(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_good")
        if rest is None:
            return
        yield event.plain_result(await self._feedback_text(event, rest, "good"))

    @filter.command("mem_bad")
    async def mem_bad(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_bad")
        if rest is None:
            return
        yield event.plain_result(await self._feedback_text(event, rest, "bad"))

    async def _feedback_text(self, event, rest, verdict):
        turn = self._turn(event)
        ids = parse_ids(rest)
        if not ids:
            return f"usage: /mem_{verdict} <ids>"
        pending = self.state.latest_pending(turn.scope)
        if not pending:
            # The CLI needs a query-log id when feedback is enabled; without a
            # recent recall there is nothing to attach the label to.
            return "no recent recall in this chat; run /mem_recall first"
        request_id = pending.get("request_id")
        marked = await self._mark(
            turn.scope, ids, verdict, request_id, "human", pending.get("id_map")
        )
        if not marked:
            return f"{verdict} not recorded (no matching recall in the last 10 min)"
        self.state.take_pending(turn.scope, ids)
        queued = self._queue_train()
        return f"{verdict} recorded for {len(ids)} id(s)" + ("; training queued" if queued else "; training debounced")

    @filter.command("mem_mute")
    async def mem_mute(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_mute")
        if rest is None:
            return
        yield event.plain_result(self._mute_text(event, rest))

    def _mute_text(self, event, rest):
        turn = self._turn(event)
        topic = rest.strip()
        if topic == "":
            return "usage: /mem_mute <topic> | /mem_mute list"
        if topic.lower() == "list":
            mutes = self.state.list_mutes(turn.scope)
            return f"muted topics in this chat: {len(mutes)}"
        key = topic_key(topic)
        if not key:
            return "usage: /mem_mute <topic>"
        until = now_ms() + MUTE_HOURS * 3600 * 1000
        self.state.add_mute(turn.scope, key, until_ms=until, kind="topic", hash_value=key, label=topic)
        return f"muted: {clip(topic, 80)} ({MUTE_HOURS // 24}d)"

    @filter.command("mem_unmute")
    async def mem_unmute(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_unmute")
        if rest is None:
            return
        yield event.plain_result(self._unmute_text(event, rest))

    def _unmute_text(self, event, rest):
        turn = self._turn(event)
        topic = rest.strip()
        key = topic_key(topic)
        if not key:
            return "usage: /mem_unmute <topic>"
        removed = self.state.remove_mute(turn.scope, key)
        return f"unmuted: {clip(topic, 80)}" if removed else "no such mute"

    @filter.command("mem_reflect")
    @_admin_only
    async def mem_reflect(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_reflect")
        if rest is None:
            return
        yield event.plain_result(await self._reflect_text(event, rest))

    async def _reflect_text(self, event, rest):
        turn = self._turn(event)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        content = rest.strip()
        if content == "":
            return "usage: /mem_reflect <text> [with [id] for the cause]"
        parent = None
        match = _ID_IN_TEXT_RE.search(content)
        if match:
            parent = match.group(1)
        ttl_days = int(self.config["proactive"]["reflection_ttl_days"])
        review_days = max(1, ttl_days - 4)  # 7 -> review after 3 days
        try:
            args, typed = await self._reflect_store(
                turn, content, parent or "", ttl_days, "QQ reflection: " + clip(content, 60)
            )
            result = await self.rsrs.run(args, timeout_ms=ORDINARY_TIMEOUT_MS)
        except RsrsError as exc:
            return f"reflect failed: {clip(str(exc), 200)}"
        summary = (result.get("envelope") or {}).get("summary") or {}
        stored_id = summary.get("id") if isinstance(summary.get("id"), str) else ""
        head = f"reflection stored [{short_id(stored_id)}]" if stored_id else "reflection submitted"
        if not typed:
            return clip(f"{head} - unverified hypothesis (metadata v1 off: stored untyped, no expiry)", 400)
        return clip(f"{head} - unverified hypothesis, expires in {ttl_days}d, review in {review_days}d", 400)

    @filter.command("mem_learn")
    @_admin_only
    async def mem_learn(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_learn")
        if rest is None:
            return
        yield event.plain_result(await self._learn_text(event, rest))

    async def _learn_text(self, event, rest):
        turn = self._turn(event)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        action = rest.strip().lower() or "status"
        probe = await self.rsrs.probe_learn()
        if not probe["learn_available"]:
            return clip(
                "learn unavailable: this rsrs build has no `learn` subcommand "
                f"({clip(probe.get('reason') or 'unknown', 120)}). Install the managed "
                "build at ~/.local/bin/rsrs-dafeiyu or set rsrs_bin.",
                400,
            )
        if action == "status":
            args = learn_status_args()
        elif action == "train":
            args = learn_train_args(limit=int(self.config["jobs"]["daily_limit"]), epochs=1)
        elif action == "reset":
            args = learn_reset_args()
        elif action == "lexical-on":
            args = learn_lexical_args(True)
        elif action == "lexical-off":
            args = learn_lexical_args(False)
        else:
            return "usage: /mem_learn [status|train|reset|lexical-on|lexical-off]"
        timeout = TRAIN_TIMEOUT_MS if action == "train" else ORDINARY_TIMEOUT_MS
        try:
            result = await self.rsrs.run(args, timeout_ms=timeout)
        except RsrsError as exc:
            return f"learn {action} failed: {clip(str(exc), 200)}"
        counts = summarize_learn(result["envelope"])
        detail = " ".join(f"{key}={value}" for key, value in counts.items() if key != "status")
        return clip(f"learn {action}: {result['envelope'].get('status', 'ok')} {detail}".strip(), 600)

    @filter.command("mem_flywheel")
    @_admin_only
    async def mem_flywheel(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_flywheel")
        if rest is None:
            return
        yield event.plain_result(await self._flywheel_text(event))

    async def _flywheel_text(self, event):
        turn = self._turn(event)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        encoder = encoder_status(
            granite_model_dir=self.config["jobs"]["granite_model_dir"],
            respire_repo=self.config["jobs"]["respire_repo"],
        )
        if not encoder["ready"]:
            return clip(
                f"flywheel unavailable: {encoder.get('reason') or 'not ready'}. "
                "Expected model.safetensors/config.json/1_Pooling in granite_model_dir "
                "and scripts/encoder/weekly_encoder.sh in respire_repo.",
                400,
            )
        # The pipeline can run for hours: schedule it and answer immediately.
        self._spawn(self._run_flywheel(encoder["script"]["script"]))
        return clip("flywheel started in background; progress goes to the encoder log", 400)

    async def _run_flywheel(self, script_path):
        env = dict(self.rsrs.env)
        env.update(
            {
                "RSRS_BIN": self.rsrs.bin or "",
                "RSRS_ENCODER_BASE": self.config["jobs"]["granite_model_dir"],
                "RSRS_ENCODER_STATE": self.config["jobs"]["anchor_dir"],
            }
        )
        if self.config["data_dir"]:
            env["RSRS_DATA_DIR"] = self.config["data_dir"]
        else:
            env.pop("RSRS_DATA_DIR", None)
        log_path = self.state.files["encoder_log"]
        append_log(log_path, f"--- manual flywheel start {script_path}")
        try:
            result = await run_process("bash", [script_path], env=env, timeout_ms=ENCODER_TIMEOUT_MS)
        except RsrsError as exc:
            append_log(log_path, f"--- manual flywheel failed: {truncate(str(exc), 500)}")
            logger.warning("dafeiyu: manual flywheel failed: %s", truncate(str(exc), 200))
            return
        append_log(log_path, truncate(result["stdout"], 20000))
        if result["stderr"]:
            append_log(log_path, truncate(result["stderr"], 20000))
        append_log(log_path, f"--- manual flywheel exit {result['code']}")

    @filter.command("mem_audit")
    async def mem_audit(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_audit")
        if rest is None:
            return
        yield event.plain_result(await self._audit_command_text(event))

    async def _audit_command_text(self, event):
        turn = self._turn(event)
        if not self._may_admin(turn):
            return "admin only"
        return await self._audit_view(1200)

    @filter.command("mem_review")
    async def mem_review(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_review")
        if rest is None:
            return
        yield event.plain_result(await self._review_text(event))

    async def _review_text(self, event):
        turn = self._turn(event)
        if not self._may_admin(turn):
            return "admin only"
        limit = int(self.config["jobs"].get("review_limit") or 10)
        try:
            result = await self.rsrs.run(
                review_due_args(limit), timeout_ms=REVIEW_TIMEOUT_MS, allow_failure=True
            )
        except RsrsError as exc:
            return f"review failed: {clip(str(exc), 200)}"
        reason = readonly_error(result)
        if reason:
            return f"review failed: {clip(reason, 200)}"
        envelope = result.get("envelope") or {}
        counts = summarize_review(envelope)
        status = counts.get("status") or envelope.get("status") or "ok"
        due = counts.get("due")
        if isinstance(due, bool) or not isinstance(due, int):
            due = counts.get("items", 0)
        lines = [f"review: {status} | due {due}"]
        lines.extend(_review_lines(envelope, 10))
        return clip("\n".join(lines), RECALL_REPLY_LIMIT)

    @filter.command("mem_selfnote")
    async def mem_selfnote(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_selfnote")
        if rest is None:
            return
        yield event.plain_result(await self._selfnote_text(event, rest))

    def _selfnote_auto_line(self, turn):
        """One read-only line describing the automatic refresh state."""
        try:
            status = self.selfnote.auto_status(turn)
        except Exception as exc:
            logger.warning("dafeiyu: selfnote auto status failed: %s", type(exc).__name__)
            return "auto: off"
        if not status.get("enabled"):
            return "auto: off"
        line = "auto: on turns {}/{} last {} failures {}".format(
            status.get("turns", 0),
            status.get("every_sessions", self.selfnote.every_sessions),
            status.get("last_auto_at") or "never",
            status.get("failures", 0),
        )
        if status.get("next_due_at"):
            line += f" next {status['next_due_at']}"
        return line

    async def _selfnote_text(self, event, rest):
        """``/mem_selfnote [show|auto|preview|refresh|set <brief>|clear]``.

        ``auto`` is read-only status; it never changes the configuration.
        Read commands allow the private owner or an admin; writes require an
        admin id. Every failure is a one-line reply, never an exception.
        """
        turn = self._turn(event)
        if turn.is_group:
            return "selfnote is managed in a private chat"
        text = str(rest or "").strip()
        action, _, argument = text.partition(" ")
        action = action.lower() or "show"
        if action == "show":
            if not self._admin_allowed(turn.user_id):
                return "admin only"
            try:
                data = await self.selfnote.show(turn)
            except SelfNoteError as exc:
                return f"selfnote failed: {clip(str(exc), 200)}"
            except Exception as exc:
                logger.warning("dafeiyu: selfnote show failed: %s", type(exc).__name__)
                return "selfnote failed"
            auto_line = self._selfnote_auto_line(turn)
            if not data.get("brief") and not data.get("id"):
                return clip(f"selfnote: none\n{auto_line}", 1200)
            head = f"selfnote: {short_id(data['id'])}" if data.get("id") else "selfnote: unversioned"
            lines = [head]
            if data.get("updated_at"):
                lines.append(f"updated: {data['updated_at']}")
            lines.append(data.get("brief") or "")
            lines.append(auto_line)
            return clip("\n".join(lines), 1200)
        if action == "auto":
            if not self._admin_allowed(turn.user_id):
                return "admin only"
            return clip(self._selfnote_auto_line(turn), 600)
        if not self._admin_allowed(turn.user_id):
            return "admin only"
        if action not in ("preview", "refresh", "set", "clear"):
            return "usage: /mem_selfnote [show|auto|preview|refresh|set <brief>|clear]"
        if action == "clear":
            try:
                removed = await self.selfnote.clear(turn)
            except SelfNoteError as exc:
                return f"selfnote failed: {clip(str(exc), 200)}"
            except Exception as exc:
                logger.warning("dafeiyu: selfnote clear failed: %s", type(exc).__name__)
                return "selfnote failed"
            if not removed:
                return "selfnote: none; memories untouched"
            return "selfnote cleared; memories untouched"
        if action == "set":
            brief = argument.strip()
            if brief == "":
                return "usage: /mem_selfnote set <brief>"
            if len(brief) > self.selfnote.max_brief:
                return f"brief too long (max {self.selfnote.max_brief})"
            try:
                result = await self.selfnote.set(turn, brief)
            except SelfNoteError as exc:
                return f"selfnote failed: {clip(str(exc), 200)}"
            except Exception as exc:
                logger.warning("dafeiyu: selfnote set failed: %s", type(exc).__name__)
                return "selfnote failed"
            suffix = "" if result.get("state_saved", True) else " (state pointer pending)"
            return clip(f"selfnote set [{short_id(result['id'])}]{suffix}\n{result['brief'][:200]}", 600)
        try:
            candidate = await self.selfnote.generate(turn)
            if action == "preview":
                sources = ", ".join(candidate.get("source_ids") or [])
                lines = [
                    f"selfnote preview (provider {clip(str(candidate.get('provider_id') or ''), 60)})",
                    "full:",
                    candidate["full"],
                    "brief:",
                    candidate["brief"],
                    f"source_ids: {sources}",
                ]
                return clip("\n".join(lines), 2400)
            result = await self.selfnote.apply(turn, candidate)
        except SelfNoteError as exc:
            return f"selfnote failed: {clip(str(exc), 200)}"
        except Exception as exc:
            logger.warning("dafeiyu: selfnote %s failed: %s", action, type(exc).__name__)
            return "selfnote failed"
        suffix = "" if result.get("state_saved", True) else " (state pointer pending)"
        return clip(
            f"selfnote updated [{short_id(result['id'])}]{suffix}\n{result['brief'][:200]}",
            600,
        )

    @filter.command("mem_clear")
    async def mem_clear(self, event: AstrMessageEvent):
        rest = self._rest(event, "mem_clear")
        if rest is None:
            return
        turn = self._turn(event)
        cleared = self.state.clear_scope(turn.scope)
        yield event.plain_result(
            f"cleared recall state for this chat (pending={cleared['pending']}); memories untouched"
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        try:
            enabled = self.config["jobs"]["enabled"]
            maintenance = self.config["jobs"].get("maintenance_enabled", True)
            if enabled or maintenance:
                started = self.jobs.start()
                self.state.log(
                    f"jobs start requested: {started} (training={enabled} maintenance={maintenance})"
                )
            else:
                self.state.log("jobs disabled by config")
        except Exception as exc:
            logger.warning("dafeiyu: job startup failed: %s", truncate(str(exc), 200))

    async def terminate(self):
        try:
            await self.jobs.stop()
        except Exception:
            pass
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
