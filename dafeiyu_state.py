"""Local plugin state for the AstrBot adapter.

Host bookkeeping only — the authority store is rsrs. This module owns:

  * the state directory (0700) and atomic JSON files (0600)
  * append-only debug / encoder logs
  * per-scope mute records and cooldowns
  * the pending-recall ring used by feedback
  * person profiles (persons.json, keyed by sha256(user_id)[:16])
  * job locks and last-run records

Every write goes through a temp file in the same directory followed by
``os.replace``, so a crash can never leave a half-written state file.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

#: Directory mode for the state root.
DIR_MODE = 0o700
#: File mode for every state file.
FILE_MODE = 0o600
#: Pending-recall snapshots live this long.
PENDING_TTL_MS = 10 * 60 * 1000
#: Maximum pending-recall snapshots retained.
PENDING_MAX = 50
#: A job lock older than this is treated as stale (crashed run).
LOCK_STALE_MS = 25 * 60 * 60 * 1000
#: Person relations, mirroring the starmem entity kinds (stranger/friend/
#: master/other_fish). ``self`` has no host-side mapping: the bot itself never
#: speaks as a member, and the owner is modelled as ``master``.
PERSON_RELATIONS = ("stranger", "friend", "master", "other_fish")
#: Nicknames retained per person (history beyond this is dropped, oldest first).
NICKNAMES_MAX = 20


def now_ms():
    """Wall-clock milliseconds; pending/cooldown TTLs must survive restarts."""
    return int(time.time() * 1000)


def person_hash(user_id):
    """Stable person key: ``sha256(user_id)[:16]``.

    The raw QQ id never reaches disk — only this truncated hash, matching the
    hashed-scope convention in ``dafeiyu_policy.hashed_id``.
    """
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:16]


def person_is_admin(user_id, admin_ids):
    """True when ``user_id`` is listed in the plugin's ``admin_ids``.

    ``admin_ids`` comes from the config as raw QQ ids (strings or ints); the
    comparison is on the string form, mirroring ``main._is_admin``.
    """
    if admin_ids is None:
        return False
    if isinstance(admin_ids, str):
        candidates = [part.strip() for part in admin_ids.replace(",", " ").split() if part.strip()]
    else:
        candidates = [str(item).strip() for item in admin_ids]
    return str(user_id).strip() in candidates


def ensure_dir(directory, mode=DIR_MODE):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    try:
        os.chmod(path, mode)
    except OSError:
        pass  # best effort on filesystems without POSIX modes
    return str(path)


def read_json(file_path, fallback):
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if text.strip() == "":
            return fallback
        return json.loads(text)
    except (OSError, ValueError):
        return fallback


def write_json_atomic(file_path, value, mode=FILE_MODE):
    """Write JSON via a 0600 temp file plus ``os.replace``."""
    path = Path(file_path)
    ensure_dir(path.parent)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        os.chmod(tmp, mode)
    except OSError:
        pass
    os.replace(tmp, path)
    try:
        os.chmod(path, mode)
    except OSError:
        pass
    return str(path)


def append_log(file_path, line):
    """Append one line, creating the file 0600. Never raises."""
    try:
        path = Path(file_path)
        ensure_dir(path.parent)
        if not path.exists():
            os.close(os.open(str(path), os.O_WRONLY | os.O_CREAT, FILE_MODE))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
    except OSError:
        pass


class StateStore:
    """Filesystem-backed state for one plugin data directory."""

    def __init__(self, state_dir, debug=False, now=None):
        self.state_dir = ensure_dir(state_dir)
        self.logs_dir = ensure_dir(os.path.join(self.state_dir, "logs"))
        self.debug = bool(debug)
        self._now = now or now_ms
        self._lock = threading.RLock()
        self.files = {
            "permissions": os.path.join(self.state_dir, "permissions.json"),
            "pending": os.path.join(self.state_dir, "pending.json"),
            "cooldowns": os.path.join(self.state_dir, "cooldowns.json"),
            "jobs": os.path.join(self.state_dir, "jobs.json"),
            "selfnote": os.path.join(self.state_dir, "selfnote.json"),
            "persons": os.path.join(self.state_dir, "persons.json"),
            "job_lock": os.path.join(self.state_dir, "jobs.lock"),
            "encoder_log": os.path.join(self.logs_dir, "encoder.log"),
            "debug_log": os.path.join(self.logs_dir, "dafeiyu.log"),
        }

    # -- logging -----------------------------------------------------------
    def log(self, message):
        if not self.debug:
            return
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        append_log(self.files["debug_log"], f"{stamp} {message}")

    # -- mutes -------------------------------------------------------------
    def _read_permissions(self):
        raw = read_json(self.files["permissions"], {"entries": {}})
        if not isinstance(raw, dict) or not isinstance(raw.get("entries"), dict):
            return {"entries": {}}
        return raw

    def _write_permissions(self, value):
        write_json_atomic(self.files["permissions"], value)

    def add_mute(self, scope, key, *, until_ms, kind=None, hash_value=None, label=None, created_at_ms=None):
        with self._lock:
            current = self._read_permissions()
            record = {
                "scope": str(scope),
                "key": str(key),
                "until_ms": int(until_ms),
                "created_at_ms": int(created_at_ms if created_at_ms is not None else self._now()),
            }
            if kind:
                record["kind"] = str(kind)
            if hash_value:
                record["hash"] = str(hash_value)
            if label:
                record["label"] = str(label)[:200]
            current["entries"][f"{scope}|{key}"] = record
            self._write_permissions(current)
            return record

    def remove_mute(self, scope, key):
        with self._lock:
            current = self._read_permissions()
            composite = f"{scope}|{key}"
            if composite not in current["entries"]:
                return False
            del current["entries"][composite]
            self._write_permissions(current)
            return True

    def list_mutes(self, scope=None):
        """Live (unexpired) mute records, optionally restricted to one scope."""
        with self._lock:
            current = self._read_permissions()
        stamp = self._now()
        out = []
        for record in current["entries"].values():
            if not isinstance(record, dict):
                continue
            if int(record.get("until_ms", 0)) <= stamp:
                continue
            if scope is not None and record.get("scope") != scope:
                continue
            out.append(dict(record))
        out.sort(key=lambda item: int(item.get("until_ms", 0)))
        return out

    def count_mutes(self, scope=None):
        return len(self.list_mutes(scope))

    def is_muted(self, scope, key):
        with self._lock:
            current = self._read_permissions()
        record = current["entries"].get(f"{scope}|{key}")
        if not isinstance(record, dict):
            return False
        return int(record.get("until_ms", 0)) > self._now()

    # -- pending recalls ---------------------------------------------------
    def _read_pending(self):
        raw = read_json(self.files["pending"], {"items": []})
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return []
        stamp = self._now()
        live = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if stamp - int(item.get("ts", 0)) >= PENDING_TTL_MS:
                continue
            live.append(item)
        return live

    def _write_pending(self, items):
        write_json_atomic(self.files["pending"], {"items": items[-PENDING_MAX:]})

    def record_pending(self, scope, ids, *, full_ids=None, id_map=None, request_id=None, query_hash=None, ts=None):
        with self._lock:
            items = self._read_pending()
            short = [str(item) for item in (ids or [])][:50]
            mapping = {str(key): str(value) for key, value in (id_map or {}).items()}
            full = [str(item) for item in (full_ids or [])][:50]
            if not full:
                full = [mapping.get(value[:8], value) for value in short]
            items.append(
                {
                    "scope": str(scope),
                    "ids": short,
                    "full_ids": full,
                    "id_map": mapping,
                    "request_id": request_id or None,
                    "query_hash": query_hash or None,
                    "ts": int(ts if ts is not None else self._now()),
                }
            )
            self._write_pending(items)
            return len(items[-PENDING_MAX:])

    def latest_pending(self, scope):
        with self._lock:
            items = self._read_pending()
        for item in reversed(items):
            if item.get("scope") == scope and item.get("ids"):
                return item
        return None

    def recent_pending(self, scope, ttl_ms=PENDING_TTL_MS):
        """Every live pending snapshot for one scope, newest first.

        The pending ring keeps several recalls, so an id from an earlier recall
        in the same chat stays usable. Read-only: nothing is written or pruned
        here (``_read_pending`` already drops TTL-expired snapshots), and the
        result is still capped at ``PENDING_MAX`` entries.
        """
        with self._lock:
            items = self._read_pending()
        stamp = self._now()
        live = []
        for item in items:
            if item.get("scope") != scope:
                continue
            if not item.get("ids"):
                continue
            if ttl_ms is not None and stamp - int(item.get("ts", 0)) >= int(ttl_ms):
                continue
            live.append(dict(item))
        live.reverse()
        return live[:PENDING_MAX]

    def take_pending(self, scope, ids=None):
        """Remove and return the newest pending entry matching ``scope``.

        When ``ids`` is given the entry is kept with those ids removed (a partial
        good-marking must not discard the remaining candidates).
        """
        with self._lock:
            items = self._read_pending()
            for index in range(len(items) - 1, -1, -1):
                item = items[index]
                if item.get("scope") != scope:
                    continue
                if ids is None:
                    items.pop(index)
                    self._write_pending(items)
                    return item
                wanted = {str(value) for value in ids}
                remaining = [value for value in item.get("ids", []) if str(value) not in wanted]
                hit = [value for value in item.get("ids", []) if str(value) in wanted]
                if not hit:
                    continue
                mapping = item.get("id_map") if isinstance(item.get("id_map"), dict) else {}
                hit_map = {key: mapping[key] for key in mapping if key in {str(value)[:8] for value in hit}}
                if remaining:
                    keep = {str(value)[:8] for value in remaining}
                    items[index] = {
                        **item,
                        "ids": remaining,
                        "full_ids": [mapping.get(value[:8], value) for value in remaining],
                        "id_map": {key: value for key, value in mapping.items() if key in keep},
                    }
                else:
                    items.pop(index)
                self._write_pending(items)
                return {**item, "ids": hit, "id_map": hit_map}
        return None

    def pending_count(self, scope=None):
        with self._lock:
            items = self._read_pending()
        if scope is None:
            return len(items)
        return len([item for item in items if item.get("scope") == scope])

    # -- cooldowns ---------------------------------------------------------
    def _read_cooldowns(self):
        raw = read_json(self.files["cooldowns"], {"scopes": {}, "topics": {}})
        if not isinstance(raw, dict):
            return {"scopes": {}, "topics": {}}
        if not isinstance(raw.get("scopes"), dict):
            raw["scopes"] = {}
        if not isinstance(raw.get("topics"), dict):
            raw["topics"] = {}
        return raw

    def get_cooldown(self, scope):
        with self._lock:
            raw = self._read_cooldowns()
        entry = raw["scopes"].get(scope)
        return dict(entry) if isinstance(entry, dict) else {}

    def set_cooldown(self, scope, *, last_inject_ms=None, query_hash=None, query_ms=None):
        with self._lock:
            raw = self._read_cooldowns()
            entry = raw["scopes"].get(scope)
            entry = dict(entry) if isinstance(entry, dict) else {}
            if last_inject_ms is not None:
                entry["last_inject_ms"] = int(last_inject_ms)
            if query_hash is not None:
                entry["query_hash"] = str(query_hash)
            if query_ms is not None:
                entry["query_ms"] = int(query_ms)
            raw["scopes"][scope] = entry
            write_json_atomic(self.files["cooldowns"], raw)
            return entry

    def clear_scope(self, scope):
        """Drop pending recalls and cooldown bookkeeping for one scope."""
        with self._lock:
            raw = self._read_cooldowns()
            removed_cooldown = raw["scopes"].pop(scope, None)
            raw["topics"].pop(scope, None)
            write_json_atomic(self.files["cooldowns"], raw)
            items = self._read_pending()
            kept = [item for item in items if item.get("scope") != scope]
            removed_pending = len(items) - len(kept)
            self._write_pending(kept)
        return {"pending": removed_pending, "cooldown": removed_cooldown is not None}

    def get_topic_cooldown(self, scope, key):
        with self._lock:
            raw = self._read_cooldowns()
        topics = raw["topics"].get(scope)
        if not isinstance(topics, dict):
            return 0
        return int(topics.get(key, 0) or 0)

    def set_topic_cooldown(self, scope, key, until_ms):
        with self._lock:
            raw = self._read_cooldowns()
            topics = raw["topics"].get(scope)
            topics = dict(topics) if isinstance(topics, dict) else {}
            topics[str(key)] = int(until_ms)
            raw["topics"][scope] = topics
            write_json_atomic(self.files["cooldowns"], raw)

    # -- self-note pointers ------------------------------------------------
    def _read_selfnote(self):
        raw = read_json(self.files["selfnote"], {"scopes": {}})
        if not isinstance(raw, dict) or not isinstance(raw.get("scopes"), dict):
            return {"scopes": {}}
        return raw

    @staticmethod
    def _selfnote_entry(value):
        """Normalize one pointer record; older files lack the auto fields."""
        entry = value if isinstance(value, dict) else {}
        try:
            auto_turns = int(entry.get("auto_turns") or 0)
        except (TypeError, ValueError):
            auto_turns = 0
        try:
            auto_failures = int(entry.get("auto_failures") or 0)
        except (TypeError, ValueError):
            auto_failures = 0
        return {
            "id": str(entry.get("id") or ""),
            "brief": str(entry.get("brief") or ""),
            "updated_at": str(entry.get("updated_at") or ""),
            "sources": [str(item) for item in (entry.get("sources") or []) if str(item)],
            "auto_turns": max(0, auto_turns),
            "last_auto_at": str(entry.get("last_auto_at") or ""),
            "auto_failures": max(0, auto_failures),
        }

    def read_selfnote(self, scope):
        """Pointer to the current self-note version for one scope, or ``None``.

        The record is a pointer plus automatic-refresh bookkeeping:
        ``{"id", "brief", "updated_at", "sources", "auto_turns",
        "last_auto_at", "auto_failures"}``. The persona block stays the visible
        authority; the rsrs record keeps the versioned full text. An older
        ``selfnote.json`` without the auto fields reads back as
        ``auto_turns=0``, ``last_auto_at=""``, ``auto_failures=0``.
        """
        with self._lock:
            raw = self._read_selfnote()
        entry = raw["scopes"].get(str(scope))
        if not isinstance(entry, dict):
            return None
        return self._selfnote_entry(entry)

    def write_selfnote(self, scope, record):
        """Atomically replace one scope's self-note pointer.

        Auto-refresh counters are preserved when the caller does not supply
        them, so a plain ``apply`` never resets progress by accident; a missing
        record still starts at ``0`` / ``""``.
        """
        value = record if isinstance(record, dict) else {}
        with self._lock:
            raw = self._read_selfnote()
            previous = raw["scopes"].get(str(scope))
            previous = previous if isinstance(previous, dict) else {}
            merged = {
                "id": value.get("id"),
                "brief": value.get("brief"),
                "updated_at": value.get("updated_at"),
                "sources": value.get("sources"),
                "auto_turns": value.get("auto_turns", previous.get("auto_turns")),
                "last_auto_at": value.get("last_auto_at", previous.get("last_auto_at")),
                "auto_failures": value.get("auto_failures", previous.get("auto_failures")),
            }
            entry = self._selfnote_entry(merged)
            raw["scopes"][str(scope)] = entry
            write_json_atomic(self.files["selfnote"], raw)
            return dict(entry)

    def bump_selfnote_turn(self, scope, ts=None):
        """Count one real private-owner reply; create the record when absent.

        ``ts`` is accepted for symmetry with :meth:`mark_selfnote_auto` and is
        intentionally unused: a turn only advances the counter.
        """
        with self._lock:
            raw = self._read_selfnote()
            entry = self._selfnote_entry(raw["scopes"].get(str(scope)))
            entry["auto_turns"] = int(entry["auto_turns"]) + 1
            raw["scopes"][str(scope)] = entry
            write_json_atomic(self.files["selfnote"], raw)
            return dict(entry)

    def mark_selfnote_auto(self, scope, *, ok, ts=None, failures_delta=None):
        """Record the outcome of one automatic refresh attempt.

        Success resets the turn counter and the failure count; failure keeps
        the counter and adds one (or ``failures_delta``) while stamping
        ``last_auto_at`` so the backoff can be measured.
        """
        stamp = str(ts) if isinstance(ts, str) and ts else str(int(ts if ts is not None else self._now()))
        with self._lock:
            raw = self._read_selfnote()
            entry = self._selfnote_entry(raw["scopes"].get(str(scope)))
            if ok:
                entry["auto_turns"] = 0
                entry["last_auto_at"] = stamp
                entry["auto_failures"] = 0
            else:
                delta = 1 if failures_delta is None else max(0, int(failures_delta))
                entry["last_auto_at"] = stamp
                entry["auto_failures"] = int(entry["auto_failures"]) + delta
            raw["scopes"][str(scope)] = entry
            write_json_atomic(self.files["selfnote"], raw)
            return dict(entry)

    def get_selfnote_auto(self, scope):
        """Read-only automatic-refresh view; ``None`` when no record exists."""
        return self.read_selfnote(scope)

    def clear_selfnote(self, scope):
        """Drop one scope's pointer; the persona block and memories stay."""
        with self._lock:
            raw = self._read_selfnote()
            removed = raw["scopes"].pop(str(scope), None)
            write_json_atomic(self.files["selfnote"], raw)
            return removed is not None

    # -- person profiles ---------------------------------------------------
    def _read_persons(self):
        raw = read_json(self.files["persons"], {})
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _person_entry(value, *, hash_key, nickname, now_ms_value, is_admin=False, group_hash=None):
        """Normalize one person record.

        A fresh record defaults to ``relation="stranger"``; an admin user id is
        auto-promoted to ``master``. A manually set relation on an existing
        record is never downgraded by a later automatic pass (task-19 手工优先).

        ``group_hash`` records a group this person was seen in. ``groups`` is a
        list field and is therefore ALWAYS carried forward: every upsert rebuilds
        the record from scratch, so a conditionally-copied list would vanish on
        the sender's next message — the same lesson ``set_by`` taught.
        """
        entry = value if isinstance(value, dict) else {}
        stamp = str(now_ms_value)
        nick = str(nickname or entry.get("current_nickname") or "").strip()
        history = entry.get("nicknames")
        history = [str(item) for item in history if str(item)] if isinstance(history, list) else []
        if nick:
            if not history or history[-1] != nick:
                history.append(nick)
            history = history[-NICKNAMES_MAX:]
        relation = str(entry.get("relation") or "").strip()
        if relation not in PERSON_RELATIONS:
            # Unknown or missing relation: default, but an admin id wins.
            relation = "master" if is_admin else "stranger"
        elif is_admin and relation != "master" and not entry.get("set_by"):
            # First sight of an admin, or a record written before task-19's
            # manual setter existed. An EMPTY set_by counts as "no marker", so
            # the guard's meaning matches how the marker is carried forward.
            relation = "master"
        groups = [str(item) for item in (entry.get("groups") or []) if str(item)]
        if group_hash:
            token = str(group_hash)
            if token not in groups:
                groups.append(token)
        record = {
            "current_nickname": nick or str(entry.get("current_nickname") or ""),
            "nicknames": history,
            "first_seen": str(entry.get("first_seen") or stamp),
            "last_seen": stamp,
            "relation": relation,
            "note": str(entry.get("note") or ""),
            "groups": groups,
            "updated_at": stamp,
        }
        # ``set_by`` is the evidence that the relation was a human decision, and
        # every upsert rebuilds this record from scratch. Without carrying it
        # forward the guard above loses its evidence on the sender's *second*
        # message and auto-promote silently overwrites a manual relation.
        # Carried only when present: an unconditional empty string would make
        # ``not entry.get("set_by")`` permanently false and disable auto-promote.
        if entry.get("set_by"):
            record["set_by"] = str(entry["set_by"])
        return record

    def read_person(self, user_hash):
        """One person record by hash, or ``None``."""
        key = str(user_hash)
        with self._lock:
            raw = self._read_persons()
        entry = raw.get(key)
        return dict(entry) if isinstance(entry, dict) else None

    def list_persons(self):
        """Every person record keyed by hash (read-only copy)."""
        with self._lock:
            raw = self._read_persons()
        return {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}

    def upsert_person(self, user_hash, nickname, now=None, *, admin=False, group_hash=None):
        """Create or refresh one person profile; idempotent per (hash, nickname).

        * new hash → record with ``relation="stranger"`` (``"master"`` when
          ``admin`` is true), ``first_seen == last_seen == now``;
        * same nickname again → only ``last_seen`` / ``updated_at`` move;
        * changed nickname → the old one is kept in ``nicknames`` history;
        * ``group_hash`` appends to ``groups`` (deduplicated) so a group read can
          find every member who has spoken in that group.

        The raw QQ id never appears here: callers pass ``person_hash(user_id)``.
        The write is atomic (temp file + ``os.replace``), and the debug log
        records only the hash — never the nickname.
        """
        key = str(user_hash)
        stamp = int(now if now is not None else self._now())
        with self._lock:
            raw = self._read_persons()
            previous = raw.get(key)
            entry = self._person_entry(
                previous,
                hash_key=key,
                nickname=nickname,
                now_ms_value=stamp,
                is_admin=bool(admin),
                group_hash=group_hash,
            )
            raw[key] = entry
            write_json_atomic(self.files["persons"], raw)
        self.log(f"person upsert hash={key} relation={entry['relation']} seen={stamp}")
        return dict(entry)

    def set_person_relation(self, user_hash, relation, *, now=None, note=None, set_by=None):
        """Set a person's relation explicitly (manual wins over auto-master).

        ``relation`` must be one of :data:`PERSON_RELATIONS`; the setter is
        recorded as ``set_by`` so a later ``admin=True`` upsert never silently
        rewrites a human decision.
        """
        if relation not in PERSON_RELATIONS:
            raise ValueError(f"relation must be one of {PERSON_RELATIONS}, got {relation!r}")
        key = str(user_hash)
        stamp = int(now if now is not None else self._now())
        with self._lock:
            raw = self._read_persons()
            previous = raw.get(key)
            entry = self._person_entry(
                previous, hash_key=key, nickname=None, now_ms_value=stamp, is_admin=False
            )
            entry["relation"] = str(relation)
            entry["updated_at"] = str(stamp)
            if note is not None:
                entry["note"] = str(note)
            if set_by is not None:
                entry["set_by"] = str(set_by)
            raw[key] = entry
            write_json_atomic(self.files["persons"], raw)
        self.log(f"person relation hash={key} relation={entry['relation']}")
        return dict(entry)

    # -- jobs --------------------------------------------------------------
    def read_jobs(self):
        raw = read_json(self.files["jobs"], {})
        return raw if isinstance(raw, dict) else {}

    def write_jobs(self, patch):
        with self._lock:
            merged = {**self.read_jobs(), **patch}
            write_json_atomic(self.files["jobs"], merged)
            return merged

    def acquire_job_lock(self, stale_ms=LOCK_STALE_MS):
        """Exclusive lock file. Returns a release callable, or ``None`` if held."""
        path = self.files["job_lock"]
        stamp = self._now()
        try:
            if os.path.exists(path):
                try:
                    age = stamp - int(os.stat(path).st_mtime * 1000)
                except OSError:
                    age = 0
                if age > stale_ms:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
        except OSError:
            return None
        try:
            os.write(descriptor, json.dumps({"pid": os.getpid(), "at": stamp}).encode("utf-8"))
        finally:
            os.close(descriptor)

        def release():
            try:
                os.unlink(path)
            except OSError:
                pass

        return release
