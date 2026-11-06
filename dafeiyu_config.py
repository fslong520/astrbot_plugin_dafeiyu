"""Configuration for the AstrBot "dafeiyu" memory adapter.

`resolve_config()` is dependency-free and never raises: every field falls back to
a default and every value is clamped into range. A typo in the AstrBot WebUI must
never be able to break a chat turn.

The AstrBot plugin config arrives as a dict-like object (``AstrBotConfig``), so
the normalizer only relies on ``keys()`` / ``[]`` and accepts both snake_case
(the schema field names) and camelCase spellings.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PLUGIN_NAME = "astrbot_plugin_dafeiyu"

#: Retrieval strategies accepted by ``retrieval``.
RETRIEVAL_MODES = ("auto", "lexical", "hybrid", "granite")

#: Privacy scopes accepted by ``scope_mode``.
SCOPE_MODES = ("per_conversation", "per_user", "global")

#: Single source of truth for defaults; also documented in ``_conf_schema.json``.
DEFAULTS = {
    "enabled": True,
    "auto_recall": True,
    "retrieval": "auto",
    "rsrs_bin": "",
    "data_dir": "",
    "rpc_port": 15270,
    "recall_limit": 20,
    "recall_budget_chars": 1500,
    "recall_excerpt_chars": 120,
    "recall_timeout_ms": 1800,
    "cooldown_ms": 30000,
    "same_query_cooldown_ms": 600000,
    "min_score_hybrid": 0.42,
    "scope_mode": "per_conversation",
    "admin_ids": (),
    "auto_diary": True,
    "auto_diary_group": False,
    "debug_log": False,
}

DEFAULT_PROACTIVE = {
    "enabled": False,
    "max_items": 1,
    "permission_turns": 5,
    "permission_ms": 1800000,
    "topic_cooldown_ms": 86400000,
    "reflection_ttl_days": 7,
    "reflection_max_chars": 350,
}

DEFAULT_JOBS = {
    "enabled": False,
    "daily_hour": 3,
    "daily_minute": 30,
    "daily_limit": 5000,
    "weekly_granite": True,
    "maintenance_enabled": True,
    "maintenance_hour": 4,
    "maintenance_minute": 0,
    "review_limit": 20,
    "spotcheck_limit": 10000,
    "spotcheck_window_days": 14,
    "respire_repo": "~/Documents/respire/respire-cli",
    "granite_model_dir": "~/.rsrs/models/granite-97m-r2",
    "anchor_dir": "~/.rsrs/encoder",
}

#: Self-note (living persona block). Limits mirror ``dafeiyu_selfnote``.
DEFAULT_SELFNOTE = {
    "enabled": True,
    "auto_enabled": False,
    "every_sessions": 20,
    "min_interval_ms": 604800000,
    "retry_backoff_ms": 21600000,
    "persona_id": "default",
    "owner_id": "",
    "parent_id": "",
    "provider_id": "",
    "evidence_limit": 20,
    "max_full_chars": 1200,
    "max_brief_chars": 300,
}

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def home_of(env=None):
    """Home directory used for ``~`` expansion."""
    env = env if isinstance(env, dict) else os.environ
    value = env.get("HOME") or env.get("USERPROFILE")
    if isinstance(value, str) and value.strip() != "":
        return value
    return str(Path.home())


def expand_path(value, env=None):
    """Expand a leading ``~`` and ``$VAR`` / ``${VAR}`` references.

    Unknown variables collapse to an empty string instead of raising.
    """
    if not isinstance(value, str):
        return ""
    out = value.strip()
    if out == "":
        return ""
    env = env if isinstance(env, dict) else os.environ
    home = home_of(env)
    if out == "~":
        out = home
    elif out.startswith("~/") or out.startswith("~\\"):
        out = str(Path(home) / out[2:])

    def _sub(match):
        key = match.group(1) or match.group(2)
        resolved = env.get(key)
        return "" if resolved is None else str(resolved)

    return _VAR_RE.sub(_sub, out)


def as_bool(value, fallback):
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "yes", "on"):
        return True
    if value in (0, "0", "false", "False", "no", "off"):
        return False
    return fallback


def as_num(value, fallback, *, minimum=0, maximum=10**12, integer=True):
    try:
        next_value = float(value)
    except (TypeError, ValueError):
        next_value = float(fallback)
    if next_value != next_value:  # NaN
        next_value = float(fallback)
    if integer:
        next_value = int(next_value)
    if next_value < minimum:
        next_value = minimum
    if next_value > maximum:
        next_value = maximum
    return next_value


def as_text(value, fallback=""):
    return value.strip() if isinstance(value, str) else fallback


def _pick(source, *names, default=None):
    for name in names:
        if name in source:
            value = source[name]
            if value is not None:
                return value
    return default


def _merge_nested(value, defaults):
    merged = dict(defaults)
    if isinstance(value, dict):
        for key, item in value.items():
            if item is not None:
                merged[str(key)] = item
    return merged


def normalize_admin_ids(value):
    """Accept a list, a tuple or a comma separated string of raw QQ ids."""
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    elif isinstance(value, str):
        items = [part.strip() for part in re.split(r"[,\s]+", value)]
    else:
        items = []
    seen = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


def default_state_dir(env=None, data_root=None):
    """``<AstrBot data>/plugin_data/astrbot_plugin_dafeiyu``."""
    env = env if isinstance(env, dict) else os.environ
    if data_root:
        base = Path(expand_path(str(data_root), env))
    elif env.get("ASTRBOT_DATA_DIR"):
        base = Path(expand_path(str(env["ASTRBOT_DATA_DIR"]), env))
    else:
        base = Path(home_of(env)) / ".astrbot" / "data"
    return str(base / "plugin_data" / PLUGIN_NAME)


def resolve_config(raw=None, *, env=None, cwd=None, home=None, data_root=None):
    """Normalize raw plugin config into a fully-resolved plain dict."""
    env = dict(os.environ) if env is None else dict(env)
    if home:
        env["HOME"] = str(home)
    cwd_path = Path(cwd or os.getcwd())

    if raw is None:
        source = {}
    elif isinstance(raw, dict):
        source = raw
    else:
        try:
            source = {str(key): raw[key] for key in raw.keys()}
        except Exception:
            source = {}

    retrieval = as_text(_pick(source, "retrieval", default=DEFAULTS["retrieval"]), DEFAULTS["retrieval"])
    if retrieval not in RETRIEVAL_MODES:
        retrieval = DEFAULTS["retrieval"]

    scope_mode = as_text(_pick(source, "scope_mode", "scopeMode", default=DEFAULTS["scope_mode"]), DEFAULTS["scope_mode"])
    if scope_mode not in SCOPE_MODES:
        scope_mode = DEFAULTS["scope_mode"]

    proactive = _merge_nested(_pick(source, "proactive"), DEFAULT_PROACTIVE)
    jobs = _merge_nested(_pick(source, "jobs"), DEFAULT_JOBS)
    selfnote = _merge_nested(_pick(source, "selfnote"), DEFAULT_SELFNOTE)

    state_raw = as_text(_pick(source, "state_dir", "stateDir", default=""))
    state_dir = expand_path(state_raw, env) if state_raw else default_state_dir(env, data_root)
    if not state_dir:
        state_dir = default_state_dir(env, data_root)

    data_dir = expand_path(as_text(_pick(source, "data_dir", "dataDir", default="")), env)
    if data_dir:
        data_dir = str((cwd_path / data_dir).resolve()) if not os.path.isabs(data_dir) else data_dir

    resolved = {
        "enabled": as_bool(_pick(source, "enabled"), DEFAULTS["enabled"]),
        "auto_recall": as_bool(_pick(source, "auto_recall", "autoRecall"), DEFAULTS["auto_recall"]),
        "retrieval": retrieval,
        "rsrs_bin": expand_path(as_text(_pick(source, "rsrs_bin", "rsrsBin", default="")), env),
        "data_dir": data_dir,
        # Dedicated RPC port for a non-default data_dir (see dafeiyu_rsrs.Rsrs):
        # 0 disables the override and keeps the machine's default runtime.
        "rpc_port": as_num(_pick(source, "rpc_port", "rpcPort"), DEFAULTS["rpc_port"], minimum=0, maximum=65535),
        "recall_limit": as_num(_pick(source, "recall_limit", "recallLimit"), DEFAULTS["recall_limit"], minimum=1, maximum=50),
        "recall_budget_chars": as_num(
            _pick(source, "recall_budget_chars", "recallBudgetChars"),
            DEFAULTS["recall_budget_chars"],
            minimum=200,
            maximum=20000,
        ),
        "recall_excerpt_chars": as_num(
            _pick(source, "recall_excerpt_chars", "recallExcerptChars"),
            DEFAULTS["recall_excerpt_chars"],
            minimum=0,
            maximum=500,
        ),
        "recall_timeout_ms": as_num(
            _pick(source, "recall_timeout_ms", "recallTimeoutMs"),
            DEFAULTS["recall_timeout_ms"],
            minimum=100,
            maximum=120000,
        ),
        "cooldown_ms": as_num(_pick(source, "cooldown_ms", "cooldownMs"), DEFAULTS["cooldown_ms"], minimum=0, maximum=86400000),
        "same_query_cooldown_ms": as_num(
            _pick(source, "same_query_cooldown_ms", "sameQueryCooldownMs"),
            DEFAULTS["same_query_cooldown_ms"],
            minimum=0,
            maximum=604800000,
        ),
        "min_score_hybrid": as_num(
            _pick(source, "min_score_hybrid", "minScoreHybrid"),
            DEFAULTS["min_score_hybrid"],
            minimum=0.0,
            maximum=1.0,
            integer=False,
        ),
        "scope_mode": scope_mode,
        "admin_ids": normalize_admin_ids(_pick(source, "admin_ids", "adminIds", default=())),
        "auto_diary": as_bool(_pick(source, "auto_diary", "autoDiary"), DEFAULTS["auto_diary"]),
        "auto_diary_group": as_bool(_pick(source, "auto_diary_group", "autoDiaryGroup"), DEFAULTS["auto_diary_group"]),
        "debug_log": as_bool(_pick(source, "debug_log", "debugLog"), DEFAULTS["debug_log"]),
        "proactive": {
            "enabled": as_bool(proactive.get("enabled"), DEFAULT_PROACTIVE["enabled"]),
            "max_items": as_num(proactive.get("max_items"), DEFAULT_PROACTIVE["max_items"], minimum=0, maximum=10),
            "permission_turns": as_num(
                proactive.get("permission_turns"), DEFAULT_PROACTIVE["permission_turns"], minimum=0, maximum=100
            ),
            "permission_ms": as_num(proactive.get("permission_ms"), DEFAULT_PROACTIVE["permission_ms"], minimum=0, maximum=86400000),
            "topic_cooldown_ms": as_num(
                proactive.get("topic_cooldown_ms"), DEFAULT_PROACTIVE["topic_cooldown_ms"], minimum=0, maximum=604800000
            ),
            "reflection_ttl_days": as_num(
                proactive.get("reflection_ttl_days"), DEFAULT_PROACTIVE["reflection_ttl_days"], minimum=1, maximum=365
            ),
            "reflection_max_chars": as_num(
                proactive.get("reflection_max_chars"), DEFAULT_PROACTIVE["reflection_max_chars"], minimum=40, maximum=2000
            ),
        },
        "jobs": {
            "enabled": as_bool(jobs.get("enabled"), DEFAULT_JOBS["enabled"]),
            "daily_hour": as_num(jobs.get("daily_hour"), DEFAULT_JOBS["daily_hour"], minimum=0, maximum=23),
            "daily_minute": as_num(jobs.get("daily_minute"), DEFAULT_JOBS["daily_minute"], minimum=0, maximum=59),
            "daily_limit": as_num(jobs.get("daily_limit"), DEFAULT_JOBS["daily_limit"], minimum=1, maximum=1000000),
            "weekly_granite": as_bool(jobs.get("weekly_granite"), DEFAULT_JOBS["weekly_granite"]),
            "maintenance_enabled": as_bool(
                jobs.get("maintenance_enabled"), DEFAULT_JOBS["maintenance_enabled"]
            ),
            "maintenance_hour": as_num(
                jobs.get("maintenance_hour"), DEFAULT_JOBS["maintenance_hour"], minimum=0, maximum=23
            ),
            "maintenance_minute": as_num(
                jobs.get("maintenance_minute"), DEFAULT_JOBS["maintenance_minute"], minimum=0, maximum=59
            ),
            "review_limit": as_num(
                jobs.get("review_limit"), DEFAULT_JOBS["review_limit"], minimum=1, maximum=100
            ),
            "spotcheck_limit": as_num(
                jobs.get("spotcheck_limit"), DEFAULT_JOBS["spotcheck_limit"], minimum=100, maximum=50000
            ),
            "spotcheck_window_days": as_num(
                jobs.get("spotcheck_window_days"), DEFAULT_JOBS["spotcheck_window_days"],
                minimum=1, maximum=365
            ),
            "respire_repo": expand_path(as_text(jobs.get("respire_repo"), DEFAULT_JOBS["respire_repo"]), env),
            "granite_model_dir": expand_path(as_text(jobs.get("granite_model_dir"), DEFAULT_JOBS["granite_model_dir"]), env),
            "anchor_dir": expand_path(as_text(jobs.get("anchor_dir"), DEFAULT_JOBS["anchor_dir"]), env),
        },
        "selfnote": {
            # `enabled` only gates the LLM tool; the explicit /mem_selfnote
            # commands keep working so an admin can always inspect or fix the
            # block. `auto_enabled` turns on the low-frequency background
            # refresh: only real private-owner replies count toward
            # `every_sessions`, and one attempt needs both the turn count and
            # `min_interval_ms` since the last attempt. A failed attempt waits
            # `retry_backoff_ms` before the next one. Off by default.
            "enabled": as_bool(selfnote.get("enabled"), DEFAULT_SELFNOTE["enabled"]),
            "auto_enabled": as_bool(selfnote.get("auto_enabled"), DEFAULT_SELFNOTE["auto_enabled"]),
            "every_sessions": as_num(
                selfnote.get("every_sessions"), DEFAULT_SELFNOTE["every_sessions"], minimum=5, maximum=100
            ),
            "min_interval_ms": as_num(
                selfnote.get("min_interval_ms"),
                DEFAULT_SELFNOTE["min_interval_ms"],
                minimum=3600000,
                maximum=2592000000,
            ),
            "retry_backoff_ms": as_num(
                selfnote.get("retry_backoff_ms"),
                DEFAULT_SELFNOTE["retry_backoff_ms"],
                minimum=600000,
                maximum=604800000,
            ),
            "persona_id": as_text(selfnote.get("persona_id"), DEFAULT_SELFNOTE["persona_id"]) or "default",
            "owner_id": str(_pick(selfnote, "owner_id", default="") or "").strip(),
            "parent_id": str(_pick(selfnote, "parent_id", default="") or "").strip(),
            "provider_id": as_text(selfnote.get("provider_id"), ""),
            "evidence_limit": as_num(
                selfnote.get("evidence_limit"), DEFAULT_SELFNOTE["evidence_limit"], minimum=1, maximum=20
            ),
            "max_full_chars": as_num(
                selfnote.get("max_full_chars"), DEFAULT_SELFNOTE["max_full_chars"], minimum=200, maximum=1200
            ),
            "max_brief_chars": as_num(
                selfnote.get("max_brief_chars"), DEFAULT_SELFNOTE["max_brief_chars"], minimum=40, maximum=300
            ),
        },
        "state_dir": state_dir,
    }
    return resolved
