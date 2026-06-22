"""
datekeywords.py — shared date-vocabulary config.

Loads `date_keywords.json` (path overridable via DATE_KEYWORDS_PATH) once at
import time and exposes it to BOTH consumers:

  * utils/datetools.py — uses the keyword lists for deterministic date selection
    (rent-commencement priority, open-ended period detection).
  * utils/llmapi.py    — injects a "Date keyword reference" block into the system
    prompt of date-bearing tasks, so the extraction stage captures the same
    signal phrases the post-processor relies on.

If the file is missing or malformed, baked-in DEFAULTS keep the pipeline working.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Fallback defaults (mirror date_keywords.json so the code works without it)
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "rentStart": {
        "label": "Rent commencement (distinct from lease start)",
        "hint": "The date rent begins to accrue, which may differ from the lease "
                "commencement date (a rent-free period). Capture the date that "
                "follows these phrases.",
        "keywords": ["租金給付始期", "起租日", "租金起算日", "租金起算",
                     "租金計算期間自", "計租起始日", "租金計算自"],
    },
    "leaseSpanStart": {
        "label": "Lease span opener",
        "hint": "The lease commencement date sits between/after these markers "
                "(e.g. 自<date>起).",
        "keywords": ["自", "起"],
    },
    "leaseSpanEnd": {
        "label": "Lease span terminator",
        "hint": "The lease end date sits before these markers (e.g. 至<date>止).",
        "keywords": ["至", "止", "迄"],
    },
    "openEnded": {
        "label": "Lease expiry (open-ended final period)",
        "hint": "Marks a period that runs until lease expiry rather than a fixed "
                "end date.",
        "keywords": ["屆滿", "届满", "期滿", "期满", "租約期滿", "租期屆滿", "租約屆滿"],
    },
    "relativePeriod": {
        "label": "Relative payment periods (preserve verbatim)",
        "hint": "Year-relative rate periods. Copy the exact phrase; downstream "
                "tools derive calendar dates from the lease/rent anchor.",
        "keywords": ["第N年至第M年", "第N年起", "第N年"],
    },
}


def _load() -> Dict[str, Any]:
    path = pathlib.Path(os.getenv("DATE_KEYWORDS_PATH", "date_keywords.json"))
    if not path.exists():
        return DEFAULTS
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return DEFAULTS
    # Shallow-merge over defaults so a partial file still resolves every group.
    merged = {k: dict(v) for k, v in DEFAULTS.items()}
    for key, group in data.items():
        if key.startswith("_") or not isinstance(group, dict):
            continue
        merged.setdefault(key, {}).update(group)
    return merged


CONFIG: Dict[str, Any] = _load()


def _kw(group: str) -> Tuple[str, ...]:
    return tuple(CONFIG.get(group, {}).get("keywords", []))


# Convenience exports consumed by datetools.py
RENT_START_KEYWORDS: Tuple[str, ...] = _kw("rentStart")
OPEN_ENDED_KEYWORDS: Tuple[str, ...] = _kw("openEnded")
LEASE_START_MARKERS: Tuple[str, ...] = _kw("leaseSpanStart")
LEASE_END_MARKERS: Tuple[str, ...] = _kw("leaseSpanEnd")


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------

def build_prompt_hint() -> str:
    """Render the keyword config as a system-prompt reference block."""
    lines = [
        "## Date keyword reference",
        "When a field concerns dates, copy the verbatim contract text into its "
        "`content` and be sure to include any of these signal phrases (and the "
        "date next to them) when present — downstream tools rely on them to "
        "compute exact calendar dates:",
    ]
    for group in CONFIG.values():
        if not isinstance(group, dict) or "keywords" not in group:
            continue
        label = group.get("label", "")
        kws = "、".join(group["keywords"])
        lines.append(f"- {label}: {kws}")
    return "\n".join(lines)


# Keys / field names that mark a task as date-bearing.
_DATE_MARKERS = ("periodStartDate", "periodEndDate")


def is_date_task(task_entry: Dict[str, Any]) -> bool:
    """True if the task's outputSchema involves period start/end dates."""
    schema = task_entry.get("outputSchema")
    if not schema:
        return False
    blob = json.dumps(schema, ensure_ascii=False)
    return any(marker in blob for marker in _DATE_MARKERS)
