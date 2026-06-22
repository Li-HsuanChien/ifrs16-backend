"""
datetools.py — deterministic "ground truth" date tools for the IFRS 16 pipeline.

Why this module exists
----------------------
The LLM is excellent at *locating* the verbatim contract text that mentions a
date, but unreliable at the *arithmetic*: ROC→Gregorian conversion and the
"第N年至第M年" → calendar-date derivation.  The prompts even ship with baked-in
mistakes (e.g. 民國122年 rendered as 2023 instead of 2033, and a 第四年至第六年
end of 2027-07-30 instead of 2027-07-31).

So we split responsibilities:
  * the LLM extracts the raw `content` splices (unchanged), and
  * these deterministic tools compute the exact ISO dates and overwrite the
    model's `value` fields — the ground truth.

Two public tools (plus an enrichment layer that applies them to a payload):

  Tool 1 — roc_to_iso(text):           ROC/民國 (or Gregorian) text → ISO 8601.
  Tool 2 — resolve_period_dates(expr, anchor, lease_end):
                                        "第N年至第M年" → (startISO, endISO).

No third-party dependencies — only the standard library.
"""

from __future__ import annotations

import copy
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

try:  # shared date vocabulary (also injected into the LLM prompts)
    from datekeywords import RENT_START_KEYWORDS, OPEN_ENDED_KEYWORDS
except ImportError:
    from utils.datekeywords import RENT_START_KEYWORDS, OPEN_ENDED_KEYWORDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Republic of China (民國) epoch offset:  Gregorian year = ROC year + 1911.
ROC_OFFSET = 1911

# Any 1–3 digit year (value < 1000) is treated as ROC; a 4-digit year is
# assumed to already be Gregorian.  ROC years today are ~100–150, so this is
# unambiguous in practice.
GREGORIAN_MIN_YEAR = 1000

ISO_SUFFIX = "T00:00:00.000Z"

# Number token: ASCII digits or CJK numerals (so we can parse both
# "112年" and "一百一十二年" / "第三年").
_NUM = r"[0-9零〇一二三四五六七八九十百兩两０-９]+"

_DATE_RE = re.compile(rf"({_NUM})\s*年\s*({_NUM})\s*月\s*({_NUM})\s*日?")
_SLASH_RE = re.compile(r"(\d{1,4})\s*[./\-]\s*(\d{1,2})\s*[./\-]\s*(\d{1,2})")

# A single date token (optionally ROC-marked) for structural-marker matching.
_DATE_TOKEN = rf"(?:民國|民国|ROC)?\s*{_NUM}\s*年\s*{_NUM}\s*月\s*{_NUM}\s*日?"

# Structural markers the contracts use to delimit a span: 自/…起 opens it, 至/止
# closes it.  Tried in order; the captured date token is then parsed.
_START_PATS = [
    re.compile(rf"自\s*({_DATE_TOKEN})\s*起"),
    re.compile(rf"({_DATE_TOKEN})\s*起"),
]
_END_PATS = [
    re.compile(rf"至\s*({_DATE_TOKEN})\s*(?:止|迄)"),
    re.compile(rf"({_DATE_TOKEN})\s*(?:止|迄)"),
    re.compile(rf"至\s*({_DATE_TOKEN})"),
]

# RENT_START_KEYWORDS (priority order) is sourced from date_keywords.json via
# datekeywords — the same list the extraction prompt is told to capture.

# Window (chars) searched for a date immediately after a keyword, so we don't
# accidentally grab a far-away unrelated date.
_KEYWORD_WINDOW = 40

# Relative-period expressions, e.g. 第一年至第三年 / 第7年起 / 第四年至第六年.
_RANGE_RE = re.compile(rf"第\s*({_NUM})\s*年\s*(?:至|到|~|～|-|—|－)\s*第?\s*({_NUM})\s*年")
_FROM_RE = re.compile(rf"第\s*({_NUM})\s*年\s*起")
_SINGLE_RE = re.compile(rf"第\s*({_NUM})\s*年")
_OPEN_END_RE = re.compile("|".join(re.escape(k) for k in OPEN_ENDED_KEYWORDS)
                          if OPEN_ENDED_KEYWORDS else r"屆滿|期滿")

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "壹": 1,   # yī  — formal "one"
    "貳": 2,   # èr  — formal "two"
    "參": 3,   # sān — formal "three"  (Traditional)
    "叄": 3,   # sān — alternate form of 參
    "肆": 4,   # sì  — formal "four"
    "伍": 5,   # wǔ  — formal "five"
    "陸": 6,   # liù — formal "six"    (Traditional)
    "陆": 6,   # liù — formal "six"    (Simplified)
    "柒": 7,   # qī  — formal "seven"
    "捌": 8,   # bā  — formal "eight"
    "玖": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "拾": 10,  "佰": 100, 
    "仟": 1000,"十": 10, "百": 100, "千": 1000, "萬": 10000, "万": 10000,
    "億": 100000000, "亿": 100000000,"拾": 10,
    "佰": 100,
    "仟": 1000,
    "萬": 10000,}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Strip OCR/markup noise: full-width digits, bracket fills, parentheticals."""
    if not text:
        return ""
    # Full-width digits → ASCII.
    text = text.translate({ord("０") + i: ord("0") + i for i in range(10)})
    # Drop parenthetical notes like （下同）, (以下同) that sit inside a date run.
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    # Remove fill/quote brackets but keep their inner digits: 【112】 → 112.
    text = re.sub(r"[【】「」〔〕\[\]]", "", text)
    return text


def _to_int(token: str) -> Optional[int]:
    """Parse an int from ASCII digits or CJK numerals (1–999)."""
    token = token.strip()
    if not token:
        return None
    # Full-width → ASCII, then plain digits.
    token = token.translate({ord("０") + i: ord("0") + i for i in range(10)})
    if token.isdigit():
        return int(token)

    total = section = number = 0
    seen = False
    for ch in token:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
            seen = True
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            section += (number or 1) * unit
            number = 0
            seen = True
    total = section + number
    return total if seen else None


def add_years(d: date, n: int) -> date:
    """d shifted by n years, clamping Feb 29 → Feb 28 on non-leap targets."""
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # Feb 29 → non-leap year
        return d.replace(year=d.year + n, day=28)


def to_iso(d: Optional[date]) -> Optional[str]:
    """date → 'YYYY-MM-DDT00:00:00.000Z' (None passes through)."""
    if d is None:
        return None
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}{ISO_SUFFIX}"


def _coerce_date(value: Union[date, str, None]) -> Optional[date]:
    """Accept a date, an ISO string, or ROC/Gregorian text and return a date."""
    if value is None or isinstance(value, date):
        return value
    iso = roc_to_iso(value)
    if iso:
        return date.fromisoformat(iso[:10])
    return None


# ---------------------------------------------------------------------------
# Tool 1 — ROC / Gregorian text → ISO 8601
# ---------------------------------------------------------------------------

def parse_all_dates(text: str) -> List[date]:
    """Return every date found in `text`, left-to-right, as Gregorian dates."""
    raw = text or ""
    is_roc_marked = bool(re.search(r"民國|民国|ROC", raw))
    cleaned = _normalize(raw)

    found: List[Tuple[int, date]] = []

    for m in _DATE_RE.finditer(cleaned):
        y, mo, d = _to_int(m.group(1)), _to_int(m.group(2)), _to_int(m.group(3))
        dt = _build_date(y, mo, d, is_roc_marked)
        if dt:
            found.append((m.start(), dt))

    for m in _SLASH_RE.finditer(cleaned):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dt = _build_date(y, mo, d, is_roc_marked)
        if dt:
            found.append((m.start(), dt))

    found.sort(key=lambda t: t[0])
    return [dt for _, dt in found]


def _build_date(y: Optional[int], mo: Optional[int], d: Optional[int],
                is_roc_marked: bool) -> Optional[date]:
    if not y or not mo or not d:
        return None
    if is_roc_marked or y < GREGORIAN_MIN_YEAR:
        y += ROC_OFFSET
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def roc_to_iso(text: str, which: str = "first") -> Optional[str]:
    """
    Tool 1.  Convert the first (or last) date in `text` to an ISO 8601 string.

    Rules:
      * An explicit 民國 / 民国 / ROC marker, OR any year < 1000, is treated as
        ROC and converted via  Gregorian = ROC + 1911.
      * A 4-digit year is assumed already Gregorian and left as-is.

    `which="last"` returns the final date in the string — useful for an end-date
    splice that carries a whole "X起至Y止" span.
    """
    dates = parse_all_dates(text)
    if not dates:
        return None
    return to_iso(dates[-1] if which == "last" else dates[0])


def extract_lease_span(text: str) -> Optional[Tuple[date, date]]:
    """
    Pull an overall "...起至...止" lease span out of `text`.

    Returns (start, end) Gregorian dates, or None if a two-endpoint span can't
    be identified.  When the text has a 起 / 至 marker we honour it; otherwise we
    fall back to (first date, last date).
    """
    cleaned = _normalize(text or "")
    m = re.search(r"(.*?起)?\s*至\s*(.*)", cleaned, re.DOTALL)
    if m and m.group(1):
        start_dates = parse_all_dates(m.group(1))
        end_dates = parse_all_dates(m.group(2))
        if start_dates and end_dates:
            return start_dates[-1], end_dates[0]

    dates = parse_all_dates(text)
    if len(dates) >= 2:
        return dates[0], dates[-1]
    return None


def _first_date_after_keyword(
    cleaned: str, keywords: Tuple[str, ...]
) -> Optional[date]:
    """First date appearing just after the highest-priority keyword that matches."""
    for kw in keywords:  # keywords are in priority order
        m = re.search(re.escape(kw), cleaned)
        if m:
            window = cleaned[m.end(): m.end() + _KEYWORD_WINDOW]
            dates = parse_all_dates(window)
            if dates:
                return dates[0]
    return None


def select_date(
    text: str,
    role: str = "start",
    keywords: Tuple[str, ...] = (),
) -> Optional[date]:
    """
    Pick the *contextually correct* date from a splice that may hold several.

    Selection order:
      1. Keyword priority — date right after the first matching `keywords` phrase.
      2. Structural markers — 自/…起 for a start, 至/…止 for an end.
      3. Positional fallback — first date for a start, last date for an end.

    `role` is "start" or "end".  Returns a Gregorian date (or None).
    """
    cleaned = _normalize(text or "")

    if keywords:
        d = _first_date_after_keyword(cleaned, keywords)
        if d:
            return d

    for pat in (_START_PATS if role == "start" else _END_PATS):
        m = pat.search(cleaned)
        if m:
            dates = parse_all_dates(m.group(1))
            if dates:
                return dates[0]

    dates = parse_all_dates(cleaned)
    if not dates:
        return None
    return dates[0] if role == "start" else dates[-1]


def find_rent_start(text: str) -> Optional[date]:
    """Locate an explicit rent-commencement anchor (起租日 / 租金計算期間自 …)."""
    return _first_date_after_keyword(_normalize(text or ""), RENT_START_KEYWORDS)


# ---------------------------------------------------------------------------
# Tool 2 — "第N年至第M年" → period start / end dates
# ---------------------------------------------------------------------------

def extract_relative_expr(text: str) -> Optional[str]:
    """Return the relative-period phrase in `text` (第N年…), or None."""
    cleaned = _normalize(text or "")
    m = _RANGE_RE.search(cleaned) or _FROM_RE.search(cleaned) or _SINGLE_RE.search(cleaned)
    return m.group(0) if m else None


def resolve_period_dates(
    expression: str,
    anchor: Union[date, str],
    lease_end: Union[date, str, None] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Tool 2.  Derive (periodStartDate, periodEndDate) ISO strings from a relative
    period expression, anchored on the lease/rent commencement date.

    Conventions (mirroring the contract semantics):
      第N年至第M年  → start = anchor + (N-1) years
                       end   = anchor +  M    years − 1 day
      第N年起[…屆滿] → start = anchor + (N-1) years
                       end   = lease_end            (open-ended final period)
      第N年 (single) → start = anchor + (N-1) years
                       end   = anchor +  N    years − 1 day

    `anchor` and `lease_end` may be date objects or ROC/Gregorian text.
    Contiguity is guaranteed: period N+1's start = period N's end + 1 day.
    """
    anchor_d = _coerce_date(anchor)
    end_d = _coerce_date(lease_end)
    if anchor_d is None:
        return None, None

    cleaned = _normalize(expression or "")
    open_ended = bool(_OPEN_END_RE.search(cleaned))

    m_range = _RANGE_RE.search(cleaned)
    m_from = _FROM_RE.search(cleaned)
    m_single = _SINGLE_RE.search(cleaned)

    if m_range:
        n = _to_int(m_range.group(1))
        m = _to_int(m_range.group(2))
        start = add_years(anchor_d, n - 1)
        end = end_d if open_ended else add_years(anchor_d, m) - timedelta(days=1)
    elif m_from:
        n = _to_int(m_from.group(1))
        start = add_years(anchor_d, n - 1)
        end = end_d  # "第N年起" with no upper bound → runs to lease end
    elif m_single:
        n = _to_int(m_single.group(1))
        start = add_years(anchor_d, n - 1)
        end = end_d if open_ended else add_years(anchor_d, n) - timedelta(days=1)
    else:
        # No relative marker — the whole span is one period.
        start, end = anchor_d, end_d

    return to_iso(start), to_iso(end)


# ---------------------------------------------------------------------------
# Enrichment layer — apply the tools to a parsed payload (the post-processor)
# ---------------------------------------------------------------------------
#
# Runs after the LLM parse step and OVERWRITES every date `value` with the
# deterministic computation, bumping confidence to 100 and tagging the splice
# so the override is auditable.

_GROUND_TRUTH_TAG = "  [date set by ground-truth tool]"


def _set_value(field: Dict[str, Any], iso: Optional[str]) -> None:
    """Overwrite a date field's value + confidence in place (if we computed one)."""
    if not isinstance(field, dict) or iso is None:
        return
    if field.get("value") != iso:
        content = field.get("content") or ""
        if _GROUND_TRUTH_TAG.strip() not in content:
            field["content"] = (content + _GROUND_TRUTH_TAG).strip()
    field["value"] = iso
    field["confidence"] = 100


def _enrich_object(obj: Dict[str, Any]) -> None:
    """LeaseTerms-shaped object: absolute periodStartDate / periodEndDate."""
    if not isinstance(obj, dict):
        return
    # LeaseTerms dates are the lease term itself → structural 起/止 markers
    # (NOT the rent-payment start, which belongs to PaymentTerms).
    start = obj.get("periodStartDate")
    end = obj.get("periodEndDate")
    if isinstance(start, dict):
        _set_value(start, to_iso(select_date(start.get("content", ""), "start")))
    if isinstance(end, dict):
        _set_value(end, to_iso(select_date(end.get("content", ""), "end")))


def _scan_lease_facts(
    items: List[Dict[str, Any]]
) -> Tuple[Optional[date], Optional[date], Optional[date]]:
    """
    Scan every period's date splices for the document-level facts:
    (anchor, lease_start, lease_end).

    The anchor is the rent-commencement date (起租日 / 租金給付始期) when one is
    present — relative 第N年 expressions count from it — otherwise it falls back
    to the lease start (起).
    """
    lease_start: Optional[date] = None
    lease_end: Optional[date] = None
    rent_start: Optional[date] = None
    for it in items:
        for fld in ("periodStartDate", "periodEndDate"):
            content = (it.get(fld) or {}).get("content", "") if isinstance(it.get(fld), dict) else ""
            if not content:
                continue
            span = extract_lease_span(content)
            if span:
                if lease_start is None:
                    lease_start = span[0]
                if lease_end is None:
                    lease_end = span[1]
            rs = find_rent_start(content)
            if rs and rent_start is None:
                rent_start = rs
    anchor = rent_start if rent_start is not None else lease_start
    return anchor, lease_start, lease_end


def _enrich_period_list(items: List[Dict[str, Any]]) -> None:
    """PaymentTerms / CostSeparation: a chronological list of rate periods."""
    if not isinstance(items, list) or not items:
        return

    # 1) Establish the anchor (rent start, else lease start) and lease end by
    #    scanning every period's date splices.
    anchor, _lease_start, lease_end = _scan_lease_facts(items)

    # 2) Resolve each period.
    for it in items:
        start_f = it.get("periodStartDate") if isinstance(it.get("periodStartDate"), dict) else None
        end_f = it.get("periodEndDate") if isinstance(it.get("periodEndDate"), dict) else None
        start_c = start_f.get("content", "") if start_f else ""
        end_c = end_f.get("content", "") if end_f else ""

        expr = extract_relative_expr(f"{start_c} {end_c}")
        if expr and anchor is not None:
            # Relative period (第N年…) → derive from the anchor (rent start if
            # present, else lease start; see anchor resolution above).
            s_iso, e_iso = resolve_period_dates(expr, anchor, lease_end)
        else:
            # Explicit dates: the period start prefers a rent-payment-start
            # phrase, the end uses the 至/止 lease terminus.
            s_iso = to_iso(select_date(start_c, "start", RENT_START_KEYWORDS)) or to_iso(anchor)
            e_iso = to_iso(select_date(end_c, "end")) or to_iso(lease_end)

        if start_f is not None:
            _set_value(start_f, s_iso)
        if end_f is not None:
            _set_value(end_f, e_iso)


# Schema-key → handler.  Keys match outputSchema.required[0] in parsecalls.json.
_HANDLERS = {
    "LeaseTerms": _enrich_object,
    "PaymentTerms": _enrich_period_list,
    "CostSeparation": _enrich_period_list,
}


def enrich_payload_dates(payload: Any) -> Any:
    """
    Apply the ground-truth date tools to a parsed parse-step payload, in place.

    `payload` is the inner object the model returned, e.g. {"LeaseTerms": {...}}
    or {"PaymentTerms": [...]}.  Unknown / dateless topics pass through
    untouched.  Returns the same object for convenience.
    """
    if not isinstance(payload, dict):
        return payload
    for key, handler in _HANDLERS.items():
        if key in payload:
            try:
                handler(payload[key])
            except Exception:  # never let date enrichment break the pipeline
                pass
    return payload


def compute_date_tool_result(extracted: Any) -> Optional[Dict[str, Any]]:
    """
    Run the deterministic date tools over an *extract-step* payload and return a
    structured report — WITHOUT mutating `extracted`.

    This is the `compute_ground_truth_dates` tool whose output is handed to the
    parse step as a tool result (see utils/extraction.py).  Computing on the rich
    extract splices — rather than the lossy parse output — means the rent-start
    anchor (租金給付始期) and 第N年 derivations are resolved from verbatim contract
    text, so the parse model never has to do ROC arithmetic itself.

    Returns a dict keyed by topic, e.g.::

        {"PaymentTerms": {"anchor": "2023-10-25", "leaseStart": "2023-08-01",
                          "leaseEnd": "2028-07-31", "rentFreePeriod": true,
                          "periods": [{"index": 0,
                                       "periodStartDate": "2023-10-25T00:00:00.000Z",
                                       "periodEndDate":   "2028-07-31T00:00:00.000Z"}]}}

    or None when there is nothing date-bearing to report.
    """
    if not isinstance(extracted, dict):
        return None

    # Reuse the (tested) enrichment logic on a throwaway copy, then harvest the
    # computed `value`s into a clean report.
    work = copy.deepcopy(extracted)
    enrich_payload_dates(work)

    report: Dict[str, Any] = {}
    for key in _HANDLERS:
        if key not in work:
            continue
        section = work[key]

        if isinstance(section, list):
            anchor, lease_start, lease_end = _scan_lease_facts(section)
            report[key] = {
                "anchor": to_iso(anchor),
                "leaseStart": to_iso(lease_start),
                "leaseEnd": to_iso(lease_end),
                "rentFreePeriod": bool(
                    anchor and lease_start and anchor != lease_start
                ),
                "periods": [
                    {
                        "index": i,
                        "periodStartDate": (it.get("periodStartDate") or {}).get("value"),
                        "periodEndDate": (it.get("periodEndDate") or {}).get("value"),
                    }
                    for i, it in enumerate(section)
                ],
            }
        elif isinstance(section, dict):
            report[key] = {
                "periodStartDate": (section.get("periodStartDate") or {}).get("value"),
                "periodEndDate": (section.get("periodEndDate") or {}).get("value"),
            }

    return report or None


# ---------------------------------------------------------------------------
# Self-test / ground-truth fixtures  (python -m utils.datetools)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    checks = []

    def check(label, got, want):
        ok = got == want
        checks.append(ok)
        flag = "ok " if ok else "FAIL"
        print(f"[{flag}] {label}\n        got={got!r}\n        want={want!r}")

    # --- Tool 1: ROC → ISO -------------------------------------------------
    check("民國110年08月01日", roc_to_iso("民國(下同)110年08月01日"),
          "2021-08-01T00:00:00.000Z")
    check("民國122年03月31日 (the prompt's buggy case)",
          roc_to_iso("122年03月 31"), "2033-03-31T00:00:00.000Z")
    check("【112】年【8】月【1】日 bracketed", roc_to_iso("民國【112】年【8】月【1】日起"),
          "2023-08-01T00:00:00.000Z")
    check("4-digit Gregorian stays put", roc_to_iso("租賃期間自2024年1月1日起"),
          "2024-01-01T00:00:00.000Z")
    check("range → last date", roc_to_iso("110年08月01日起至122年03月31日止", which="last"),
          "2033-03-31T00:00:00.000Z")

    # --- span + rent-start extraction -------------------------------------
    span = extract_lease_span("民國(下同)110年08月01日起至122年03月31日止")
    check("extract_lease_span", span, (date(2021, 8, 1), date(2033, 3, 31)))
    check("find_rent_start", find_rent_start("租金給付始期為民國112年10月25日"),
          date(2023, 10, 25))

    # --- Tool 2: relative periods (anchor = 2021-08-01) -------------------
    anchor, end = date(2021, 8, 1), date(2033, 3, 31)
    check("第一年至第三年", resolve_period_dates("第一年至第三年", anchor, end),
          ("2021-08-01T00:00:00.000Z", "2024-07-31T00:00:00.000Z"))
    check("第四年至第六年 (prompt had 2027-07-30, wrong)",
          resolve_period_dates("第四年至第六年", anchor, end),
          ("2024-08-01T00:00:00.000Z", "2027-07-31T00:00:00.000Z"))
    check("第七年起至租約屆滿", resolve_period_dates("第七年起至租約屆滿", anchor, end),
          ("2027-08-01T00:00:00.000Z", "2033-03-31T00:00:00.000Z"))

    # --- rent-free anchor case (anchor = 2023-04-01) ----------------------
    a2, e2 = date(2023, 4, 1), date(2029, 12, 31)
    check("rent-free 第一年至第三年", resolve_period_dates("第一年至第三年", a2, e2),
          ("2023-04-01T00:00:00.000Z", "2026-03-31T00:00:00.000Z"))

    # --- keyword-priority selection (two dates in one splice) -------------
    mixed = ("本契約之租賃期間自民國(以下同)【112】年【8】月【1】日起至【117】年【7】月【31】日止。"
             "租金給付始期為民國【112】年【10】月【25】日")
    check("select start by rent keyword", select_date(mixed, "start", RENT_START_KEYWORDS),
          date(2023, 10, 25))
    check("select start structural (no kw) → lease 起", select_date(mixed, "start"),
          date(2023, 8, 1))
    check("select end structural → 止", select_date(mixed, "end"),
          date(2028, 7, 31))

    # --- enrichment over a payload ----------------------------------------
    payload = {
        "PaymentTerms": [
            {"periodStartDate": {"value": None, "content": "110年08月01日起至122年03月31日止 第一年至第三年", "confidence": 50},
             "periodEndDate": {"value": None, "content": "110年08月01日起至122年03月31日止 第一年至第三年", "confidence": 50}},
            {"periodStartDate": {"value": None, "content": "第七年起", "confidence": 50},
             "periodEndDate": {"value": None, "content": "租約屆滿", "confidence": 50}},
        ]
    }
    enrich_payload_dates(payload)
    pt = payload["PaymentTerms"]
    check("enrich period 1 start", pt[0]["periodStartDate"]["value"], "2021-08-01T00:00:00.000Z")
    check("enrich period 1 end", pt[0]["periodEndDate"]["value"], "2024-07-31T00:00:00.000Z")
    check("enrich period 2 end → lease end", pt[1]["periodEndDate"]["value"], "2033-03-31T00:00:00.000Z")

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
