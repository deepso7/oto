"""Krisp Voice Isolation Benchmark scoring, stdlib only.

The benchmark (HF `Krisp-AI/VoiceIsolation-Benchmark-Dataset`, CC BY-NC 4.0, evaluation only)
is 265 real 32 kHz recordings by 47 speakers in three scenarios (`work`, `call_center`,
`phone_calls`). Every file has a verbatim transcript of the primary speaker and time-stamped
segments labelled `primary`, `mix` (both talk), `secondary` (only the background talker) and
`noise` (no speech). There is no clean reference, so scoring is ASR-based:

- **Text normalisation** (Krisp's published methodology): four regex rules that split IDs and
  digit strings, NVIDIA NeMo text normalisation (numbers to words), then the jiwer transform
  (lower case, common English contractions, punctuation to spaces, filler words removed).
  NeMo TN is used when `nemo_text_processing` is installed; otherwise a stdlib verbaliser
  covers cardinals, ordinals, years, decimals, times, currency and percentages.
- **WER** is pooled at corpus level: sum(S + D + I) / sum(reference words).
- **Missed caller words** = sum(S + D) / sum(reference words): caller words lost or overwritten.
- **Leak** = ASR words whose midpoint falls inside a `secondary` segment (each shrunk by a
  0.25 s guard, segments of 0.5 s or less skipped), per minute of those segments. The correct
  transcript there is empty. The same rule over `noise` segments gives hallucinated words/min.
- **Confidence intervals**: speaker-clustered bootstrap (resample the 47 speakers with
  replacement, re-pool), and a paired version for "model minus reference" differences.

Leak, missed and the bootstrap follow the VoxLock scorer (github.com/arbazsiddiqui/voxlock,
Apache-2.0, `bench/krisp_bench.py` and `bench/publish_table.py`), which produced the
third-party rows quoted in docs/plans/0004 (raw 23.21% / 66.48, Krisp VI 2.5 Default
5.30% / 1.52). Per-clip rows use that scorer's field names, so its published
`krisp.clips.jsonl` can be read back and re-pooled.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

SCENARIOS = ("work", "call_center", "phone_calls")
SEGMENT_LABELS = ("primary", "mix", "secondary", "noise")
GUARD_S = 0.25
MIN_SEGMENT_S = 0.5
FILLERS = frozenset(["um", "uh", "umm", "uhh", "hmm", "mm", "mhm", "erm"])
COUNT_KEYS = ("H", "S", "D", "I", "leak", "bgmin", "noise_words", "noise_min")

# Published rows (Parakeet TDT 0.6B v2, Krisp's normalisation, pooled over all 265 files),
# from VoxLock's TABLE_krisp.md. `missed` is caller words missed (%), `leak` words/min.
PUBLISHED = {
    "raw": {"wer": 23.21, "missed": 4.28, "leak": 66.48},
    "vi_2_5_default": {"wer": 5.30, "missed": 4.21, "leak": 1.52},
    "vi_2_5_balanced": {"wer": 5.32, "missed": 4.06, "leak": 2.02},
    "vi_2_5_hd": {"wer": 5.79, "missed": 4.37, "leak": 2.74},
    "vi_2_5_lite": {"wer": 6.21, "missed": 4.26, "leak": 4.52},
    "dfn3": {"wer": 13.30, "missed": 6.46, "leak": 22.77},
}
REPRODUCE_TOLERANCE_WER = 0.2  # percentage points (plan 0004, Phase 0 "done when")


# ------------------------------------------------------------------------------------------
# Stage 1: Krisp's regex rules (ID and digit-string splitting before NeMo TN)

_GENERAL_ID_RE = re.compile(r"\b([A-Z]{1,4})(?:-?(\d+)|[ ](\d{3,}))(?:[ ]?(\d)([A-Z]{2}))?([A-Z]{0,2})\b")
_DIGIT_FIRST_RE = re.compile(r"\b(\d{4,})-?([A-Za-z])\b")
_UNITS = r"(?:dollars?|pounds?|euros?|percent|miles?|days?|months?|hours?|minutes?|nights?|years?|weeks?|points?|gigabytes?|megabits?)"
_STANDALONE_4_RE = re.compile(rf"(?<![$£€\d,])(?<!\d)\b(\d{{4}})\b(?!\d)(?!\s+{_UNITS})")
_DIGIT_GROUPS_RE = re.compile(r"\b\d{1,16}(?:\s+\d{1,16}){1,}\b")


def split_general_id(text: str) -> str:
    """`AB-1234` / `AB 1234` / `SW1A 1AA`-like IDs are read character by character."""
    def repl(m):
        g = m.groups()
        parts = list(g[0]) + list(g[1] or g[2] or "")
        if g[3]:
            parts += list(g[3]) + list(g[4])
        parts += list(g[5] or "")
        return " ".join(parts)

    return _GENERAL_ID_RE.sub(repl, text)


def split_digit_first(text: str) -> str:
    """`12345B` -> `1 2 3 4 5 B` (but `1990s` stays)."""
    def repl(m):
        if m.group(2).lower() == "s":
            return m.group(0)
        return " ".join(list(m.group(1)) + [m.group(2)])

    return _DIGIT_FIRST_RE.sub(repl, text)


def split_standalone_4digits(text: str) -> str:
    """A lone 4-digit number outside 1000–2099 (not money, not followed by a unit) is a code: digits."""
    def repl(m):
        return m.group(0) if 1000 <= int(m.group(1)) <= 2099 else " ".join(m.group(1))

    return _STANDALONE_4_RE.sub(repl, text)


def split_adjacent_digit_groups(text: str) -> str:
    """`555 123 4567` -> one digit per word."""
    return _DIGIT_GROUPS_RE.sub(lambda m: " ".join(m.group(0).replace(" ", "")), text)


def krisp_regex_rules(text: str) -> str:
    return split_adjacent_digit_groups(split_standalone_4digits(split_digit_first(split_general_id(text))))


# ------------------------------------------------------------------------------------------
# Stage 2: text normalisation (NeMo when installed, else a stdlib verbaliser)

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_ORD_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def number_words(n: int) -> str:
    """Cardinal in words, without "and": 123 -> one hundred twenty three."""
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " " + number_words(r))
    for value, name in _SCALES:
        if n >= value:
            q, r = divmod(n, value)
            return number_words(q) + " " + name + ("" if r == 0 else " " + number_words(r))
    raise AssertionError(n)  # pragma: no cover


def ordinal_words(n: int) -> str:
    words = number_words(n).split()
    last = words[-1]
    if last in _ORD_IRREGULAR:
        words[-1] = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year_words(n: int) -> str:
    """1969 -> nineteen sixty nine, 2024 -> twenty twenty four, 2005 -> two thousand five."""
    hi, lo = divmod(n, 100)
    if 2000 <= n <= 2009 or n % 1000 == 0:
        return number_words(n)
    if lo == 0:
        return number_words(hi) + " hundred"
    return number_words(hi) + (" oh " + _ONES[lo] if lo < 10 else " " + number_words(lo))


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _digits_words(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"), "€": ("euro", "euros", "cent", "cents")}
_NUM = r"\d{1,3}(?:,\d{3})+|\d+"
_RE_MONEY = re.compile(rf"([$£€])\s?({_NUM})(?:\.(\d{{1,2}}))?(?:\s?(thousand|million|billion|trillion)\b)?", re.I)
_RE_PERCENT = re.compile(rf"({_NUM})(?:\.(\d+))?\s?%")
_RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b(?:\s?([ap])\.?\s?m\b\.?)?", re.I)
_RE_AMPM = re.compile(r"\b(\d{1,2})\s?([ap])\.?\s?m\b\.?", re.I)
_RE_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.I)
_RE_DECADE = re.compile(r"\b(\d{2,4})'?s\b")
_RE_DECIMAL = re.compile(rf"(?<![\d.])({_NUM})\.(\d+)\b")
_RE_YEAR = re.compile(r"(?<![\d,.])\b(1[0-9]{3}|20[0-9]{2})\b(?!\d|,\d)")
_RE_CARDINAL = re.compile(rf"(?<![\d.])\b({_NUM})\b(?![\d])")


def stdlib_verbalize(text: str) -> str:
    """A small, deterministic stand-in for NeMo English text normalisation (cased input)."""

    def money(m):
        sym, whole, frac, scale = m.group(1), _int(m.group(2)), m.group(3), m.group(4)
        one, many, sub1, subn = _CURRENCY[sym]
        if scale:
            return f"{number_words(whole)} {scale.lower()} {many}"
        out = f"{number_words(whole)} {one if whole == 1 else many}"
        if frac and int(frac.ljust(2, '0')):
            cents = int(frac.ljust(2, "0"))
            out += f" {number_words(cents)} {sub1 if cents == 1 else subn}"
        return out

    def percent(m):
        w = number_words(_int(m.group(1)))
        if m.group(2):
            w += " point " + _digits_words(m.group(2))
        return w + " percent"

    def ampm(letter):
        return f" {letter.lower()} m" if letter else ""

    def clock(m):
        h, mm, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        if mm == 0:
            words = number_words(h) + ("" if ap else " o'clock")
        elif mm < 10:
            words = number_words(h) + " oh " + _ONES[mm]
        else:
            words = number_words(h) + " " + number_words(mm)
        return words + ampm(ap)

    def decade(m):
        n = int(m.group(1))
        base = year_words(n) if n >= 1000 else number_words(n)
        last = base.split()
        last[-1] = last[-1][:-1] + "ies" if last[-1].endswith("y") else last[-1] + "s"
        return " ".join(last)

    t = text.replace("&", " and ")
    t = _RE_MONEY.sub(money, t)
    t = _RE_PERCENT.sub(percent, t)
    t = _RE_TIME.sub(clock, t)
    t = _RE_AMPM.sub(lambda m: number_words(int(m.group(1))) + ampm(m.group(2)), t)
    t = _RE_ORDINAL.sub(lambda m: ordinal_words(int(m.group(1))), t)
    t = _RE_DECADE.sub(decade, t)
    t = _RE_DECIMAL.sub(lambda m: number_words(_int(m.group(1))) + " point " + _digits_words(m.group(2)), t)
    t = _RE_YEAR.sub(lambda m: year_words(int(m.group(1))), t)
    t = _RE_CARDINAL.sub(lambda m: number_words(_int(m.group(1))), t)
    return t


# ------------------------------------------------------------------------------------------
# Stage 3: jiwer's transform (ToLowerCase, ExpandCommonEnglishContractions,
# SubstituteRegexes([^\w\s] -> " "), RemoveSpecificWords(fillers), RemoveMultipleSpaces, Strip)

_CONTRACTIONS = [(r"won't", "will not"), (r"can't", "can not"), (r"let's", "let us"), (r"n't", " not"), (r"'re", " are"), (r"'s", " is"), (r"'d", " would"), (r"'ll", " will"), (r"'t", " not"), (r"'ve", " have"), (r"'m", " am")]
_CONTRACTIONS_RE = [(re.compile(a), b) for a, b in _CONTRACTIONS]
_NON_WORD = re.compile(r"[^\w\s]")


def jiwer_words(text: str) -> list[str]:
    t = text.lower()
    for rx, rep in _CONTRACTIONS_RE:
        t = rx.sub(rep, t)
    t = _NON_WORD.sub(" ", t)
    return [w for w in t.split() if w not in FILLERS]


def leak_tokens(word: str) -> list[str]:
    """Krisp/VoxLock leak counting: lower case, punctuation to spaces, fillers dropped (no
    contraction expansion, so `don't` counts as two tokens, as in the published scorer)."""
    return [w for w in _NON_WORD.sub(" ", word.lower()).split() if w not in FILLERS]


class KrispNormalizer:
    """Krisp's three-stage normalisation. `backend`: "auto" (NeMo if importable), "nemo" or
    "stdlib". Results are memoised: NeMo TN is slow and the corpus has few distinct texts."""

    def __init__(self, backend: str = "auto", cache_dir: str | None = None):
        self._nemo = None
        if backend in ("auto", "nemo"):
            try:
                from nemo_text_processing.text_normalization.normalize import Normalizer
            except ImportError:
                if backend == "nemo":
                    raise RuntimeError("NeMo text normalisation needs `nemo_text_processing` (extra: nemo-tn)") from None
            else:  # pragma: no cover - optional dependency
                kw = {"cache_dir": cache_dir} if cache_dir else {}
                self._nemo = Normalizer(input_case="cased", lang="en", **kw)
        self.backend = "nemo" if self._nemo is not None else "stdlib"
        self._memo: dict[str, list[str]] = {}

    def stage2(self, text: str) -> str:
        t = krisp_regex_rules(text)
        if self._nemo is not None:  # pragma: no cover - optional dependency
            return self._nemo.normalize(t)
        return stdlib_verbalize(t)

    def __call__(self, text: str | None) -> list[str]:
        text = text or ""
        if text not in self._memo:
            self._memo[text] = jiwer_words(self.stage2(text))
        return self._memo[text]


# ------------------------------------------------------------------------------------------
# Segments and per-clip counts


def validate_segments(segments) -> list[dict]:
    out = []
    for s in segments or []:
        label = s.get("label")
        if label not in SEGMENT_LABELS:
            raise ValueError(f"unknown Krisp segment label {label!r}; expected one of {', '.join(SEGMENT_LABELS)}")
        a, b = float(s["start"]), float(s["end"])
        if not (math.isfinite(a) and math.isfinite(b)) or b < a:
            raise ValueError(f"bad segment times {a}..{b}")
        out.append({"start": a, "end": b, "label": label, "text": s.get("text")})
    return out


def regions(segments, label: str = "secondary", guard: float = GUARD_S, min_len: float = MIN_SEGMENT_S) -> list[tuple[float, float]]:
    """`label` segments longer than `min_len`, each shrunk by `guard` seconds at both ends."""
    return [(s["start"] + guard, s["end"] - guard) for s in segments if s["label"] == label and s["end"] - s["start"] > min_len]


def segment_minutes(segments, label: str) -> float:
    return sum(s["end"] - s["start"] for s in segments if s["label"] == label) / 60.0


def words_in_regions(words, regs, offset: float = 0.0) -> int:
    """Tokens of timed ASR words `(word, start, end)` whose midpoint (minus `offset`, the
    processing delay) falls in one of `regs`."""
    n = 0
    for w, ws, we in words:
        mid = (ws + we) / 2 - offset
        if any(b <= mid <= e for b, e in regs):
            n += len(leak_tokens(w))
    return n


@dataclass
class Counts:
    hits: int
    substitutions: int
    deletions: int
    insertions: int


def word_counts(ref: list[str], hyp: list[str]) -> Counts:
    """H/S/D/I of a minimum-edit alignment (asr.align, same totals as jiwer)."""
    import asr

    a = asr.align(ref, hyp)
    return Counts(a.ref_len - a.substitutions - a.deletions, a.substitutions, a.deletions, a.insertions)


def clip_row(clip_id: str, scenario: str, speaker: str, ref_text: str, hyp_text: str, words, segments, normalizer: KrispNormalizer, offset: float = 0.0) -> dict:
    """One per-clip row in the VoxLock `clips.jsonl` format, plus noise-segment hallucinations.
    `words` are timed ASR words (needed for leak); `offset` is the output's delay in seconds."""
    segs = validate_segments(segments)
    c = word_counts(normalizer(ref_text), normalizer(hyp_text))
    sec, noi = regions(segs, "secondary"), regions(segs, "noise")
    return {
        "i": clip_id, "scen": scenario, "caller": speaker,
        "H": c.hits, "S": c.substitutions, "D": c.deletions, "I": c.insertions,
        "leak": words_in_regions(words or [], sec, offset), "bgmin": round(sum(e - b for b, e in sec) / 60, 6),
        "noise_words": words_in_regions(words or [], noi, offset), "noise_min": round(sum(e - b for b, e in noi) / 60, 6),
    }


# ------------------------------------------------------------------------------------------
# Pooling and the speaker-clustered bootstrap


def _totals(rows) -> dict:
    return {k: math.fsum(r.get(k, 0) for r in rows) for k in COUNT_KEYS}


def pooled_from_totals(t: dict) -> dict:
    n = t["H"] + t["S"] + t["D"]
    nan = float("nan")
    return {
        "wer": 100 * (t["S"] + t["D"] + t["I"]) / n if n else nan,
        "missed": 100 * (t["S"] + t["D"]) / n if n else nan,
        "del": 100 * t["D"] / n if n else nan,
        "ins": 100 * t["I"] / n if n else nan,
        "sub": 100 * t["S"] / n if n else nan,
        "leak": t["leak"] / t["bgmin"] if t["bgmin"] > 0 else nan,
        "noise_wpm": t["noise_words"] / t["noise_min"] if t.get("noise_min", 0) > 0 else nan,
        "ref_words": n,
    }


def pooled(rows) -> dict:
    """Corpus-level WER, missed caller words, del/ins/sub (all %), leak and noise words/min."""
    return pooled_from_totals(_totals(rows))


STAT_KEYS = ("wer", "missed", "del", "ins", "sub", "leak", "noise_wpm")


def percentile(values, q: float) -> float:
    """numpy's default (linear) percentile, NaNs ignored."""
    v = sorted(x for x in values if not math.isnan(x))
    if not v:
        return float("nan")
    pos = (len(v) - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _by_speaker(rows, key: str = "caller") -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key) or r.get("i")), []).append(r)
    return groups


def speaker_bootstrap(rows, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """95% interval of every pooled statistic, resampling speakers with replacement."""
    groups = _by_speaker(rows)
    names = sorted(groups)
    sums = [_totals(groups[s]) for s in names]
    rng = random.Random(seed)
    draws = {k: [] for k in STAT_KEYS}
    for _ in range(n_boot):
        t = dict.fromkeys(COUNT_KEYS, 0.0)
        for j in (rng.randrange(len(names)) for _ in names):
            for k in COUNT_KEYS:
                t[k] += sums[j][k]
        p = pooled_from_totals(t)
        for k in STAT_KEYS:
            draws[k].append(p[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (percentile(v, lo), percentile(v, hi)) for k, v in draws.items()} | {"speakers": len(names), "n_boot": n_boot}


def _check_unique_ids(rows) -> None:
    seen: set[str] = set()
    for r in rows:
        cid = r.get("i")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"missing or empty clip id {cid!r}")
        if cid in seen:
            raise ValueError(f"duplicate clip id {cid!r}")
        seen.add(cid)


def paired_bootstrap(rows, ref_rows, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Difference `rows` minus `ref_rows` (same clips, matched by `i`) with a paired
    speaker-clustered bootstrap. Returns `{stat: (point, lo, hi)}`; a difference is
    significant when the interval excludes zero."""
    _check_unique_ids(rows)
    _check_unique_ids(ref_rows)
    ref = {r["i"]: r for r in ref_rows}
    pairs = [(r, ref[r["i"]]) for r in rows if r["i"] in ref]
    if not pairs:
        raise ValueError("no clips in common")
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for a, b in pairs:
        groups.setdefault(str(a.get("caller") or a["i"]), []).append((a, b))
    names = sorted(groups)
    sa = [_totals([a for a, _ in groups[s]]) for s in names]
    sb = [_totals([b for _, b in groups[s]]) for s in names]
    pa, pb = pooled([a for a, _ in pairs]), pooled([b for _, b in pairs])
    rng = random.Random(seed)
    draws = {k: [] for k in STAT_KEYS}
    for _ in range(n_boot):
        ta, tb = dict.fromkeys(COUNT_KEYS, 0.0), dict.fromkeys(COUNT_KEYS, 0.0)
        for j in (rng.randrange(len(names)) for _ in names):
            for k in COUNT_KEYS:
                ta[k] += sa[j][k]
                tb[k] += sb[j][k]
        qa, qb = pooled_from_totals(ta), pooled_from_totals(tb)
        for k in STAT_KEYS:
            draws[k].append(qa[k] - qb[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {k: (pa[k] - pb[k], percentile(v, lo), percentile(v, hi)) for k, v in draws.items()}
    out["clips"] = len(pairs)
    out["speakers"] = len(names)
    return out


# Metric names used by eval.py and gates.toml (`[oto1.isolate]`).
METRIC_NAMES = {"wer": "krisp_wer_pct", "missed": "missed_caller_pct", "del": "deleted_pct", "ins": "inserted_pct", "sub": "substituted_pct", "leak": "secondary_leaked_words_per_min", "noise_wpm": "noise_hallucinated_words_per_min"}


def summary_metrics(rows, prefix: str = "", n_boot: int = 2000, seed: int = 0) -> dict:
    """Flat metrics for a set of clip rows: pooled values, speaker-bootstrap CIs
    (`*_ci_lo` / `*_ci_hi`), per-scenario pooled WER and leak, and file/speaker counts."""
    if not rows:
        return {}
    _check_unique_ids(rows)
    out: dict = {}
    p = pooled(rows)
    ci = speaker_bootstrap(rows, n_boot, seed)
    for k, name in METRIC_NAMES.items():
        if not math.isnan(p[k]):
            out[prefix + name] = p[k]
            lo, hi = ci[k]
            if not math.isnan(lo):
                out[prefix + name + "_ci_lo"], out[prefix + name + "_ci_hi"] = lo, hi
    out[prefix + "krisp_files"] = len(rows)
    out[prefix + "krisp_speakers"] = ci["speakers"]
    for scen in SCENARIOS:
        sel = [r for r in rows if r.get("scen") == scen]
        if sel:
            ps = pooled(sel)
            out[f"{prefix}{scen}_krisp_wer_pct"] = ps["wer"]
            if not math.isnan(ps["leak"]):
                out[f"{prefix}{scen}_secondary_leaked_words_per_min"] = ps["leak"]
    return out


def delta_metrics(rows, ref_rows, ref_name: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """`<metric>_delta_vs_<ref_name>` (point, `_ci_lo`, `_ci_hi`): rows minus the reference,
    plus matched clip and speaker counts for the paired intersection."""
    d = paired_bootstrap(rows, ref_rows, n_boot, seed)
    out: dict = {
        f"krisp_matched_files_vs_{ref_name}": d["clips"],
        f"krisp_matched_speakers_vs_{ref_name}": d["speakers"],
    }
    for k, name in METRIC_NAMES.items():
        point, lo, hi = d[k]
        if math.isnan(point):
            continue
        base = f"{name}_delta_vs_{ref_name}"
        out[base] = point
        if not math.isnan(lo):
            out[base + "_ci_lo"], out[base + "_ci_hi"] = lo, hi
    return out


def cluster_mean_ci(values, groups, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of per-clip `values` and its bootstrap interval, resampling `groups` (speakers)
    with replacement. Used for paired per-clip differences such as DNSMOS(Oto) − DNSMOS(ref)."""
    by: dict[str, list[float]] = {}
    for v, g in zip(values, groups):
        if v is not None and math.isfinite(v):
            by.setdefault(str(g), []).append(v)
    if not by:
        return (float("nan"),) * 3
    names = sorted(by)
    sums = [(math.fsum(by[g]), len(by[g])) for g in names]
    point = math.fsum(s for s, _ in sums) / sum(n for _, n in sums)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        tot = cnt = 0.0
        for j in (rng.randrange(len(names)) for _ in names):
            tot += sums[j][0]
            cnt += sums[j][1]
        draws.append(tot / cnt)
    return point, percentile(draws, 100 * alpha / 2), percentile(draws, 100 * (1 - alpha / 2))
