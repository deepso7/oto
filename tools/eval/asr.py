"""ASR-based metrics. The text metrics are pure Python; transcription needs the optional
`faster-whisper` extra.

Same-ASR head-to-head: every candidate (and the unprocessed input) is transcribed by one
Transcriber instance with one fixed configuration, and only differences between candidates
are reported. Absolute WER from different ASR systems is never compared.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_PUNCT = re.compile(r"[^\w\s']")


def normalize(text: str) -> list[str]:
    """Lower-case words without punctuation; apostrophes kept inside words."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _PUNCT.sub(" ", t)
    return [w.strip("'") for w in t.split() if w.strip("'")]


@dataclass
class Alignment:
    substitutions: int
    deletions: int
    insertions: int
    ref_len: int
    # For each reference word: True if it was deleted.
    deleted: list[bool]

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.ref_len == 0:
            return 0.0 if self.insertions == 0 else float(self.insertions)
        return self.errors / self.ref_len


def align(ref: list[str], hyp: list[str]) -> Alignment:
    """Levenshtein alignment of word lists. Ties prefer substitution over deletion, so a
    deletion is only reported when it is needed."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j = n, m
    s = de = ins = 0
    deleted = [False] * n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            de += 1
            deleted[i - 1] = True
            i -= 1
        else:
            ins += 1
            j -= 1
    return Alignment(s, de, ins, n, deleted)


def wer(ref: str, hyp: str) -> float:
    return align(normalize(ref), normalize(hyp)).wer


def first_word_deleted(ref: str, hyp: str) -> bool:
    """True when the reference's first word is missing from the hypothesis (deleted or
    replaced): the clipped-onset failure a denoiser's gate causes."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return False
    a = align(r, h)
    if a.deleted[0]:
        return True
    return r[0] not in h[:2]


def words_per_minute(text: str, seconds: float) -> float:
    return len(normalize(text)) / (seconds / 60.0) if seconds > 0 else 0.0


def wrong_speaker(hyp: str, target: str, interferer: str) -> bool:
    """The output followed the interfering talker rather than the target."""
    return wer(interferer, hyp) < wer(target, hyp)


class Transcriber:
    """faster-whisper with one fixed configuration (recorded in the report). VAD is off on
    purpose: a VAD in front of the ASR would hide the hallucinations we want to count."""

    def __init__(self, model: str = "small.en", device: str = "cpu", compute_type: str = "int8", language: str = "en"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("ASR needs the `asr` extra: uv sync --extra asr (faster-whisper)") from e
        self.config = {"engine": "faster-whisper", "model": model, "device": device, "compute_type": compute_type, "language": language, "beam_size": 5, "temperature": 0.0, "vad_filter": False, "condition_on_previous_text": False}
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def __call__(self, wav_path: str) -> str:  # pragma: no cover - needs model weights
        c = self.config
        segments, _ = self._model.transcribe(str(wav_path), language=c["language"], beam_size=c["beam_size"], temperature=c["temperature"], vad_filter=c["vad_filter"], condition_on_previous_text=c["condition_on_previous_text"])
        return " ".join(s.text.strip() for s in segments)
