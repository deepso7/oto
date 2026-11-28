"""ASR-based metrics. The text metrics are pure Python; transcription needs an optional extra:
`asr` (faster-whisper) or `parakeet` (NVIDIA Parakeet TDT 0.6B v2 through onnx-asr, the
recogniser used by the Krisp Voice Isolation Benchmark rows in docs/plans/0004).

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
    # For each reference word: True if it aligned as an exact match.
    matched: list[bool]

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
    matched = [False] * n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] == hyp[j - 1]:
                matched[i - 1] = True
            else:
                s += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            de += 1
            deleted[i - 1] = True
            i -= 1
        else:
            ins += 1
            j -= 1
    return Alignment(s, de, ins, n, deleted, matched)


def wer(ref: str, hyp: str) -> float:
    return align(normalize(ref), normalize(hyp)).wer


def first_word_deleted(ref: str, hyp: str) -> bool:
    """True when the reference's first word was deleted or substituted in alignment (leading
    filler words before a preserved first word count as WER insertions, not missing speech)."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return False
    return not align(r, h).matched[0]


def words_per_minute(text: str, seconds: float) -> float:
    return len(normalize(text)) / (seconds / 60.0) if seconds > 0 else 0.0


def wrong_speaker(hyp: str, target: str, interferer: str) -> bool:
    """The output followed the interfering talker rather than the target."""
    return wer(interferer, hyp) < wer(target, hyp)


Word = tuple[str, float, float]  # (word, start_s, end_s)


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

    def _run(self, wav_path, words: bool):  # pragma: no cover - needs model weights
        c = self.config
        segments, _ = self._model.transcribe(str(wav_path), language=c["language"], beam_size=c["beam_size"], temperature=c["temperature"], vad_filter=c["vad_filter"], condition_on_previous_text=c["condition_on_previous_text"], word_timestamps=words)
        return list(segments)

    def __call__(self, wav_path: str) -> str:  # pragma: no cover - needs model weights
        return " ".join(s.text.strip() for s in self._run(wav_path, False))

    def transcribe_words(self, wav_path: str) -> tuple[str, list[Word]]:  # pragma: no cover - needs model weights
        segs = self._run(wav_path, True)
        return " ".join(s.text.strip() for s in segs), [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in (s.words or [])]


def words_from_tokens(tokens, timestamps, frame_s: float = 0.08) -> list[Word]:
    """Group subword tokens into timed words. A token starting with a space (onnx-asr) or
    `▁` (SentencePiece) starts a word; a word ends one encoder frame (`frame_s`, 80 ms for
    FastConformer) after its last token's start."""
    words: list[list] = []
    for tok, t in zip(tokens or [], timestamps or []):
        starts = tok[:1] in (" ", "\u2581")
        piece = tok.lstrip(" \u2581")
        if starts or not words:
            if piece or starts:
                words.append([piece, float(t), float(t) + frame_s])
        else:
            words[-1][0] += piece
            words[-1][2] = float(t) + frame_s
    return [(w, round(a, 4), round(b, 4)) for w, a, b in words if w]


def default_parakeet_providers(available) -> list[str]:
    """Skip macOS CoreMLExecutionProvider (splits FastConformer into ~291 partitions and leaks
    Metal/CoreML contexts across variable-length clips until OOM) and AzureExecutionProvider."""
    skip = {"CoreMLExecutionProvider", "AzureExecutionProvider"}
    return [p for p in available if p not in skip] or ["CPUExecutionProvider"]


def parakeet_result_text(result) -> tuple[str, list[Word]]:
    """Decode an onnx-asr timestamped result; reconstruct text from grouped tokens because
    onnx-asr's `result.text` drops spaces before currency symbols (£/$/€)."""
    words = words_from_tokens(result.tokens, result.timestamps)
    text = " ".join(w for w, _, _ in words) if words else (result.text or "").strip()
    return text, words


class ParakeetTranscriber:
    """NVIDIA Parakeet TDT 0.6B v2 (CC-BY-4.0 weights) through `onnx-asr` (MIT; numpy +
    onnxruntime, no torch or NeMo), with token timestamps for leak scoring. Greedy TDT
    decoding, whole file in one pass (Krisp clips are at most ~150 s). `quantization="int8"`
    loads the ~0.65 GB int8 export instead of the ~2.4 GB fp32 one; the Krisp reproduction uses
    fp32, the default. Weights download from Hugging Face into its cache on first use."""

    DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v2"
    RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

    def __init__(self, model: str | None = None, quantization: str | None = None, providers: list[str] | None = None):
        try:
            import numpy as np
            import onnx_asr
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("Parakeet needs the `parakeet` extra: uv sync --project tools/eval --extra parakeet (onnx-asr, onnxruntime)") from e
        model = model or self.DEFAULT_MODEL
        self._np = np
        kw = {"quantization": quantization} if quantization else {}
        if providers is None:  # pragma: no cover - hardware dependent
            import onnxruntime as ort

            providers = default_parakeet_providers(ort.get_available_providers())
        kw["providers"] = providers  # pragma: no cover - hardware dependent
        self._model = onnx_asr.load_model(model, **kw).with_timestamps()  # pragma: no cover - needs weights
        self.config = {"engine": "onnx-asr", "model": model, "quantization": quantization or "fp32", "decoding": "greedy-tdt", "timestamps": True, "vad": False}  # pragma: no cover

    def _load(self, wav_path):  # pragma: no cover - needs model weights
        from wavio import read_wav

        a = read_wav(wav_path)
        x, rate = a.mono, a.rate
        if rate not in self.RATES:
            import metrics

            x, rate = metrics.resample_linear(x, rate, 16000), 16000
        return self._np.asarray(x, dtype=self._np.float32), rate

    def transcribe_words(self, wav_path: str) -> tuple[str, list[Word]]:  # pragma: no cover - needs model weights
        x, rate = self._load(wav_path)
        return parakeet_result_text(self._model.recognize(x, sample_rate=rate))

    def __call__(self, wav_path: str) -> str:  # pragma: no cover - needs model weights
        return self.transcribe_words(wav_path)[0]


ASR_BACKENDS = ("faster-whisper", "parakeet")


def make_transcriber(backend: str = "faster-whisper", model: str | None = None, quantization: str | None = None):
    """One transcriber, chosen by name; `model` None picks the backend's default."""
    if backend == "faster-whisper":
        return Transcriber(model or "small.en")
    if backend == "parakeet":
        return ParakeetTranscriber(model, quantization)
    raise ValueError(f"unknown ASR backend {backend!r}; use {', '.join(ASR_BACKENDS)}")


def transcribe(transcriber, wav_path, want_words: bool = False) -> tuple[str, list[Word] | None]:
    """Text, plus timed words when wanted and the transcriber can produce them."""
    if want_words and hasattr(transcriber, "transcribe_words"):
        text, words = transcriber.transcribe_words(str(wav_path))
        return text, list(words)
    return transcriber(str(wav_path)), None
