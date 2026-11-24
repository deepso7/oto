"""Minimal stdlib-only WAV reader and writer.

Reads PCM 8/16/24/32-bit, IEEE float 32/64 and WAVE_FORMAT_EXTENSIBLE files. Writes PCM 16-bit
or float 32-bit. Samples are Python floats in [-1, 1] (float files may exceed that, on purpose:
clipping is something the harness measures).
"""

from __future__ import annotations

import array
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

PCM = 1
IEEE_FLOAT = 3
EXTENSIBLE = 0xFFFE


@dataclass
class Audio:
    rate: int
    channels: list[list[float]]

    @property
    def mono(self) -> list[float]:
        """Downmixed mono matching `oto_engine::pipeline::downmix`: average 2ch, ch0 for >2ch."""
        if not self.channels:
            return []
        if len(self.channels) == 2:
            return [(a + b) * 0.5 for a, b in zip(self.channels[0], self.channels[1])]
        return self.channels[0]

    @property
    def frames(self) -> int:
        return len(self.mono)


def _native(a: array.array) -> array.array:
    if sys.byteorder != "little":
        a.byteswap()
    return a


def read_wav(path: str | Path) -> Audio:
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos = 12
    fmt = None
    payload = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    if len(fmt) < 16:
        raise ValueError(f"{path}: truncated fmt chunk ({len(fmt)} bytes)")
    tag, nch, rate, _, block_align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if tag == EXTENSIBLE:
        if len(fmt) < 26:
            raise ValueError(f"{path}: truncated extensible fmt chunk ({len(fmt)} bytes)")
        tag = struct.unpack_from("<H", fmt, 24)[0]
    if nch < 1:
        raise ValueError(f"{path}: no channels")
    if rate < 1:
        raise ValueError(f"{path}: invalid sample rate {rate}")
    if (tag == IEEE_FLOAT and bits in (32, 64)) or (tag == PCM and bits in (8, 16, 24, 32)):
        width = bits // 8
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag} with {bits} bits")
    usable = len(payload) - len(payload) % (width * nch)
    payload = payload[:usable]
    if tag == IEEE_FLOAT and bits == 32:
        flat = list(_native(array.array("f", payload)))
    elif tag == IEEE_FLOAT and bits == 64:
        flat = list(_native(array.array("d", payload)))
    elif tag == PCM and bits == 16:
        flat = [s / 32768.0 for s in _native(array.array("h", payload))]
    elif tag == PCM and bits == 32:
        flat = [s / 2147483648.0 for s in _native(array.array("i", payload))]
    elif tag == PCM and bits == 8:
        flat = [(b - 128) / 128.0 for b in payload]
    elif tag == PCM and bits == 24:
        flat = []
        for i in range(0, len(payload), 3):
            v = payload[i] | (payload[i + 1] << 8) | (payload[i + 2] << 16)
            if v & 0x800000:
                v -= 1 << 24
            flat.append(v / 8388608.0)
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag} with {bits} bits")
    return Audio(rate, [flat[c::nch] for c in range(nch)])


def write_wav(path: str | Path, rate: int, samples, fmt: str = "pcm16") -> None:
    """Writes mono `samples`. `fmt` is "pcm16" (clipped and rounded) or "float32"."""
    if fmt == "pcm16":
        a = array.array("h", (max(-32768, min(32767, round(s * 32768.0))) for s in samples))
        tag, bits = PCM, 16
    elif fmt == "float32":
        a = array.array("f", samples)
        tag, bits = IEEE_FLOAT, 32
    else:
        raise ValueError(f"unknown WAV format {fmt!r}")
    raw = _native(a).tobytes()
    width = bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, tag, 1, rate, rate * width, width, bits)
    header += b"data" + struct.pack("<I", len(raw))
    Path(path).write_bytes(header + raw + (b"\0" if len(raw) & 1 else b""))
