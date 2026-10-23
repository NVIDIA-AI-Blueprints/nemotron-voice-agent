# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G.711 μ-law and A-law codecs as numpy lookup tables.

``audioop`` was removed in Python 3.13, so the tables are built here once, at
import time, from the ITU-T G.711 reference algorithm (the classic Sun
``g711.c``). Encoding indexes a 65 536-entry table by the unsigned view of each
16-bit sample; decoding indexes a 256-entry table by the code byte.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_SIGN_BIT = 0x80
_QUANT_MASK = 0x0F
_SEG_SHIFT = 4
_SEG_MASK = 0x70
_ULAW_BIAS = 0x84
_ULAW_CLIP = 8159
_SEG_UEND = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)
_SEG_AEND = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)

#: μ-law code for digital silence, as streamed continuously by tau2.
ULAW_SILENCE = 0xFF
#: A-law code for digital silence.
ALAW_SILENCE = 0xD5


def _segment(value: int, table: tuple[int, ...]) -> int:
    for index, end in enumerate(table):
        if value <= end:
            return index
    return len(table)


def _linear_to_ulaw(sample: int) -> int:
    value = sample >> 2
    if value < 0:
        value = -value
        mask = 0x7F
    else:
        mask = 0xFF
    value = min(value, _ULAW_CLIP) + (_ULAW_BIAS >> 2)
    seg = _segment(value, _SEG_UEND)
    if seg >= 8:
        return 0x7F ^ mask
    return ((seg << 4) | ((value >> (seg + 1)) & 0x0F)) ^ mask


def _ulaw_to_linear(code: int) -> int:
    code = ~code & 0xFF
    value = ((code & _QUANT_MASK) << 3) + _ULAW_BIAS
    value <<= (code & _SEG_MASK) >> _SEG_SHIFT
    return (_ULAW_BIAS - value) if code & _SIGN_BIT else (value - _ULAW_BIAS)


def _linear_to_alaw(sample: int) -> int:
    value = sample >> 3
    if value >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        value = -value - 1
    seg = _segment(value, _SEG_AEND)
    if seg >= 8:
        return 0x7F ^ mask
    code = seg << 4
    code |= ((value >> 1) if seg < 2 else (value >> seg)) & _QUANT_MASK
    return code ^ mask


def _alaw_to_linear(code: int) -> int:
    code ^= 0x55
    value = (code & _QUANT_MASK) << 4
    seg = (code & _SEG_MASK) >> _SEG_SHIFT
    if seg == 0:
        value += 8
    elif seg == 1:
        value += 0x108
    else:
        value = (value + 0x108) << (seg - 1)
    return value if code & _SIGN_BIT else -value


def _encode_table(encoder: Callable[[int], int]) -> np.ndarray:
    samples = np.arange(65536, dtype=np.uint16).view(np.int16)
    return np.array([encoder(int(sample)) for sample in samples], dtype=np.uint8)


def _decode_table(decoder: Callable[[int], int]) -> np.ndarray:
    return np.array([decoder(code) for code in range(256)], dtype=np.int16)


_ULAW_ENCODE = _encode_table(_linear_to_ulaw)
_ULAW_DECODE = _decode_table(_ulaw_to_linear)
_ALAW_ENCODE = _encode_table(_linear_to_alaw)
_ALAW_DECODE = _decode_table(_alaw_to_linear)


def ulaw_decode(data: bytes) -> bytes:
    """Decode μ-law bytes to little-endian PCM16."""
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)].astype("<i2").tobytes()


def ulaw_encode(pcm: bytes) -> bytes:
    """Encode little-endian PCM16 to μ-law bytes."""
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2")
    return _ULAW_ENCODE[samples.view(np.uint16)].tobytes()


def alaw_decode(data: bytes) -> bytes:
    """Decode A-law bytes to little-endian PCM16."""
    return _ALAW_DECODE[np.frombuffer(data, dtype=np.uint8)].astype("<i2").tobytes()


def alaw_encode(pcm: bytes) -> bytes:
    """Encode little-endian PCM16 to A-law bytes."""
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2")
    return _ALAW_ENCODE[samples.view(np.uint16)].tobytes()
