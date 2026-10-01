"""Power timeline (paper Eq. 1).

Each 100 ms chunk of raw IQ is reduced to one scalar power value in
uncalibrated dBX (dB relative to ADC full scale, normalised per Hz):

    P[n] = 10 log10( (1/N) * sum_k |x[nN + k]|^2 ) - 10 log10(BW)
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

DEFAULT_SAMPLE_RATE = 1_000_000  # 1 MHz capture bandwidth
DEFAULT_CHUNK_SECONDS = 0.1  # 100 ms capture period


def chunk_size(sample_rate: float = DEFAULT_SAMPLE_RATE,
               chunk_seconds: float = DEFAULT_CHUNK_SECONDS) -> int:
    """Number of IQ samples per chunk (N = 1e5 at 1 MHz / 100 ms)."""
    return int(round(sample_rate * chunk_seconds))


def power_timeline(iq: np.ndarray,
                   sample_rate: float = DEFAULT_SAMPLE_RATE,
                   chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
                   bandwidth: float | None = None) -> np.ndarray:
    """Compute the per-chunk power timeline from a complex IQ array.

    Trailing samples that do not fill a whole chunk are discarded.
    """
    n = chunk_size(sample_rate, chunk_seconds)
    bw = sample_rate if bandwidth is None else bandwidth
    usable = (len(iq) // n) * n
    if usable == 0:
        raise ValueError(f"need at least {n} samples, got {len(iq)}")
    blocks = np.asarray(iq[:usable]).reshape(-1, n)
    mean_pow = np.mean(np.abs(blocks) ** 2, axis=1)
    return 10.0 * np.log10(mean_pow + 1e-30) - 10.0 * np.log10(bw)


def power_timeline_streaming(chunks: Iterable[np.ndarray],
                             bandwidth: float = DEFAULT_SAMPLE_RATE) -> Iterator[float]:
    """Streaming variant for on-node use: yields P[n] for each IQ chunk.

    Lets a 5.4 GB capture be processed without holding it in memory.
    """
    off = 10.0 * np.log10(bandwidth)
    for c in chunks:
        yield float(10.0 * np.log10(np.mean(np.abs(c) ** 2) + 1e-30) - off)


def read_iq_file(path: str, dtype: str = "complex64",
                 chunk_samples: int | None = None) -> Iterator[np.ndarray]:
    """Iterate over a raw interleaved IQ file (e.g. written by UHD rx_samples_to_file).

    ``dtype`` is ``complex64`` (fc32) or ``int16`` (sc16, interleaved I/Q).
    """
    n = chunk_samples or chunk_size()
    if dtype == "complex64":
        mm = np.memmap(path, dtype=np.complex64, mode="r")
        for i in range(0, len(mm) - n + 1, n):
            yield np.asarray(mm[i:i + n])
    elif dtype == "int16":
        mm = np.memmap(path, dtype=np.int16, mode="r")
        step = 2 * n
        for i in range(0, len(mm) - step + 1, step):
            raw = np.asarray(mm[i:i + step], dtype=np.float32) / 32768.0
            yield raw[0::2] + 1j * raw[1::2]
    else:
        raise ValueError(f"unsupported dtype {dtype!r}")
