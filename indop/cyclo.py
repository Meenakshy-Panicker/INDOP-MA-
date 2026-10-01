"""Cyclostationary validation via the Spectral Correlation Function (Sec. 3.5).

    SCF(f, alpha) = (1/L) sum_l FFT_l[f] * conj(FFT_l[f - alpha])

The alpha-profile y(alpha) = max_f |SCF(f, alpha)| is computed with
N_fft = 8192 (122 Hz resolution at 1 MHz). LTE shows spikes at
alpha = 1000 Hz (1 ms subframe) and 2000 Hz (0.5 ms slot).

Note: the cyclic frequencies of interest here (kHz) are far below the
capture bandwidth, so the SCF is evaluated with an FFT-accumulation
approach on the *instantaneous power envelope*, which exposes the
subframe periodicity directly and is cheap enough to run on the node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LTE_ALPHAS_HZ = (1000.0, 2000.0)


@dataclass
class AlphaProfile:
    alphas: np.ndarray        # Hz
    profile_db: np.ndarray    # dB, normalised to the median (noise) level
    n_segments: int

    def peak_db(self, alpha_hz: float, tol_hz: float | None = None) -> float:
        """Peak height (dB above noise) within one bin of ``alpha_hz``."""
        res = self.alphas[1] - self.alphas[0]
        tol = res if tol_hz is None else tol_hz
        m = np.abs(self.alphas - alpha_hz) <= tol
        return float(self.profile_db[m].max()) if m.any() else float("nan")


def scf_alpha_power(iq: np.ndarray, sample_rate: float, nfft: int = 8192,
                    max_alpha_hz: float = 5000.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (alphas, |SCF|-profile linear, n_segments) before normalisation.

    Implementation: the cyclic autocorrelation at lag 0 is the Fourier
    transform of |x|^2; averaging |FFT(|x_l|^2)| over L segments of length
    nfft yields max_f |SCF(f, alpha)| up to a constant for signals whose
    cyclic features appear in the power envelope (LTE frame structure).
    """
    x = np.asarray(iq)
    L = len(x) // nfft
    if L < 1:
        raise ValueError(f"need >= {nfft} samples")
    segs = np.abs(x[:L * nfft].reshape(L, nfft)) ** 2
    segs = segs - segs.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(segs * np.hanning(nfft), axis=1))
    prof = spec.mean(axis=0)
    alphas = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    keep = (alphas > 0) & (alphas <= max_alpha_hz)
    return alphas[keep], prof[keep], L


def alpha_profile(iq: np.ndarray, sample_rate: float = 1e6, nfft: int = 8192,
                  max_alpha_hz: float = 5000.0) -> AlphaProfile:
    alphas, prof, L = scf_alpha_power(iq, sample_rate, nfft, max_alpha_hz)
    return AlphaProfile(alphas=alphas,
                        profile_db=10 * np.log10(prof / np.median(prof)),
                        n_segments=L)


def average_profiles(profiles: list[AlphaProfile]) -> AlphaProfile:
    """Average alpha-profiles in the linear domain (used by INDOP-MA)."""
    lin = np.mean([10 ** (p.profile_db / 10) for p in profiles], axis=0)
    return AlphaProfile(alphas=profiles[0].alphas,
                        profile_db=10 * np.log10(lin / np.median(lin)),
                        n_segments=sum(p.n_segments for p in profiles))
