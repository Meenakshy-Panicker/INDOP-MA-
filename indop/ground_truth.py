"""Physics ground truth (paper Sec. 4.3, Eq. 5) and the self-calibrated fallback.

    T_dBm = kTB + NF(f, g) + 3 dB

converted to the uncalibrated dBX domain with the radio's gain constant K.
For CBRS 3595 MHz at gain 60 dB on serial 31722B7: K = -37.79 dB,
NF = 16.19 dB. Other radios use T_self = mu_noise + 3 sigma_noise.
"""

from __future__ import annotations

import numpy as np

BOLTZMANN = 1.380649e-23
T0_KELVIN = 290.0

# Per-radio calibration constants available from the lab (serial -> (f_MHz, gain) -> (K, NF)).
CALIBRATION_TABLES: dict[str, dict[tuple[int, int], tuple[float, float]]] = {
    "31722B7": {(3595, 60): (-37.79, 16.19)},
}


def ktb_dbm_per_hz(temperature_k: float = T0_KELVIN) -> float:
    """Thermal noise density in dBm/Hz (about -174 dBm/Hz at 290 K)."""
    return 10 * np.log10(BOLTZMANN * temperature_k * 1e3)


def physics_threshold_dbx(k_db: float, nf_db: float, margin_db: float = 3.0,
                          temperature_k: float = T0_KELVIN) -> float:
    """Calibrated threshold in the per-Hz dBX domain used by the power timeline.

    dBm/Hz = dBX/Hz + K  =>  dBX/Hz = dBm/Hz - K.
    For the Sagepoint B210 at 3595 MHz / 60 dB this gives about -117.0 dBX,
    which matches the anchor-GMM threshold in paper Table 1.
    """
    return ktb_dbm_per_hz(temperature_k) + nf_db + margin_db - k_db


def physics_threshold_for(serial: str, freq_mhz: int, gain_db: int) -> float | None:
    """Look up the calibrated threshold; ``None`` if no table exists for this radio."""
    entry = CALIBRATION_TABLES.get(serial, {}).get((freq_mhz, gain_db))
    return None if entry is None else physics_threshold_dbx(*entry)


def self_calibrated_threshold(mu_noise: float, sigma_noise: float) -> float:
    """Fallback GT for radios without tables: T_self = mu_noise + 3 sigma_noise."""
    return mu_noise + 3.0 * sigma_noise
