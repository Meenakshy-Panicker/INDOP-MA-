"""Synthetic multi-node CBRS scenario generator.

Everything produced here is SYNTHETIC. It exists so the INDOP-MA algorithms
can be developed, unit-tested and stress-tested with known ground truth before
(and alongside) the real POWDER campaigns. Parameters are chosen to resemble
the published Sagepoint fit (paper Table 4): noise floor about -117.2 dBX,
noise-cluster sigma about 0.07 dB, signal clusters a few tenths of a dB up.

Model per node i and 100 ms chunk n
------------------------------------
  noise_i[n]   = mu_i + offset_i + drift_i[n] + jitter
  active[n]    = transmitter on/off (two-state Markov, duty from schedule)
  u[n]         = fraction of the chunk carrying signal (bursty LTE load)
  snr_i        = received SNR from log-distance path loss + shadowing
  P_i[n] (lin) = noise_lin * (1 + active[n] * u[n] * snr_lin_i)
  truth_i[n]   = active[n] and 10log10(1 + u[n] snr_lin_i) >= gt_margin_db
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CHUNK_S = 0.1


@dataclass
class NodeHW:
    """Per-radio hardware behaviour."""
    mu_noise_dbx: float = -117.17
    jitter_db: float = 0.07
    unit_offset_db: float = 0.0          # unit-to-unit gain/NF spread
    drift_onset_s: float = 80.0          # USRP B210 front-end drift starts
    drift_db: float = -0.05              # total drift over the session
    anchor_bias_db: float = 0.0          # 'unlucky anchor': floor offset during first anchor_bias_s
    anchor_bias_s: float = 72.0


@dataclass
class Transmitter:
    xy: np.ndarray
    snr_at_ref_db: float = 6.0           # SNR at ref distance (dB)
    ref_m: float = 100.0
    pathloss_exp: float = 1.2           # low effective exponent: elevated rooftop links
    mean_on_s: float = 2.0               # mean burst length when active


@dataclass
class Scenario:
    node_names: list[str]
    node_xy: np.ndarray
    hw: list[NodeHW]
    tx: Transmitter
    shadow_db: np.ndarray
    seed: int = 0
    gt_margin_db: float = 0.15
    extra: dict = field(default_factory=dict)

    def snr_db(self, xy: np.ndarray | None = None) -> np.ndarray:
        """Received SNR at nodes (default) or arbitrary points (no shadowing there)."""
        pts = self.node_xy if xy is None else np.atleast_2d(xy)
        d = np.maximum(np.sqrt(((pts - self.tx.xy) ** 2).sum(-1)), 10.0)
        s = self.tx.snr_at_ref_db - 10 * self.tx.pathloss_exp * np.log10(d / self.tx.ref_m)
        return s + (self.shadow_db if xy is None else 0.0)


def make_scenario(node_names, node_xy, tx_xy, seed: int = 0, shadow_sigma_db: float = 2.0,
                  unit_spread_db: float = 0.0, hw_overrides: dict | None = None,
                  **tx_kwargs) -> Scenario:
    rng = np.random.default_rng(seed)
    hw = []
    for name in node_names:
        h = NodeHW(unit_offset_db=float(rng.normal(0, unit_spread_db)),
                   drift_onset_s=float(rng.uniform(70, 95)),
                   drift_db=float(rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)))
        for k, v in (hw_overrides or {}).get(name, {}).items():
            setattr(h, k, v)
        hw.append(h)
    return Scenario(node_names=list(node_names), node_xy=np.asarray(node_xy, float), hw=hw,
                    tx=Transmitter(xy=np.asarray(tx_xy, float), **tx_kwargs),
                    shadow_db=rng.normal(0, shadow_sigma_db, len(node_names)), seed=seed)


def activity(n_chunks: int, duty: np.ndarray | float, mean_on_s: float,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Transmitter on/off sequence and per-chunk load fraction u in (0, 1]."""
    duty = np.broadcast_to(np.asarray(duty, float), (n_chunks,))
    p_off = CHUNK_S / mean_on_s                       # on -> off
    on = np.zeros(n_chunks, bool)
    state = rng.random() < duty[0]
    for n in range(n_chunks):
        d = duty[n]
        if d <= 0:
            state = False
        elif d >= 1:
            state = True
        else:
            p_on = p_off * d / (1 - d)                 # off -> on giving stationary duty d
            state = (rng.random() > p_off) if state else (rng.random() < p_on)
        on[n] = state
    # Bursty LTE load: mostly light frames with occasional full ones (multi-cluster GMM).
    u = np.where(rng.random(n_chunks) < 0.85, rng.beta(2, 5, n_chunks), rng.beta(5, 1.5, n_chunks))
    return on, np.clip(u, 0.02, 1.0)


def simulate_power(sc: Scenario, minutes: float = 20.0, duty: np.ndarray | float = 0.25,
                   start_hour: float | None = None, seed: int | None = None) -> dict:
    """Simulate simultaneous power timelines for every node.

    Returns dict with ``power`` (n_nodes, n_chunks) dBX, ``truth`` (n_nodes,
    n_chunks) bool, ``active`` (n_chunks,) bool, ``snr_db`` (n_nodes,),
    ``noise_true`` (n_nodes, n_chunks) and ``t`` seconds.
    """
    rng = np.random.default_rng(sc.seed if seed is None else seed)
    n = int(round(minutes * 60 / CHUNK_S))
    t = np.arange(n) * CHUNK_S
    if callable(duty):
        duty = duty((start_hour or 0.0) + t / 3600.0)
    on, u = activity(n, duty, sc.tx.mean_on_s, rng)
    snr_db = sc.snr_db()
    power = np.empty((len(sc.hw), n))
    truth = np.empty((len(sc.hw), n), bool)
    noise_true = np.empty((len(sc.hw), n))
    for i, h in enumerate(sc.hw):
        base = h.mu_noise_dbx + h.unit_offset_db
        drift = np.where(t > h.drift_onset_s,
                         h.drift_db * np.clip((t - h.drift_onset_s) / (minutes * 60 - h.drift_onset_s), 0, 1),
                         0.0)
        drift = drift + np.where(t < h.anchor_bias_s, h.anchor_bias_db, 0.0)
        noise_db = base + drift
        noise_true[i] = noise_db
        rise_db = 10 * np.log10(1 + on * u * 10 ** (snr_db[i] / 10))
        truth[i] = on & (rise_db >= sc.gt_margin_db)
        power[i] = noise_db + rise_db + rng.normal(0, h.jitter_db, n)
    return {"power": power, "truth": truth, "active": on, "load": u, "snr_db": snr_db,
            "noise_true": noise_true, "t": t}


def carrier_schedule(hours: np.ndarray, idle_start: float = 4.0, idle_end: float = 7.0,
                     peak: float = 0.45, trough: float = 0.12) -> np.ndarray:
    """Diurnal duty cycle with a cooperating-carrier idle window (OpenZMS 4-7 am)."""
    h = np.mod(hours, 24.0)
    load = trough + (peak - trough) * 0.5 * (1 - np.cos(2 * np.pi * (h - 4.0) / 24.0))
    return np.where((h >= idle_start) & (h < idle_end), 0.0, load)


def lte_like_iq(seconds: float, snr_db: float, fs: float = 1e6,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Complex baseband with LTE-like 1 ms subframe / 0.5 ms slot structure in AWGN.

    The OFDM payload is modelled as complex Gaussian whose power envelope has a
    control/reference region at each subframe start and a weaker one at each
    slot start, which yields cyclic features at alpha = 1 kHz and 2 kHz.
    """
    rng = rng or np.random.default_rng()
    n = int(seconds * fs)
    t = np.arange(n) / fs
    sub_phase = np.mod(t, 1e-3)
    slot_phase = np.mod(t, 0.5e-3)
    env = 1.0 + 1.2 * (sub_phase < 0.07e-3) + 0.6 * (slot_phase < 0.07e-3)
    env = env / np.sqrt(np.mean(env ** 2))
    sig = env * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    return (10 ** (snr_db / 20)) * sig + noise
