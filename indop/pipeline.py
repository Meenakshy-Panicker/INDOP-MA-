"""End-to-end single-node INDOP pipeline.

power timeline -> anchor-window selection -> anchor GMM -> HMM -> (optional) SCF.
The result is a small, JSON/NPZ-serialisable record (~200 KB with the
posterior) so raw IQ never has to leave the node.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .anchor_gmm import GMMFit, fit_anchor_gmm, occupancy, select_anchor_window
from .cyclo import LTE_ALPHAS_HZ, AlphaProfile, alpha_profile
from .hmm import HMMResult, forward_filter


@dataclass
class INDOPConfig:
    anchor_chunks: int | None = 720   # 72 s at 100 ms; None -> automatic drift detection
    k_max: int = 5
    run_hmm: bool = True
    scf_nfft: int = 8192
    min_margin_sigma: float | None = None  # optional separation guard (None = paper Eq. 3)


@dataclass
class INDOPResult:
    node: str
    gmm: GMMFit
    occupancy_gmm: float
    hmm: HMMResult | None = None
    scf: AlphaProfile | None = None
    meta: dict = field(default_factory=dict)

    @property
    def occupancy_hmm_hard(self) -> float | None:
        return None if self.hmm is None else self.hmm.hard_occupancy

    @property
    def occupancy_hmm_soft(self) -> float | None:
        return None if self.hmm is None else self.hmm.soft_occupancy

    def summary(self) -> dict:
        s = {
            "node": self.node,
            "k": self.gmm.k,
            "mu_noise_dbx": self.gmm.mu_noise,
            "sigma_noise_db": self.gmm.sigma_noise,
            "threshold_dbx": self.gmm.threshold,
            "anchor_chunks": self.gmm.anchor_chunks,
            "occupancy_gmm": self.occupancy_gmm,
            "occupancy_hmm_hard": self.occupancy_hmm_hard,
            "occupancy_hmm_soft": self.occupancy_hmm_soft,
            **self.meta,
        }
        if self.scf is not None:
            s["scf_peaks_db"] = {f"{a:.0f}": self.scf.peak_db(a) for a in LTE_ALPHAS_HZ}
        return s

    def save_npz(self, path: str) -> None:
        """Persist the compact result record that is shipped off-node."""
        np.savez_compressed(
            path,
            summary=json.dumps(self.summary()),
            gmm=json.dumps(self.gmm.to_dict()),
            posterior=(self.hmm.posterior.astype(np.float16)
                       if self.hmm is not None else np.array([])),
            scf_alphas=self.scf.alphas if self.scf is not None else np.array([]),
            scf_profile_db=self.scf.profile_db if self.scf is not None else np.array([]),
        )


def run_indop(power: np.ndarray, node: str = "node", config: INDOPConfig | None = None,
              iq_for_scf: np.ndarray | None = None, sample_rate: float = 1e6,
              meta: dict | None = None) -> INDOPResult:
    cfg = config or INDOPConfig()
    power = np.asarray(power, dtype=float)
    w = cfg.anchor_chunks
    if w is None:
        w = select_anchor_window(power).anchor_chunks
    fit = fit_anchor_gmm(power, w, k_max=cfg.k_max, min_margin_sigma=cfg.min_margin_sigma)
    meta = dict(meta or {})
    if len(power) >= 2 * w:
        # Post-anchor reference noise mean, shared with peers by INDOP-MA consensus.
        meta["mu_ref_dbx"] = fit_anchor_gmm(power[w:2 * w], k_max=cfg.k_max).mu_noise
    hmm = forward_filter(power, fit) if cfg.run_hmm else None
    scf = alpha_profile(iq_for_scf, sample_rate, cfg.scf_nfft) if iq_for_scf is not None else None
    return INDOPResult(node=node, gmm=fit, occupancy_gmm=occupancy(power, fit.threshold),
                       hmm=hmm, scf=scf, meta=meta)


def load_result_npz(path: str) -> dict:
    """Load the summary dict from a result NPZ produced by ``save_npz``."""
    with np.load(path, allow_pickle=False) as z:
        s = json.loads(str(z["summary"]))
        s["gmm"] = json.loads(str(z["gmm"]))
        s["posterior"] = z["posterior"].astype(float)
    return s
