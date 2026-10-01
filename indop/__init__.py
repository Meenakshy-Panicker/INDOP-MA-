"""INDOP: calibration-free spectrum occupancy estimation from raw IQ.

Single-node core used by every INDOP-MA agent.
"""

from .anchor_gmm import GMMFit, fit_anchor_gmm, occupancy, select_anchor_window
from .cyclo import AlphaProfile, alpha_profile, average_profiles
from .hmm import HMMResult, forward_filter
from .pipeline import INDOPConfig, INDOPResult, load_result_npz, run_indop
from .power import power_timeline

__all__ = [
    "AlphaProfile", "GMMFit", "HMMResult", "INDOPConfig", "INDOPResult",
    "alpha_profile", "average_profiles", "fit_anchor_gmm", "forward_filter",
    "load_result_npz", "occupancy", "power_timeline", "run_indop",
    "select_anchor_window",
]

__version__ = "0.1.0"
