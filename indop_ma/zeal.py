"""OpenZMS / ZeAL observation payloads.

INDOP-MA is additive to the existing ZeAL Monitor -> DST interface: each agent
still posts its own Observation; jointly derived fields ride in an extra
``indop_ma`` block that a DST unaware of INDOP-MA can ignore.

The field names below are an illustrative schema for this project, not the
official ZeAL wire format. Map them onto the OpenZMS client of the deployment
(``zms-client`` / ZeAL Monitor API) in ``build_observation``'s caller.
"""

from __future__ import annotations

import datetime as _dt
import json


def build_observation(summary: dict, monitor_id: str, freq_start_hz: float,
                      freq_end_hz: float, lat: float, lon: float,
                      session_start: _dt.datetime, session_seconds: float,
                      ma_block: dict | None = None) -> dict:
    """``summary`` is ``INDOPResult.summary()`` or a loaded result NPZ summary."""
    s = summary
    obs = {
        "monitor_id": monitor_id,
        "kind": "occupancy",
        "freq_start": freq_start_hz,
        "freq_end": freq_end_hz,
        "starts_at": session_start.isoformat(),
        "ends_at": (session_start + _dt.timedelta(seconds=session_seconds)).isoformat(),
        "location": {"lat": lat, "lon": lon},
        "occupancy": s["occupancy_hmm_soft"] if s["occupancy_hmm_soft"] is not None
        else s["occupancy_gmm"],
        "resolution_ms": 100,
        "method": "INDOP anchor-GMM + HMM (calibration-free)",
        "detail": {k: s[k] for k in ("mu_noise_dbx", "sigma_noise_db", "threshold_dbx",
                                     "occupancy_gmm", "occupancy_hmm_hard",
                                     "occupancy_hmm_soft")},
    }
    if ma_block:
        obs["indop_ma"] = ma_block
    return obs


def to_json(obs: dict) -> str:
    return json.dumps(obs, indent=2, default=float)
