"""Spatial occupancy maps from simultaneous multi-node observations.

Interpolates per-node occupancy onto a regular grid of *proxels* (the DST's
spatial cells). Two estimators:

* IDW   - inverse-distance weighting, cheap and assumption-free.
* GP    - Gaussian-process regression (ordinary-kriging style) with an RBF +
          white-noise kernel; also returns a per-proxel standard deviation,
          i.e. *where* the map should not be trusted.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyMap:
    x: np.ndarray        # (nx,) metres
    y: np.ndarray        # (ny,) metres
    value: np.ndarray    # (ny, nx) occupancy in [0, 1]
    std: np.ndarray | None = None
    method: str = "idw"

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Nearest-proxel lookup for (m, 2) query points."""
        ix = np.clip(np.searchsorted(self.x, xy[:, 0]), 0, len(self.x) - 1)
        iy = np.clip(np.searchsorted(self.y, xy[:, 1]), 0, len(self.y) - 1)
        return self.value[iy, ix]


def make_grid(xy: np.ndarray, pad_m: float = 250.0, res_m: float = 20.0):
    x = np.arange(xy[:, 0].min() - pad_m, xy[:, 0].max() + pad_m + res_m, res_m)
    y = np.arange(xy[:, 1].min() - pad_m, xy[:, 1].max() + pad_m + res_m, res_m)
    return x, y


def idw_map(xy: np.ndarray, occ: np.ndarray, power: float = 2.0,
            pad_m: float = 250.0, res_m: float = 20.0) -> OccupancyMap:
    x, y = make_grid(xy, pad_m, res_m)
    gx, gy = np.meshgrid(x, y)
    d = np.sqrt((gx[..., None] - xy[:, 0]) ** 2 + (gy[..., None] - xy[:, 1]) ** 2)
    w = 1.0 / np.maximum(d, 1.0) ** power
    val = (w * np.asarray(occ)).sum(-1) / w.sum(-1)
    return OccupancyMap(x=x, y=y, value=val, method="idw")


def gp_map(xy: np.ndarray, occ: np.ndarray, length_scale_m: float = 400.0,
           noise: float = 1e-3, pad_m: float = 250.0, res_m: float = 20.0) -> OccupancyMap:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    occ = np.asarray(occ, float)
    kernel = (ConstantKernel(np.var(occ) + 1e-4, (1e-5, 1.0))
              * RBF(length_scale_m, (100.0, 5000.0)) + WhiteKernel(noise, (1e-6, 1e-1)))
    with warnings.catch_warnings():  # 7 points: hyper-parameters often hit their bounds
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=0).fit(xy, occ)
    x, y = make_grid(xy, pad_m, res_m)
    gx, gy = np.meshgrid(x, y)
    mu, sd = gp.predict(np.column_stack([gx.ravel(), gy.ravel()]), return_std=True)
    return OccupancyMap(x=x, y=y, value=np.clip(mu, 0, 1).reshape(gx.shape),
                        std=sd.reshape(gx.shape), method="gp")


def map_error(est: OccupancyMap, truth: np.ndarray) -> dict:
    """MAE / RMSE / max error (percentage points) against a truth grid."""
    e = 100 * (est.value - truth)
    return {"mae_pp": float(np.mean(np.abs(e))),
            "rmse_pp": float(np.sqrt(np.mean(e ** 2))),
            "max_pp": float(np.max(np.abs(e)))}


def load_dst_proxels_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an OpenZMS DST occupancy export as (lat/lon (m, 2), occupancy (m,)).

    Expected CSV columns: ``lat,lon,occupancy`` (occupancy in [0, 1]). This is
    the hook for comparing INDOP maps against OpenZMS Figure 7 maps once the
    DST export for the same session window is available.
    """
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return np.column_stack([arr["lat"], arr["lon"]]), arr["occupancy"]
