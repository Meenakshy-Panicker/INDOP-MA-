"""Node topology: positions, distances and peer neighbourhoods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

EARTH_RADIUS_M = 6_371_000.0
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NODES_YAML = REPO_ROOT / "config" / "nodes.yaml"


@dataclass(frozen=True)
class Node:
    name: str
    lat: float
    lon: float
    location: str = ""
    serial: str | None = None
    radio: str = "USRP B210"


def load_config(path: str | Path = DEFAULT_NODES_YAML) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_nodes(path: str | Path = DEFAULT_NODES_YAML) -> list[Node]:
    cfg = load_config(path)
    return [Node(name=n["name"], lat=n["lat"], lon=n["lon"],
                 location=n.get("location", ""), serial=n.get("serial"),
                 radio=n.get("radio", "USRP B210")) for n in cfg["nodes"]]


def to_local_xy(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection to metres around (lat0, lon0)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def node_xy(nodes: list[Node]) -> tuple[np.ndarray, float, float]:
    """(n, 2) metre coordinates plus the reference origin used."""
    lat0 = float(np.mean([n.lat for n in nodes]))
    lon0 = float(np.mean([n.lon for n in nodes]))
    x, y = to_local_xy([n.lat for n in nodes], [n.lon for n in nodes], lat0, lon0)
    return np.column_stack([x, y]), lat0, lon0


def distance_matrix(xy: np.ndarray) -> np.ndarray:
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def neighbours(xy: np.ndarray, radius_m: float | None = None,
               k: int | None = None) -> list[list[int]]:
    """Peer sets: all nodes within ``radius_m`` and/or the ``k`` nearest (excluding self)."""
    D = distance_matrix(xy)
    out = []
    for i in range(len(xy)):
        order = [j for j in np.argsort(D[i]) if j != i]
        if radius_m is not None:
            order = [j for j in order if D[i, j] <= radius_m]
        if k is not None:
            order = order[:k]
        out.append(order)
    return out
