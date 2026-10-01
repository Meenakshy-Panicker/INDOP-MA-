"""INDOP-MA: multi-agent extension of INDOP.

Agents (one per POWDER node) coordinate before reporting to the OpenZMS DST:

1. ``consensus``   - noise-floor consensus with majority outlier correction
2. ``fusion``      - joint occupancy fusion (OR / AND / k-of-n / weighted)
   and ``spatial_map`` - location-aware occupancy maps on DST proxels
3. ``coop_cyclo``  - cooperative cyclostationary (SCF) detection
"""

from .consensus import NoiseReport, gossip_average, noise_floor_consensus
from .coop_cyclo import cooperative_detect, detect_lte
from .fusion import fuse, grant_metrics, idw_weights
from .spatial_map import gp_map, idw_map, map_error
from .topology import Node, load_nodes, node_xy

__all__ = [
    "Node", "NoiseReport", "cooperative_detect", "detect_lte", "fuse", "gossip_average",
    "gp_map", "grant_metrics", "idw_map", "idw_weights", "load_nodes", "map_error",
    "node_xy", "noise_floor_consensus",
]
