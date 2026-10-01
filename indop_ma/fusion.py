"""Joint occupancy fusion rules and grant-quality metrics (INDOP-MA form 2).

Decisions are made per 100 ms chunk. A *grant* to a secondary consumer is
issued when the fused decision says "idle".

Rules
-----
* ``or``        occupied if any node sees it  -> protects incumbents
* ``and``       occupied only if all nodes see it -> maximises reuse
* ``k_of_n``    occupied if at least k nodes see it
* ``weighted``  occupied if sum_i w_i * p_i >= tau, with w_i from distance
                to the consumer location (IDW) times per-node confidence;
                intended for DSA scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def or_rule(decisions: np.ndarray) -> np.ndarray:
    """decisions: (n_nodes, n_chunks) bool -> (n_chunks,) bool."""
    return np.any(decisions, axis=0)


def and_rule(decisions: np.ndarray) -> np.ndarray:
    return np.all(decisions, axis=0)


def k_of_n_rule(decisions: np.ndarray, k: int) -> np.ndarray:
    return np.sum(decisions, axis=0) >= k


def idw_weights(node_xy: np.ndarray, target_xy: np.ndarray, power: float = 2.0,
                eps_m: float = 25.0) -> np.ndarray:
    d = np.sqrt(((node_xy - np.asarray(target_xy)) ** 2).sum(-1))
    w = 1.0 / (d + eps_m) ** power
    return w / w.sum()


def weighted_rule(probs: np.ndarray, weights: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """probs: (n_nodes, n_chunks) posteriors in [0, 1]; weights sum to 1."""
    w = np.asarray(weights, float)[:, None]
    return (w * probs).sum(axis=0) / w.sum() >= tau


def fuse(rule: str, decisions: np.ndarray, probs: np.ndarray | None = None,
         weights: np.ndarray | None = None, k: int = 2, tau: float = 0.5) -> np.ndarray:
    if rule == "or":
        return or_rule(decisions)
    if rule == "and":
        return and_rule(decisions)
    if rule == "k_of_n":
        return k_of_n_rule(decisions, k)
    if rule == "majority":
        return k_of_n_rule(decisions, decisions.shape[0] // 2 + 1)
    if rule == "weighted":
        p = decisions.astype(float) if probs is None else probs
        w = np.full(p.shape[0], 1.0 / p.shape[0]) if weights is None else weights
        return weighted_rule(p, w, tau)
    raise ValueError(f"unknown rule {rule!r}")


@dataclass
class GrantMetrics:
    false_grant_rate: float      # P(grant | consumer location occupied)
    missed_detection: float      # P(fused idle | incumbent active anywhere protected)
    false_denial_rate: float     # P(deny | consumer location idle) - lost opportunity
    grant_utilisation: float     # fraction of truly idle time that was granted

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def grant_metrics(fused_occupied: np.ndarray, truth_consumer: np.ndarray,
                  truth_incumbent: np.ndarray) -> GrantMetrics:
    fused = np.asarray(fused_occupied, bool)
    tc, ti = np.asarray(truth_consumer, bool), np.asarray(truth_incumbent, bool)
    grant = ~fused

    def rate(mask_num, mask_den):
        den = mask_den.sum()
        return float((mask_num & mask_den).sum() / den) if den else 0.0

    return GrantMetrics(
        false_grant_rate=rate(grant, tc),
        missed_detection=rate(grant, ti),
        false_denial_rate=rate(fused, ~tc),
        grant_utilisation=rate(grant, ~tc),
    )
