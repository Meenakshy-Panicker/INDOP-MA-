import numpy as np
import pytest

from indop import fit_anchor_gmm, forward_filter, power_timeline, run_indop
from indop.anchor_gmm import select_anchor_window, threshold_from_components
from indop.baselines import dyspan_percentile_threshold
from indop.cyclo import alpha_profile
from indop.ground_truth import ktb_dbm_per_hz, physics_threshold_for
from sim.synthetic import lte_like_iq


def two_cluster_power(n=3000, occ=0.2, seed=0):
    rng = np.random.default_rng(seed)
    busy = rng.random(n) < occ
    p = np.where(busy, rng.normal(-116.5, 0.2, n), rng.normal(-117.2, 0.07, n))
    return p, busy


def test_power_timeline_matches_definition():
    rng = np.random.default_rng(1)
    iq = (rng.normal(size=300_000) + 1j * rng.normal(size=300_000)) / np.sqrt(2)
    p = power_timeline(iq, sample_rate=1e6, chunk_seconds=0.1)
    assert p.shape == (3,)
    # unit-power complex noise: 10log10(1) - 10log10(1e6) = -60 dB
    assert np.allclose(p, -60.0, atol=0.05)


def test_threshold_rule():
    assert threshold_from_components(np.array([-117.0, -116.0]), np.array([0.1, 0.2])) == -116.5
    assert threshold_from_components(np.array([-117.0]), np.array([0.1])) == pytest.approx(-116.7)
    guarded = threshold_from_components(np.array([-117.0, -116.95]), np.array([0.1, 0.1]), 2.5)
    assert guarded == pytest.approx(-116.75)


def test_anchor_gmm_recovers_noise_cluster():
    p, busy = two_cluster_power()
    fit = fit_anchor_gmm(p, 720)
    assert fit.k >= 2
    assert fit.mu_noise == pytest.approx(-117.2, abs=0.02)
    occ = np.mean(p > fit.threshold)
    assert abs(occ - busy.mean()) < 0.03


def test_anchor_window_stops_at_drift():
    rng = np.random.default_rng(3)
    p = rng.normal(-117.2, 0.05, 2000)
    p[900:] -= 0.3  # downward floor drift at 90 s, as in paper Table 1
    sel = select_anchor_window(p, start_chunks=300, step_chunks=100)
    assert 600 <= sel.anchor_chunks <= 900


def test_hmm_posterior_tracks_states():
    p, busy = two_cluster_power(seed=4)
    fit = fit_anchor_gmm(p, 720)
    hmm = forward_filter(p, fit)
    assert hmm.posterior.shape == p.shape
    assert np.mean((hmm.posterior > 0.5) == busy) > 0.95


def test_dyspan_baseline_misses_low_occupancy():
    p, busy = two_cluster_power(occ=0.19, seed=5)
    t = dyspan_percentile_threshold(p)
    assert np.mean(p > t) < 0.05 < busy.mean()  # the paper's failure mode


def test_physics_gt_matches_paper_threshold():
    assert ktb_dbm_per_hz() == pytest.approx(-173.98, abs=0.01)
    # Paper Table 1: anchor-GMM threshold at 72 s = -116.996 dBX
    assert physics_threshold_for("31722B7", 3595, 60) == pytest.approx(-116.996, abs=0.01)
    assert physics_threshold_for("unknown", 3595, 60) is None


def test_scf_detects_lte_cyclic_features():
    rng = np.random.default_rng(6)
    prof = alpha_profile(lte_like_iq(0.5, 0.0, rng=rng))
    assert prof.peak_db(1000.0) > 6
    assert prof.peak_db(2000.0) > 3
    noise = alpha_profile((rng.normal(size=500_000) + 1j * rng.normal(size=500_000)))
    assert noise.peak_db(1000.0) < 3


def test_run_indop_and_npz_roundtrip(tmp_path):
    from indop.pipeline import load_result_npz

    p, _ = two_cluster_power(n=2000)
    res = run_indop(p, "TestNode")
    assert 0 <= res.occupancy_gmm <= 1
    assert "mu_ref_dbx" in res.meta
    f = tmp_path / "TestNode.npz"
    res.save_npz(str(f))
    s = load_result_npz(str(f))
    assert s["node"] == "TestNode"
    assert len(s["posterior"]) == 2000
    assert f.stat().st_size < 300_000  # "~200 KB per session" budget
