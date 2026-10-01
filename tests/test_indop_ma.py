import numpy as np
import pytest

from indop_ma.consensus import NoiseReport, gossip_average, noise_floor_consensus, robust_z
from indop_ma.coop_cyclo import cooperative_detect
from indop_ma.fusion import fuse, grant_metrics, idw_weights
from indop_ma.spatial_map import gp_map, idw_map, map_error
from indop_ma.topology import distance_matrix, load_nodes, neighbours, node_xy


def reports(mu_offsets, ref_offsets=None):
    ref_offsets = ref_offsets or [0.0] * len(mu_offsets)
    out = []
    for i, (m, r) in enumerate(zip(mu_offsets, ref_offsets)):
        base = -117.17 + 0.3 * np.sin(i)  # heterogeneous hardware
        out.append(NoiseReport(f"n{i}", mu_noise=base + m, sigma_noise=0.07,
                               threshold=base + m + 0.17, mu_ref=base + r))
    return out


def test_robust_z_flags_single_outlier():
    z = robust_z(np.array([0.0, 0.01, -0.01, 0.005, 0.2]))
    assert abs(z[-1]) > 3.5 and np.all(np.abs(z[:-1]) < 3.5)


def test_offset_consensus_corrects_only_the_faulty_node():
    reps = reports([0, 0, -0.10, 0, 0, 0, 0], [0.004, -0.003, 0, 0.002, -0.001, 0.003, 0])
    out = noise_floor_consensus(reps, mode="offset")
    assert out.outliers == ["n2"]
    fixed = out.corrected[2]
    assert fixed.threshold - reps[2].threshold == pytest.approx(0.10, abs=0.01)
    for a, b in zip(reps, out.corrected):
        if a.node != "n2":
            assert a == b


def test_absolute_consensus_fails_on_heterogeneous_hardware():
    reps = reports([0, 0, -0.10, 0, 0, 0, 0], [0] * 7)
    out = noise_floor_consensus(reps, mode="absolute")
    # 0.3 dB unit spread swamps a 0.1 dB anchor error: the faulty node is not singled out
    assert "n2" not in out.outliers or len(out.outliers) > 1


def test_offset_mode_requires_reference():
    r = NoiseReport("x", -117, 0.07, -116.8)
    with pytest.raises(ValueError):
        noise_floor_consensus([r, r, r], mode="offset")


def test_gossip_converges_to_mean():
    peers = [[1], [0, 2], [1, 3], [2]]
    x = gossip_average(np.array([1.0, 2.0, 3.0, 10.0]), peers, rounds=200, step=0.3)
    assert np.allclose(x, x.mean(), atol=0.05)


def test_fusion_rules():
    d = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], bool)
    assert fuse("or", d).tolist() == [True, False, True, True]
    assert fuse("and", d).tolist() == [False, False, False, True]
    assert fuse("majority", d).tolist() == [False, False, False, True]
    w = np.array([0.8, 0.1, 0.1])
    assert fuse("weighted", d, weights=w).tolist() == [True, False, False, True]


def test_grant_metrics():
    fused = np.array([1, 0, 0, 1], bool)
    truth = np.array([1, 1, 0, 0], bool)
    m = grant_metrics(fused, truth, truth)
    assert m.false_grant_rate == 0.5
    assert m.grant_utilisation == 0.5
    assert m.false_denial_rate == 0.5


def test_idw_weights_favour_nearest():
    xy = np.array([[0, 0], [1000, 0]], float)
    w = idw_weights(xy, np.array([10.0, 0.0]))
    assert w[0] > 0.9 and np.isclose(w.sum(), 1)


def test_topology_from_config():
    nodes = load_nodes()
    assert [n.name for n in nodes] == ["Sagepoint", "Web", "Guesthouse", "Moran", "Madsen",
                                       "Law73", "EBC"]
    xy, _, _ = node_xy(nodes)
    D = distance_matrix(xy)
    assert D.shape == (7, 7) and np.all(D[np.triu_indices(7, 1)] > 50)
    assert all(len(p) == 2 for p in neighbours(xy, k=2))


def test_maps_interpolate_exactly_at_nodes():
    xy = np.array([[0, 0], [500, 0], [0, 500], [500, 500]], float)
    occ = np.array([0.1, 0.2, 0.3, 0.4])
    for m in (idw_map(xy, occ, res_m=10), gp_map(xy, occ, res_m=10)):
        assert np.allclose(m.sample(xy), occ, atol=0.03)
        assert map_error(m, m.value)["mae_pp"] == 0


def test_cooperative_cyclo_runs():
    from indop.cyclo import alpha_profile
    from sim.synthetic import lte_like_iq

    rng = np.random.default_rng(0)
    profs = [alpha_profile(lte_like_iq(0.25, -5, rng=rng)) for _ in range(3)]
    assert cooperative_detect(profs, threshold_db=1.0).detected
