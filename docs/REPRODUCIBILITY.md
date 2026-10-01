# Reproducing and extending the campaign

## 1. Local setup (no radio needed)

```bash
git clone https://github.com/Meenakshy-Panicker/INDOP-MA-.git
cd INDOP-MA-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # 20 unit tests, ~3 s
```

## 2. Regenerate every figure

| Command | Output | Data | Runtime |
|---|---|---|---|
| `python scripts/plot_paper_figures.py` | fig01–fig06 | **measured** (paper tables) | seconds |
| `python scripts/plot_pipeline_example.py` | fig07 | synthetic, or `--power file.npy` | seconds |
| `python scripts/run_consensus_experiment.py` | fig10, `results/consensus_synthetic.csv` | synthetic | ~5 min |
| `python scripts/run_fusion_experiment.py` | fig11, `results/fusion_synthetic.csv` | synthetic | ~3 min |
| `python scripts/run_spatial_map.py` | fig12–13, `results/spatial_map_synthetic.json` | synthetic | ~10 s |
| `python scripts/run_coop_cyclo_experiment.py` | fig14, `results/coop_cyclo_synthetic.csv` | synthetic | ~5 min |
| `python scripts/run_temporal_campaign.py` | fig15, `results/temporal_campaign_synthetic.csv` | synthetic | ~6 min |

`make figures` runs all of them.

## 3. Running on POWDER

### 3.1 Reserve nodes

Create an experiment that has the seven fixed-endpoint NUC + B210 nodes listed in `config/nodes.yaml` (Sagepoint, Web, Guesthouse, Moran, Madsen, Law73, EBC). Use an image with UHD ≥ 3.15 and its Python bindings. Also request an RF/spectrum reservation that covers 3594.5–3595.5 MHz for receive-only operation.

### 3.2 Install on every node

```bash
git clone https://github.com/Meenakshy-Panicker/INDOP-MA-.git && cd INDOP-MA-
pip install --user -r requirements.txt
python -c "import uhd; print(uhd.get_version_string())"
```

### 3.3 Capture simultaneously

All nodes must start within a second of each other. A simple approach is to launch from the experiment head with `parallel-ssh` at a fixed wall-clock time:

```bash
# session tag = hour of day, used by run_temporal_campaign.py --npz-dir
pssh -h nodes.txt -t 0 -i \
  'cd INDOP-MA- && python scripts/capture_uhd.py --node $(hostname -s | cut -d. -f1) \
       --minutes 20 --gain 60 --freq 3595e6 --session $(date -u +%H) --out results/npz'
```

`capture_uhd.py` streams IQ, computes the 100 ms power timeline on the fly, keeps 2 s of IQ for the SCF, and runs the full INDOP pipeline on the node. Each node writes `<Node>_<HH>.npz` (≈200 KB) and `<Node>_<HH>_power.npy`. **Raw IQ is not stored unless `--to-file` is given.**

Rename the hostnames to the node names in `config/nodes.yaml`, or pass `--node` explicitly.

### 3.4 Proposed temporal schedule (extension 1)

| Session | UTC−6 local | Purpose |
|---|---|---|
| Night | 01:00, 03:00 | baseline before the idle window |
| **Carrier idle** | **04:00, 05:00, 06:00** | can INDOP see the 4–7 am idle window on all 7 nodes? |
| Morning | 08:00, 10:00 | ramp-up |
| Afternoon | 13:00, 15:00 | peak load |
| Evening | 18:00, 21:00 | decay |

Repeat the schedule on at least 3 weekdays and 1 weekend day.

### 3.5 Collect and coordinate

```bash
rsync node-*:INDOP-MA-/results/npz/*.npz results/npz/   # ~200 KB × 7 per session
python scripts/run_multinode_campaign.py --coordinate --out results/npz
python scripts/run_consensus_experiment.py --npz-dir results/npz
python scripts/run_temporal_campaign.py --npz-dir results/npz
python scripts/plot_pipeline_example.py --power results/npz/Guesthouse_05_power.npy
```

### 3.6 Ground truth for validation

* **Sagepoint (serial 31722B7):** exact physics GT, computed automatically from `CALIBRATION_TABLES` in `indop/ground_truth.py`.
* **Other nodes:** add each B210's (K, NF) calibration constants to `CALIBRATION_TABLES` once they are measured. Until then the runner falls back to the self-calibrated GT $\mu+3\sigma$, which is the approximation the paper's Table 3 uses.

## 4. Comparing with OpenZMS Figure 7 (extension 3)

1. Export the DST proxel occupancy for the session window to CSV with columns `lat,lon,occupancy`.
2. Run `python scripts/run_spatial_map.py --dst-csv dst.csv --npz-dir results/npz`.
3. Report the per-node differences, and the map MAE against the physics-GT node values using leave-one-node-out.
