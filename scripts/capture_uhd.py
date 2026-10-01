"""Capture raw IQ on a POWDER node (USRP B210) and stream it into INDOP.

Runs ON the node (Intel NUC + USB3 B210). Requires the UHD Python API
(`import uhd`, shipped with UHD >= 3.15 / the POWDER UHD images).

Two modes:
  --to-file  write fc32 IQ to disk (then run run_multinode_campaign.py --iq FILE)
  default    stream: compute the 100 ms power timeline on the fly and keep a
             short IQ snippet for SCF, so the 5.4 GB raw capture never hits disk.

Usage (on node):
  python scripts/capture_uhd.py --node Web --minutes 20 --out results/npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indop import INDOPConfig, run_indop  # noqa: E402
from indop.power import chunk_size  # noqa: E402


def open_usrp(args):
    import uhd  # imported lazily: only available on the radio host

    usrp = uhd.usrp.MultiUSRP(args.device_args)
    usrp.set_rx_rate(args.rate, 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(args.freq), 0)
    usrp.set_rx_gain(args.gain, 0)
    usrp.set_rx_antenna(args.antenna, 0)
    st = uhd.usrp.StreamArgs("fc32", "sc16")
    st.channels = [0]
    streamer = usrp.get_rx_stream(st)
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)
    return uhd, usrp, streamer


def capture(args) -> None:
    uhd, usrp, streamer = open_usrp(args)
    n_chunk = chunk_size(args.rate, 0.1)
    total_chunks = int(args.minutes * 60 / 0.1)
    md = uhd.types.RXMetadata()
    buf = np.zeros(streamer.get_max_num_samps(), dtype=np.complex64)
    chunk = np.empty(n_chunk, dtype=np.complex64)
    power = np.empty(total_chunks)
    scf_iq = np.empty(int(args.scf_seconds * args.rate), dtype=np.complex64)
    fout = open(args.to_file, "wb") if args.to_file else None
    filled, c, scf_filled, overflows = 0, 0, 0, 0
    started = dt.datetime.now(dt.timezone.utc)
    while c < total_chunks:
        n = streamer.recv(buf, md, 1.0)
        if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
            overflows += 1
        elif md.error_code != uhd.types.RXMetadataErrorCode.none:
            raise RuntimeError(md.strerror())
        i = 0
        while i < n and c < total_chunks:
            take = min(n - i, n_chunk - filled)
            chunk[filled:filled + take] = buf[i:i + take]
            filled += take
            i += take
            if filled == n_chunk:
                power[c] = 10 * np.log10(np.mean(np.abs(chunk) ** 2) + 1e-30) - 10 * np.log10(args.rate)
                if scf_filled < len(scf_iq):
                    k = min(n_chunk, len(scf_iq) - scf_filled)
                    scf_iq[scf_filled:scf_filled + k] = chunk[:k]
                    scf_filled += k
                if fout:
                    chunk.tofile(fout)
                c += 1
                filled = 0
    streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    if fout:
        fout.close()

    meta = {"freq_hz": args.freq, "gain_db": args.gain, "rate": args.rate,
            "started_utc": started.isoformat(), "overflows": overflows}
    res = run_indop(power, args.node, INDOPConfig(), iq_for_scf=scf_iq[:scf_filled],
                    sample_rate=args.rate, meta=meta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.session or started.strftime("%H")
    res.save_npz(str(out / f"{args.node}_{tag}.npz"))
    np.save(out / f"{args.node}_{tag}_power.npy", power.astype(np.float32))
    print(res.summary())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--freq", type=float, default=3595e6)
    ap.add_argument("--rate", type=float, default=1e6)
    ap.add_argument("--gain", type=float, default=60)
    ap.add_argument("--antenna", default="RX2")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--scf-seconds", type=float, default=2.0)
    ap.add_argument("--device-args", default="type=b200")
    ap.add_argument("--to-file", default=None)
    ap.add_argument("--session", default=None, help="tag, e.g. 05 for the 05:00 session")
    ap.add_argument("--out", default="results/npz")
    capture(ap.parse_args())
