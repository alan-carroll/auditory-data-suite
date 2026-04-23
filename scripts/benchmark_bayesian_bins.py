#!/usr/bin/env python3
"""Benchmark bayesian_bins against frozen demo PSTHs."""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_DEMO_JSON = ROOT / "demo" / "output" / "analyzed_demo.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-json", type=Path, default=DEFAULT_DEMO_JSON)
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode", choices=("both", "sdf", "no-sdf"), default="both")
    parser.add_argument(
        "--ref", action="append", default=[],
        help="Git ref to compare against, e.g. HEAD or main. May repeat.")
    parser.add_argument("--skip-latency-check", action="store_true")
    return parser.parse_args()


def load_demo_sites(demo_json):
    obj = json.loads(demo_json.read_text())
    metadata = next(iter(obj["metadata"].values()))
    n_sweeps = metadata["project_configuration"]["densetc_num_tones"]
    frozen_id = next(
        value["_id"]
        for value in obj["analysis_metadata"].values()
        if value.get("frozen")
    )
    docs = next(
        value
        for key, value in obj.items()
        if "densetc_analysis" in key and "IC" not in key
    )
    sites = sorted(
        (doc for doc in docs.values() if doc.get("analysis_id") == frozen_id),
        key=lambda doc: doc["number"],
    )
    return sites, n_sweeps


def load_module(label):
    if label == "worktree":
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        return importlib.import_module("bayesian_bins")

    source = subprocess.check_output(
        ["git", "show", f"{label}:src/bayesian_bins.py"],
        cwd=ROOT,
        text=True,
    )
    module = types.ModuleType(f"bayesian_bins_{label.replace('/', '_')}")
    module.__file__ = f"{label}:src/bayesian_bins.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def analyze_site(module, site, n_sweeps, return_sdf):
    return module.analyze_psth(
        np.array(site["psth"], dtype=np.int64),
        n_sweeps,
        site["spont_firing_rate_hz"],
        max_t=250,
        max_m=10,
        lat_start=1,
        lat_end=150,
        l_bound=4,
        u_bound=10,
        return_sdf=return_sdf,
    )


def get_densetc_lats_from_result(site, result):
    psth = np.array(site["psth"], dtype=np.int64)
    sdf = result["sdf"]
    lats = np.nan_to_num(result["lats"][1:], nan=0.0, posinf=1.0,
                         neginf=0.0)
    max_prob = float(np.amax(lats))
    onset = np.where(0.15 <= lats)[0]
    if onset.any():
        onset = int(onset[0] + 1)
    else:
        onset = int(np.argmax(lats) + 1)

    total_prob = float(np.nan_to_num(result["total_prob"], nan=0.0,
                                     posinf=1.0, neginf=0.0))
    if (total_prob < 0.2) or (max_prob < 0.1):
        return 50, None, 300

    d_sdf = np.diff(sdf)
    d_norm_sdf = 2.0 * (d_sdf - np.min(d_sdf)) / np.ptp(d_sdf) - 1
    norm_mean = np.mean(d_norm_sdf)
    norm_std = np.std(d_norm_sdf)
    equals_mean = np.isclose(d_norm_sdf[onset:], norm_mean, atol=1e-2)
    offsets = np.where(d_norm_sdf[onset:] < (norm_mean - norm_std))[0]
    if offsets.any():
        potential_offsets = np.where(equals_mean[offsets[0]:] == 1)[0]
        if potential_offsets.any():
            seqs = 1 + np.where(np.diff(potential_offsets) != 1)[0]
            offset_seqs = np.split(potential_offsets, seqs)
            passing_offsets = np.where(
                np.array([len(x) for x in offset_seqs]) >= 10)[0]
            if passing_offsets.any():
                offset = int(
                    offset_seqs[passing_offsets[0]][0] + offsets[0] + onset)
            else:
                offset = int(offset_seqs[-1][0] + offsets[0] + onset)
        else:
            offset = int(offsets[0] + onset)
    else:
        offset = 300

    peak = int(np.argmax(psth[onset:offset])) + onset
    return onset, peak, offset


def check_latencies(module, sites, n_sweeps):
    matches = 0
    for site in sites:
        result = analyze_site(module, site, n_sweeps, return_sdf=True)
        got = get_densetc_lats_from_result(site, result)
        expected = (site["onset_ms"], site["peak_ms"], site["offset_ms"])
        if got == expected and np.isfinite(result["lats"]).all():
            matches += 1
    return matches


def bench(module, sites, n_sweeps, return_sdf, repeats):
    analyze_site(module, sites[0], n_sweeps, return_sdf)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for site in sites:
            analyze_site(module, site, n_sweeps, return_sdf)
        times.append(time.perf_counter() - start)
    return times


def main():
    args = parse_args()
    sites, n_sweeps = load_demo_sites(args.demo_json)
    sites = sites[:args.sites]
    labels = ["worktree", *args.ref]
    modes = []
    if args.mode in ("both", "no-sdf"):
        modes.append(("no_sdf", False))
    if args.mode in ("both", "sdf"):
        modes.append(("sdf", True))

    print(f"sites={len(sites)} repeats={args.repeats} n_sweeps={n_sweeps}")
    for label in labels:
        module = load_module(label)
        print(f"\n[{label}]")
        if not args.skip_latency_check:
            matches = check_latencies(module, sites, n_sweeps)
            print(f"latency_matches={matches}/{len(sites)}")
        for mode_label, return_sdf in modes:
            times = bench(module, sites, n_sweeps, return_sdf, args.repeats)
            median = statistics.median(times)
            per_site = median / len(sites)
            rounded = ", ".join(f"{value:.3f}" for value in times)
            print(
                f"{mode_label}: median={median:.3f}s "
                f"per_site={per_site:.3f}s runs=[{rounded}]")


if __name__ == "__main__":
    main()
