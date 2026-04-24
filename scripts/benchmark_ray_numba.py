#!/usr/bin/env python3
"""Benchmark Ray outer parallelism against Numba inner parallelism."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_DEMO_JSON = ROOT / "demo" / "output" / "analyzed_demo.json"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime_config import (  # noqa: E402
    configure_analysis_process_environment,
    configure_numba_worker_threads,
    ray_numba_threads,
    ray_worker_env_vars,
)

configure_analysis_process_environment()

import ray  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-json", type=Path, default=DEFAULT_DEMO_JSON)
    parser.add_argument("--analysis", default="frozen")
    parser.add_argument("--sites", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--threads",
        default="auto,1,2,3,4,6,12",
        help="Comma-separated Numba thread counts, or auto.")
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument(
        "--mode", choices=("sdf", "no-sdf"), default="sdf")
    return parser.parse_args()


def select_analysis_id(analysis_metadata, selector):
    if selector == "frozen":
        return next(
            value["_id"] for value in analysis_metadata.values()
            if value.get("frozen")
        )
    if selector == "manual":
        return next(
            value["_id"] for value in analysis_metadata.values()
            if not value.get("frozen")
        )

    for key, value in analysis_metadata.items():
        if selector in (key, value.get("_id"), value.get("name")):
            return value["_id"]
    raise ValueError(f"Unknown analysis target: {selector}")


def load_demo_sites(demo_json, analysis, site_count):
    obj = json.loads(demo_json.read_text())
    metadata = next(iter(obj["metadata"].values()))
    n_sweeps = metadata["project_configuration"]["densetc_num_tones"]
    analysis_id = select_analysis_id(obj["analysis_metadata"], analysis)
    docs = next(
        value
        for key, value in obj.items()
        if "densetc_analysis" in key and "IC" not in key
    )
    sites = sorted(
        (doc for doc in docs.values() if doc.get("analysis_id") == analysis_id),
        key=lambda doc: doc["number"],
    )[:site_count]
    return sites, n_sweeps, analysis_id


def parse_threads(value, site_count):
    threads = []
    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part == "auto":
            thread_count = ray_numba_threads()
        else:
            thread_count = int(part)
        if thread_count > 0 and thread_count not in threads:
            threads.append(thread_count)
    return threads


@ray.remote
def analyze_site(site, n_sweeps, numba_threads, return_sdf):
    configure_numba_worker_threads(numba_threads)
    import bayesian_bins as bb

    return bb.analyze_psth(
        np.array(site["psth"], dtype=np.int64),
        n_sweeps,
        site["spont_firing_rate_hz"],
        max_t=250,
        max_m=10,
        lat_start=1,
        lat_end=150,
        l_bound=4,
        min_sig_bound=0.001,
        max_sig_bound=0.025,
        return_sdf=return_sdf,
    )["total_prob"]


def run_once(sites, n_sweeps, numba_threads, return_sdf):
    task = analyze_site.options(
        num_cpus=numba_threads,
        runtime_env={"env_vars": ray_worker_env_vars(numba_threads)},
    )
    start = time.perf_counter()
    ray.get([
        task.remote(site, n_sweeps, numba_threads, return_sdf)
        for site in sites
    ])
    return time.perf_counter() - start


def main():
    args = parse_args()
    sites, n_sweeps, analysis_id = load_demo_sites(
        args.demo_json, args.analysis, args.sites)
    thread_counts = parse_threads(args.threads, len(sites))
    return_sdf = args.mode == "sdf"

    ray.init(num_cpus=args.num_cpus, include_dashboard=False,
             ignore_reinit_error=True, log_to_driver=False)
    print(
        f"analysis={args.analysis} analysis_id={analysis_id} "
        f"sites={len(sites)} n_sweeps={n_sweeps} mode={args.mode}"
    )
    for numba_threads in thread_counts:
        # Warm workers/JIT before timing.
        run_once(sites, n_sweeps, numba_threads, return_sdf)
        times = [
            run_once(sites, n_sweeps, numba_threads, return_sdf)
            for _ in range(args.repeats)
        ]
        median = sorted(times)[len(times) // 2]
        runs = ", ".join(f"{value:.3f}" for value in times)
        print(
            f"threads={numba_threads} median={median:.3f}s "
            f"per_site={median / len(sites):.3f}s runs=[{runs}]"
        )
    ray.shutdown()


if __name__ == "__main__":
    main()
