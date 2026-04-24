#!/usr/bin/env python3
"""Benchmark stdlib process workers against serial Bayesian Bins analysis."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_DEMO_JSON = ROOT / "demo" / "output" / "analyzed_demo.json"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime_config import (  # noqa: E402
    analysis_worker_count,
    configure_analysis_process_environment,
    configure_worker_process,
    worker_env_vars,
    worker_numba_threads,
)

configure_analysis_process_environment()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-json", type=Path, default=DEFAULT_DEMO_JSON)
    parser.add_argument("--analysis", default="frozen")
    parser.add_argument("--sites", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--numba-threads", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--serial", action="store_true")
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


def analyze_site(site, n_sweeps, numba_threads, return_sdf):
    configure_worker_process(numba_threads)
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


def run_serial_once(sites, n_sweeps, numba_threads, return_sdf):
    start = time.perf_counter()
    for site in sites:
        analyze_site(site, n_sweeps, numba_threads, return_sdf)
    return time.perf_counter() - start


def run_process_once(executor, sites, n_sweeps, numba_threads, return_sdf):
    start = time.perf_counter()
    futures = [
        executor.submit(analyze_site, site, n_sweeps, numba_threads,
                        return_sdf)
        for site in sites
    ]
    for future in futures:
        future.result()
    return time.perf_counter() - start


def main():
    args = parse_args()
    sites, n_sweeps, analysis_id = load_demo_sites(
        args.demo_json, args.analysis, args.sites)
    numba_threads = args.numba_threads or worker_numba_threads()
    workers = args.workers or analysis_worker_count(
        len(sites), numba_threads=numba_threads)
    return_sdf = args.mode == "sdf"

    print(
        f"analysis={args.analysis} analysis_id={analysis_id} "
        f"sites={len(sites)} n_sweeps={n_sweeps} mode={args.mode} "
        f"workers={workers} numba_threads={numba_threads} "
        f"serial={args.serial}"
    )

    if args.serial or workers == 1:
        run_serial_once(sites, n_sweeps, numba_threads, return_sdf)
        times = [
            run_serial_once(sites, n_sweeps, numba_threads, return_sdf)
            for _ in range(args.repeats)
        ]
    else:
        os.environ.update(worker_env_vars(numba_threads))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=configure_worker_process,
            initargs=(numba_threads,),
        ) as executor:
            run_process_once(executor, sites, n_sweeps, numba_threads,
                             return_sdf)
            times = [
                run_process_once(executor, sites, n_sweeps, numba_threads,
                                 return_sdf)
                for _ in range(args.repeats)
            ]
    median = sorted(times)[len(times) // 2]
    runs = ", ".join(f"{value:.3f}" for value in times)
    print(
        f"median={median:.3f}s per_site={median / len(sites):.3f}s "
        f"runs=[{runs}]"
    )


if __name__ == "__main__":
    main()
