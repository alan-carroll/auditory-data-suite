#!/usr/bin/env python3
"""Benchmark bayesian_bins against demo PSTHs and saved latency analyses."""
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
    parser.add_argument(
        "--analysis", default="frozen",
        help="Analysis target: frozen, manual, analysis _id, metadata key, or name.")
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode", choices=("both", "sdf", "no-sdf"), default="both")
    parser.add_argument(
        "--ref", action="append", default=[],
        help="Git ref to compare against, e.g. HEAD or main. May repeat.")
    parser.add_argument("--skip-latency-check", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--show-mismatches", type=int, default=0)
    parser.add_argument("--max-t", type=int, default=250)
    parser.add_argument("--max-m", type=int, default=10)
    parser.add_argument("--lat-start", type=int, default=1)
    parser.add_argument("--lat-end", type=int, default=150)
    parser.add_argument("--l-bound", type=int, default=4)
    parser.add_argument("--u-bound", type=int)
    parser.add_argument(
        "--signal-bounds", choices=("densetc", "cxx", "custom"),
        default="densetc")
    parser.add_argument("--min-sig-bound", type=float)
    parser.add_argument("--max-sig-bound", type=float)
    parser.add_argument("--offset-mean-atol", type=float, default=1e-2)
    parser.add_argument("--offset-std-scale", type=float, default=1.0)
    parser.add_argument("--offset-min-run", type=int, default=10)
    parser.add_argument("--offset-adjust-ms", type=int, default=0)
    parser.add_argument("--sweep-offset", action="store_true")
    parser.add_argument("--sweep-min-run-range", default="1:15")
    parser.add_argument("--sweep-adjust-range", default="-15:5")
    parser.add_argument("--sweep-top", type=int, default=15)
    parser.add_argument("--expect-latency-matches", type=int)
    parser.add_argument("--expect-onset-matches", type=int)
    parser.add_argument("--expect-peak-matches", type=int)
    parser.add_argument("--expect-offset-matches", type=int)
    parser.add_argument("--fit-priors", action="store_true")
    parser.add_argument("--prior-fit-maxiter", type=int, default=200)
    parser.add_argument("--initial-sigma", type=float)
    parser.add_argument("--initial-gamma", type=float)
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


def load_demo_sites(demo_json, analysis):
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
    )
    return sites, n_sweeps, analysis_id


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


def densetc_signal_bounds(spont):
    if spont < 25:
        return 0.001, 0.025
    if spont < 50:
        return 0.025, 0.050
    if spont < 100:
        return 0.050, 0.100
    return 0.100, 0.150


def get_signal_bounds(args, spont):
    if args.signal_bounds == "densetc":
        return densetc_signal_bounds(spont)
    if args.signal_bounds == "cxx":
        return 0.0, 0.1
    if args.min_sig_bound is None or args.max_sig_bound is None:
        raise ValueError(
            "--signal-bounds custom requires --min-sig-bound and "
            "--max-sig-bound")
    return args.min_sig_bound, args.max_sig_bound


def analyze_site(module, site, n_sweeps, return_sdf, args):
    min_sig_bound, max_sig_bound = get_signal_bounds(
        args, site["spont_firing_rate_hz"])
    u_bound = args.u_bound if args.u_bound is not None else args.max_m
    prior_fit_options = {"maxiter": args.prior_fit_maxiter}
    if args.initial_sigma is not None:
        prior_fit_options["initial_sigma"] = args.initial_sigma
    if args.initial_gamma is not None:
        prior_fit_options["initial_gamma"] = args.initial_gamma

    return module.analyze_psth(
        np.array(site["psth"], dtype=np.int64),
        n_sweeps,
        site["spont_firing_rate_hz"],
        max_t=args.max_t,
        max_m=args.max_m,
        lat_start=args.lat_start,
        lat_end=args.lat_end,
        l_bound=args.l_bound,
        u_bound=u_bound,
        min_sig_bound=min_sig_bound,
        max_sig_bound=max_sig_bound,
        return_sdf=return_sdf,
        fit_priors=args.fit_priors,
        prior_fit_options=prior_fit_options,
    )


def get_densetc_offset(sdf, onset, args, default_offset=300):
    d_sdf = np.diff(sdf)
    d_sdf_range = np.ptp(d_sdf)
    if d_sdf_range == 0:
        return default_offset

    d_norm_sdf = 2.0 * (d_sdf - np.min(d_sdf)) / d_sdf_range - 1
    norm_mean = np.mean(d_norm_sdf)
    norm_std = np.std(d_norm_sdf)
    equals_mean = np.isclose(d_norm_sdf[onset:], norm_mean,
                             atol=args.offset_mean_atol)
    offsets = np.where(
        d_norm_sdf[onset:] <
        (norm_mean - args.offset_std_scale * norm_std))[0]
    if offsets.any():
        potential_offsets = np.where(equals_mean[offsets[0]:] == 1)[0]
        if potential_offsets.any():
            seqs = 1 + np.where(np.diff(potential_offsets) != 1)[0]
            offset_seqs = np.split(potential_offsets, seqs)
            passing_offsets = np.where(
                np.array([len(x) for x in offset_seqs]) >=
                args.offset_min_run)[0]
            if passing_offsets.any():
                offset = int(
                    offset_seqs[passing_offsets[0]][0] + offsets[0] + onset)
            else:
                offset = int(offset_seqs[-1][0] + offsets[0] + onset)
        else:
            offset = int(offsets[0] + onset)
    else:
        offset = default_offset

    if offset != default_offset and args.offset_adjust_ms:
        offset = max(onset + 1, offset + args.offset_adjust_ms)
        offset = min(default_offset, offset)
    return offset


def get_densetc_lats_from_result(site, result, lat_start, args):
    psth = np.array(site["psth"], dtype=np.int64)
    sdf = result["sdf"]
    lats = np.nan_to_num(result["lats"][lat_start:], nan=0.0, posinf=1.0,
                         neginf=0.0)
    max_prob = float(np.amax(lats))
    onset = np.where(0.15 <= lats)[0]
    if onset.any():
        onset = int(onset[0] + lat_start)
    else:
        onset = int(np.argmax(lats) + lat_start)

    total_prob = float(np.nan_to_num(result["total_prob"], nan=0.0,
                                     posinf=1.0, neginf=0.0))
    if (total_prob < 0.2) or (max_prob < 0.1):
        return 50, None, 300

    offset = get_densetc_offset(sdf, onset, args)
    peak = int(np.argmax(psth[onset:offset])) + onset
    return onset, peak, offset


def score_latencies(site_results, args):
    matches = 0
    mismatches = []
    component_matches = {"onset": 0, "peak": 0, "offset": 0}
    abs_errors = {"onset": [], "peak": [], "offset": []}
    for site, result in site_results:
        got = get_densetc_lats_from_result(
            site, result, args.lat_start, args)
        expected = (site["onset_ms"], site["peak_ms"], site["offset_ms"])
        for idx, key in enumerate(("onset", "peak", "offset")):
            if got[idx] == expected[idx]:
                component_matches[key] += 1
            if got[idx] is not None and expected[idx] is not None:
                abs_errors[key].append(abs(got[idx] - expected[idx]))
        if got == expected and np.isfinite(result["lats"]).all():
            matches += 1
        else:
            mismatches.append((site["number"], expected, got))
    return matches, mismatches, component_matches, abs_errors


def check_latencies(module, sites, n_sweeps, args):
    site_results = [
        (site, analyze_site(module, site, n_sweeps, return_sdf=True,
                            args=args))
        for site in sites
    ]
    return score_latencies(site_results, args)


def parse_int_range(value):
    start, end = (int(part) for part in value.split(":", maxsplit=1))
    step = 1 if start <= end else -1
    return range(start, end + step, step)


def sweep_offsets(module, sites, n_sweeps, args):
    site_results = [
        (site, analyze_site(module, site, n_sweeps, return_sdf=True,
                            args=args))
        for site in sites
    ]
    rows = []
    for offset_min_run in parse_int_range(args.sweep_min_run_range):
        for offset_adjust_ms in parse_int_range(args.sweep_adjust_range):
            sweep_args = types.SimpleNamespace(**vars(args))
            sweep_args.offset_min_run = offset_min_run
            sweep_args.offset_adjust_ms = offset_adjust_ms
            score = score_latencies(site_results, sweep_args)
            matches, _, component_matches, abs_errors = score
            rows.append({
                "offset_min_run": offset_min_run,
                "offset_adjust_ms": offset_adjust_ms,
                "matches": matches,
                "offset_matches": component_matches["offset"],
                "onset_matches": component_matches["onset"],
                "peak_matches": component_matches["peak"],
                "offset_mae": mean(abs_errors["offset"]),
                "score": score,
            })

    rows.sort(key=lambda row: (
        -row["matches"], row["offset_mae"], -row["offset_matches"],
        abs(row["offset_adjust_ms"]), row["offset_min_run"]))
    return rows


def mean(values):
    if not values:
        return float("nan")
    return statistics.mean(values)


def check_expectation(label, actual, expected):
    if expected is None:
        return
    if actual != expected:
        raise SystemExit(
            f"{label} expected {expected}, got {actual}")


def bench(module, sites, n_sweeps, return_sdf, repeats, args):
    analyze_site(module, sites[0], n_sweeps, return_sdf, args)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for site in sites:
            analyze_site(module, site, n_sweeps, return_sdf, args)
        times.append(time.perf_counter() - start)
    return times


def main():
    args = parse_args()
    sites, n_sweeps, analysis_id = load_demo_sites(args.demo_json,
                                                   args.analysis)
    sites = sites[:args.sites]
    labels = ["worktree", *args.ref]
    modes = []
    if args.mode in ("both", "no-sdf"):
        modes.append(("no_sdf", False))
    if args.mode in ("both", "sdf"):
        modes.append(("sdf", True))

    run_desc = f"sites={len(sites)} n_sweeps={n_sweeps}"
    if args.check_only:
        run_desc += " check_only=True"
    else:
        run_desc += f" repeats={args.repeats}"
    print(
        f"analysis={args.analysis} analysis_id={analysis_id} {run_desc}")
    for label in labels:
        module = load_module(label)
        print(f"\n[{label}]")
        if args.sweep_offset:
            rows = sweep_offsets(module, sites, n_sweeps, args)
            for row in rows[:args.sweep_top]:
                print(
                    "offset_sweep "
                    f"min_run={row['offset_min_run']} "
                    f"adjust_ms={row['offset_adjust_ms']} "
                    f"latency_matches={row['matches']}/{len(sites)} "
                    f"component_matches="
                    f"onset:{row['onset_matches']}/{len(sites)} "
                    f"peak:{row['peak_matches']}/{len(sites)} "
                    f"offset:{row['offset_matches']}/{len(sites)} "
                    f"offset_mae={row['offset_mae']:.2f}")
            continue
        if not args.skip_latency_check:
            matches, mismatches, component_matches, abs_errors = (
                check_latencies(module, sites, n_sweeps, args)
            )
            print(f"latency_matches={matches}/{len(sites)}")
            print(
                "component_matches="
                f"onset:{component_matches['onset']}/{len(sites)} "
                f"peak:{component_matches['peak']}/{len(sites)} "
                f"offset:{component_matches['offset']}/{len(sites)}")
            print(
                "mean_abs_ms="
                f"onset:{mean(abs_errors['onset']):.2f} "
                f"peak:{mean(abs_errors['peak']):.2f} "
                f"offset:{mean(abs_errors['offset']):.2f}")
            check_expectation(
                "latency_matches", matches, args.expect_latency_matches)
            check_expectation(
                "onset_matches", component_matches["onset"],
                args.expect_onset_matches)
            check_expectation(
                "peak_matches", component_matches["peak"],
                args.expect_peak_matches)
            check_expectation(
                "offset_matches", component_matches["offset"],
                args.expect_offset_matches)
            if args.show_mismatches:
                for number, expected, got in mismatches[:args.show_mismatches]:
                    print(
                        f"mismatch number={number} expected={expected} "
                        f"got={got}")
        if args.check_only:
            continue
        for mode_label, return_sdf in modes:
            times = bench(module, sites, n_sweeps, return_sdf, args.repeats,
                          args)
            median = statistics.median(times)
            per_site = median / len(sites)
            rounded = ", ".join(f"{value:.3f}" for value in times)
            print(
                f"{mode_label}: median={median:.3f}s "
                f"per_site={per_site:.3f}s runs=[{rounded}]")


if __name__ == "__main__":
    main()
