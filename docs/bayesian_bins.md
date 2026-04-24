# Bayesian Bins Notes

This document is for maintainers touching `bayesian_bins.py`, DenseTC
latency detection, or the benchmark/regression tooling around them.

## What Lives Where

`src/bayesian_bins.py` is intended to stay close to the original C++ Bayesian
Bins / SDF algorithm. Its defaults should be generic and C++-ish where possible.

`src/stim_types/densetc.py` owns the auditory-neurophysiology policy choices
that worked best for DenseTC analysis:

- `max_t=250`
- `max_m=10`
- `lat_start=1`
- `lat_end=150`
- `l_bound=4`
- signal separator buckets based on spontaneous firing rate
- the SDF-derived offset heuristic

The split is intentional. If a choice is specific to this project's DenseTC
maps, keep it in `densetc.py`; if it is part of the general algorithm, keep it
in `bayesian_bins.py`.

## C++ Mapping

The old C++ source separates a few concepts that are easy to blur together:

- The beta prior hyperparameters are `efire` and `egap`.
- The optional simplex optimizer is for those prior hyperparameters.
- The latency signal separator is optimized separately with golden-section
  search.
- The C++ separator search uses roughly `0.0..0.1` by default.

In Python naming, `sigma` and `gamma` play the role of beta prior exponents.
DenseTC usually estimates them from spontaneous rate instead of fitting them.

## Current DenseTC Baseline

The current default DenseTC behavior intentionally preserves the frozen demo
auto-analysis:

```bash
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis frozen \
  --sites 72 \
  --check-only
```

Expected result:

```text
latency_matches=72/72
component_matches=onset:72/72 peak:72/72 offset:72/72
```

The same defaults compared with the manually corrected demo analysis currently
produce:

```bash
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis manual \
  --sites 72 \
  --check-only \
  --show-mismatches 10
```

Expected result:

```text
latency_matches=53/72
component_matches=onset:66/72 peak:72/72 offset:56/72
mean_abs_ms=onset:0.17 peak:0.00 offset:2.60
```

Most manual mismatches are offset differences, not peak or Bayesian onset
failures.

## Slow Regression Check

Before changing Bayesian Bins, DenseTC latency defaults, or offset logic, run:

```bash
uv tool run --with 'nox[uv]' nox -s bayesian-bins
```

That nox session runs the frozen and manual demo comparisons with expected
counts. It is intentionally slower than the normal unit tests because it runs
the full SDF path across all 72 demo DenseTC sites.

For a quicker direct run without nox environment setup:

```bash
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis frozen \
  --sites 72 \
  --check-only \
  --expect-latency-matches 72 \
  --expect-onset-matches 72 \
  --expect-peak-matches 72 \
  --expect-offset-matches 72
```

## Timing Benchmark

Use the benchmark script to compare worktree performance against an older ref:

```bash
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis frozen \
  --sites 3 \
  --repeats 2 \
  --ref HEAD~3
```

Useful modes:

```bash
# Time latency-only analysis.
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis frozen \
  --sites 10 \
  --repeats 2 \
  --mode no-sdf \
  --skip-latency-check

# Time the full SDF path.
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis frozen \
  --sites 10 \
  --repeats 1 \
  --mode sdf
```

## Parameter Sweeps

The benchmark script can compare algorithm parameters against either frozen or
manual saved analyses.

Signal separator bounds:

```bash
# DenseTC auditory defaults.
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis manual \
  --sites 72 \
  --check-only \
  --signal-bounds densetc

# C++-style broad separator search.
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis manual \
  --sites 72 \
  --check-only \
  --signal-bounds cxx

# Custom signal separator range.
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis manual \
  --sites 72 \
  --check-only \
  --signal-bounds custom \
  --min-sig-bound 0.001 \
  --max-sig-bound 0.025
```

Offset heuristic sweep:

```bash
./src/.venv/bin/python scripts/benchmark_bayesian_bins.py \
  --analysis manual \
  --sites 72 \
  --check-only \
  --sweep-offset \
  --sweep-top 10
```

Current finding: `offset_min_run=9` slightly improves manual offset MAE, but it
breaks frozen exact matching. Keep the production default at `offset_min_run=10`
unless there is a deliberate decision to prioritize manual-style offsets over
frozen auto-analysis compatibility.

## Known Findings

- Disabling broad `fastmath` on `nb_max_logsumexp` fixed the all-`-inf` to
  `nan` failure mode that broke on newer dependency versions.
- Cached interval evidence and cached signal-dependent beta logs give major
  speedups over previous Python version used.
- Widening DenseTC separator bounds to C++-style `0.0..0.1` made demo matches
  worse.
- Changing DenseTC `l_bound` from `4` to `1` did not improve demo matches.
- Lowering `max_m` hurt demo matches.
- The remaining manual-vs-auto gap is mostly offset cleanup, not Bayesian onset
  or peak detection.

## Future Work

- Add an optional prior-hyperparameter fitter mirroring the C++ simplex path.
- Revisit Ray plus Numba thread settings so outer process parallelism and inner
  Numba parallelism do not oversubscribe CPUs.
- Re-evaluate SVML as an opt-in acceleration path on supported platforms only.