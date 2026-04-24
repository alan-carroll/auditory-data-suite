"""Runtime defaults for optional acceleration and nested parallelism."""
from __future__ import annotations

import os
import platform


def _positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def available_cpus():
    return max(1, os.cpu_count() or 1)


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def svml_enabled():
    """Return whether Bayesian Bins should ask llvmlite to use SVML.

    SVML is kept off by default because it is not portable across the target
    setups and has previously crashed Ray workers. Users can still opt in with
    ADS_ENABLE_SVML=1 when benchmarking a known-good machine.
    """
    return _truthy(os.environ.get("ADS_ENABLE_SVML", "0"))


def configure_analysis_process_environment():
    """Set safe process-level defaults before Ray/Numba-heavy imports."""
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    os.environ.setdefault("ADS_ENABLE_SVML", "0")
    configure_svml_environment()


def configure_svml_environment():
    """Make ADS_ENABLE_SVML control both explicit and automatic SVML paths."""
    if not svml_enabled():
        os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")


def ray_numba_threads(cpu_count=None):
    """Return the Numba thread count to use inside each Ray worker.

    Subject analysis usually runs many files, so the simple and stable default
    is one Numba thread per Ray task and Ray owns outer parallelism.
    ADS_RAY_NUMBA_THREADS is available for benchmarking or machine-specific
    tuning.
    """
    override = os.environ.get("ADS_RAY_NUMBA_THREADS")
    cpu_count = available_cpus() if cpu_count is None else max(1, cpu_count)
    if not override:
        return 1
    return min(cpu_count, _positive_int(override, 1))


def ray_worker_env_vars(numba_threads):
    """Environment vars Ray should apply before worker imports/JIT work."""
    threads = str(max(1, int(numba_threads)))
    env_vars = {
        "NUMBA_NUM_THREADS": threads,
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "RAY_DEDUP_LOGS": os.environ.get("RAY_DEDUP_LOGS", "0"),
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": os.environ.get(
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0"),
        "ADS_ENABLE_SVML": os.environ.get("ADS_RAY_ENABLE_SVML", "0"),
        "NUMBA_DISABLE_INTEL_SVML": (
            "0" if _truthy(os.environ.get("ADS_RAY_ENABLE_SVML", "0"))
            else "1"
        ),
    }
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath
    return env_vars


def configure_numba_worker_threads(numba_threads):
    """Mask Numba's thread pool in a Ray worker.

    NUMBA_NUM_THREADS sets the maximum pool size at import time; set_num_threads
    masks the active count for already-imported workers.
    """
    threads = max(1, int(numba_threads))
    try:
        import numba as nb
    except ImportError:
        return threads

    try:
        nb.set_num_threads(threads)
    except ValueError:
        max_threads = _positive_int(os.environ.get("NUMBA_NUM_THREADS"),
                                    available_cpus())
        threads = max(1, min(threads, max_threads))
        nb.set_num_threads(threads)
    return threads


def runtime_summary():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": available_cpus(),
        "ray_numba_threads": ray_numba_threads(),
        "svml_enabled": svml_enabled(),
    }
