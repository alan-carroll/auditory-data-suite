"""Runtime defaults for optional acceleration and local parallelism."""
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
    """Return whether Bayesian Bins should ask llvmlite to use SVML."""
    return _truthy(os.environ.get("ADS_ENABLE_SVML", "0"))


def configure_svml_environment():
    """Make ADS_ENABLE_SVML control both explicit and automatic SVML paths."""
    if not svml_enabled():
        os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")


def configure_analysis_process_environment():
    """Set safe process-level defaults before Numba-heavy imports."""
    os.environ.setdefault("ADS_ENABLE_SVML", "0")
    configure_svml_environment()


def worker_numba_threads(cpu_count=None):
    """Return the Numba thread count to use inside each analysis worker.

    Subject analysis usually runs many files, so the stable default is one Numba
    thread per file worker and process-level parallelism owns the outer loop.
    ADS_WORKER_NUMBA_THREADS is available for benchmarking or machine-specific
    tuning.
    """
    override = os.environ.get("ADS_WORKER_NUMBA_THREADS")
    cpu_count = available_cpus() if cpu_count is None else max(1, cpu_count)
    if not override:
        return 1
    return min(cpu_count, _positive_int(override, 1))


def analysis_worker_count(total_tasks, numba_threads=None, cpu_count=None):
    """Return how many local worker processes to run."""
    total_tasks = max(1, int(total_tasks))
    cpu_count = available_cpus() if cpu_count is None else max(1, cpu_count)
    numba_threads = (
        worker_numba_threads(cpu_count=cpu_count)
        if numba_threads is None else max(1, int(numba_threads))
    )
    default_workers = max(1, cpu_count // numba_threads)
    override = os.environ.get("ADS_ANALYSIS_WORKERS")
    if override:
        default_workers = _positive_int(override, default_workers)
    return max(1, min(total_tasks, default_workers))


def worker_env_vars(numba_threads):
    """Environment vars analysis workers should inherit before imports/JIT."""
    threads = str(max(1, int(numba_threads)))
    return {
        "NUMBA_NUM_THREADS": threads,
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "ADS_ENABLE_SVML": os.environ.get("ADS_WORKER_ENABLE_SVML", "0"),
        "NUMBA_DISABLE_INTEL_SVML": (
            "0" if _truthy(os.environ.get("ADS_WORKER_ENABLE_SVML", "0"))
            else "1"
        ),
    }


def configure_numba_threads(numba_threads):
    """Mask Numba's thread pool in a worker process."""
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


def configure_worker_process(numba_threads):
    """Apply worker process env/thread settings."""
    os.environ.update(worker_env_vars(numba_threads))
    return configure_numba_threads(numba_threads)


def runtime_summary():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": available_cpus(),
        "worker_processes": analysis_worker_count(10),
        "worker_numba_threads": worker_numba_threads(),
        "svml_enabled": svml_enabled(),
    }
