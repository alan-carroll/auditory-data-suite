import os
import unittest
from unittest.mock import patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_worker_numba_threads_defaults_to_one(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_config.worker_numba_threads(cpu_count=12),
                             1)

    def test_worker_numba_thread_override_is_clamped_to_cpu_count(self):
        with patch.dict(os.environ, {"ADS_WORKER_NUMBA_THREADS": "99"},
                        clear=True):
            self.assertEqual(
                runtime_config.worker_numba_threads(cpu_count=12), 12)

    def test_analysis_worker_count_respects_thread_count(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                runtime_config.analysis_worker_count(
                    20, numba_threads=1, cpu_count=12),
                12)
            self.assertEqual(
                runtime_config.analysis_worker_count(
                    20, numba_threads=4, cpu_count=12),
                3)

    def test_svml_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(runtime_config.svml_enabled())
        with patch.dict(os.environ, {"ADS_ENABLE_SVML": "1"}, clear=True):
            self.assertTrue(runtime_config.svml_enabled())

    def test_svml_config_disables_numba_auto_svml_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime_config.configure_analysis_process_environment()
            self.assertEqual(os.environ["ADS_ENABLE_SVML"], "0")
            self.assertEqual(os.environ["NUMBA_DISABLE_INTEL_SVML"], "1")


if __name__ == "__main__":
    unittest.main()
