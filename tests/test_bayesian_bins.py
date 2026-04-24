import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.makedirs(Path(tempfile.gettempdir()) / "ads-mpl-tests", exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ads-mpl-tests")
)

import bayesian_bins as bb
from stim_types.densetc import get_densetc_bb_lats


class BayesianBinsRegressionTests(unittest.TestCase):
    @staticmethod
    def _load_demo_site(site_number):
        demo_file = (
            Path(__file__).resolve().parent.parent
            / "demo"
            / "output"
            / "analyzed_demo.json"
        )
        obj = json.loads(demo_file.read_text())
        docs = next(
            value
            for key, value in obj.items()
            if "densetc_analysis" in key and "IC" not in key
        )
        return next(
            doc for doc in docs.values() if doc.get("number") == site_number
        )

    def test_nb_max_logsumexp_handles_all_negative_infinity(self):
        value = bb.nb_max_logsumexp(np.array([-np.inf, -np.inf], dtype=np.float64))

        self.assertTrue(math.isinf(value))
        self.assertLess(value, 0)

    def test_demo_site_1_matches_frozen_latency_analysis(self):
        site = self._load_demo_site(1)

        lat = get_densetc_bb_lats(
            np.array(site["psth"], dtype=np.int64),
            n_sweeps=1296,
            spont=site["spont_firing_rate_hz"],
            return_sdf=True,
        )

        self.assertEqual(lat["onset"], site["onset_ms"])
        self.assertEqual(lat["peak"], site["peak_ms"])
        self.assertEqual(lat["offset"], site["offset_ms"])
        self.assertTrue(np.isfinite(lat["lats"]).all())
        self.assertAlmostEqual(lat["max_prob"], site["bb_latency_prob"], places=6)
        self.assertAlmostEqual(lat["total_prob"], site["bb_total_lat_prob"], places=6)

    def test_demo_unknown_site_stays_unknown(self):
        site = self._load_demo_site(39)

        lat = get_densetc_bb_lats(
            np.array(site["psth"], dtype=np.int64),
            n_sweeps=1296,
            spont=site["spont_firing_rate_hz"],
            return_sdf=True,
        )

        self.assertEqual(lat["onset"], site["onset_ms"])
        self.assertIsNone(lat["peak"])
        self.assertEqual(lat["offset"], site["offset_ms"])
        self.assertTrue(np.isfinite(lat["lats"]).all())

    def test_prior_exponent_fit_returns_finite_bounded_values(self):
        site = self._load_demo_site(1)

        fit = bb.fit_prior_exponents(
            np.array(site["psth"], dtype=np.int64),
            n_sweeps=1296,
            max_t=40,
            max_m=3,
            maxiter=5,
        )

        self.assertTrue(np.isfinite(fit["sigma"]))
        self.assertTrue(np.isfinite(fit["gamma"]))
        self.assertGreaterEqual(fit["sigma"], 0.001)
        self.assertLessEqual(fit["sigma"], 300)
        self.assertGreaterEqual(fit["gamma"], 1)
        self.assertLessEqual(fit["gamma"], 300)


if __name__ == "__main__":
    unittest.main()
