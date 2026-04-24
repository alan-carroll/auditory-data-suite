import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainware import BrainwareSrcIO
from site_model import StimConfig
from stim_types import DENSETC
from subject_analysis import _run_stimulus_file_analysis


ROOT = Path(__file__).resolve().parents[1]


class SubjectAnalysisWorkerTests(unittest.TestCase):
    def test_densetc_worker_reopens_file_from_path(self):
        cfg_dict = json.loads((ROOT / "demo" / "demo_config.json").read_text())
        worker_kwargs = {
            "cfg": StimConfig.from_project_config(cfg_dict),
            "analysis_id": "worker-smoke",
            "final_file_df": None,
            "return_sdf": False,
        }
        file = BrainwareSrcIO(
            filename=str(
                ROOT / "demo" / "data" /
                "DenseTC_singleRP2#001G_RZ5-1_007.src"
            )
        )
        with patch.dict("os.environ", {"ADS_ANALYSIS_WORKERS": "1"}):
            results = _run_stimulus_file_analysis(
                DENSETC, [file], False, [], worker_kwargs)

        self.assertEqual(len(results), 1)
        analysis_doc = results[0]["docs"]["analysis"]
        self.assertEqual(analysis_doc["number"], 1)
        self.assertEqual(analysis_doc["analysis_id"], "worker-smoke")


if __name__ == "__main__":
    unittest.main()
