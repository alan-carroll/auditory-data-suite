import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_autosave import GUIAutosave


class GUIAutosaveTests(unittest.TestCase):
    def test_write_load_and_delete_sidecar_next_to_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subject.json"
            autosave = GUIAutosave(db_path, "analysis/1")

            payload = autosave.write(
                overview={12: {"field_assignment": "A1", "marked": True}},
                detail_sites={7: {"onset": 10, "bw_idx": {20: [1, 3]}}},
            )

            self.assertEqual(
                autosave.path,
                Path(tmpdir) / "subject.json.autosave.analysis_1.json")
            self.assertTrue(autosave.path.exists())
            self.assertEqual(payload["overview"]["12"]["field_assignment"], "A1")

            loaded = autosave.load()
            self.assertEqual(loaded["analysis_id"], "analysis/1")
            self.assertEqual(loaded["detail_sites"]["7"]["bw_idx"]["20"], [1, 3])

            autosave.write(overview={}, detail_sites={})
            self.assertFalse(autosave.path.exists())

    def test_load_ignores_other_analysis_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subject.json"
            GUIAutosave(db_path, "other").write(
                overview={1: {"field_assignment": "A1", "marked": False}})

            self.assertIsNone(GUIAutosave(db_path, "analysis").load())

    def test_load_ignores_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            autosave = GUIAutosave(Path(tmpdir) / "subject.json", "analysis")
            autosave.path.write_text("{", encoding="utf-8")

            self.assertIsNone(autosave.load())


if __name__ == "__main__":
    unittest.main()
