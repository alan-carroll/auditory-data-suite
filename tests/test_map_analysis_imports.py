import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class MapAnalysisImportTests(unittest.TestCase):
    def test_entrypoint_import_does_not_load_heavy_modules(self):
        src_dir = Path(__file__).resolve().parents[1] / "src"
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            str(src_dir)
            if not env.get("PYTHONPATH")
            else f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
        )
        code = """
import json
import sys
import map_analysis
mods = ["analysis_functions", "subject_analysis", "pandas", "matplotlib"]
print(json.dumps({name: name in sys.modules for name in mods}))
"""

        output = subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=src_dir.parent,
            env=env,
            text=True,
        )

        self.assertEqual(
            json.loads(output),
            {
                "analysis_functions": False,
                "subject_analysis": False,
                "pandas": False,
                "matplotlib": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
