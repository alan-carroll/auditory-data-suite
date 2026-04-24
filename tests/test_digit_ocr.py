import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from digit_ocr import DigitOCR
from subject_analysis import extract_number_crops


ROOT = Path(__file__).resolve().parents[1]


class DigitOCRBootstrapTests(unittest.TestCase):
    def test_bootstrap_demo_digits_does_not_skip_three(self):
        numbers_image = cv2.imread(
            str(ROOT / "demo" / "img" / "demo_num.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        mask_image = cv2.imread(
            str(ROOT / "demo" / "img" / "demo_msk.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        _, crops = extract_number_crops(numbers_image, mask_image)
        labels = iter(["4", "9", "2", "1", "6", "5", "0", "7", "3", "8"])

        with redirect_stdout(StringIO()), patch.object(plt, "show"), patch(
            "builtins.input",
            side_effect=lambda _prompt: next(labels),
        ):
            ocr = DigitOCR.bootstrap(crops)

        missing = [
            digit for digit, template in ocr._working_templates.items()
            if template is None
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
