import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dialogs


class FakeRoot:
    def __init__(self):
        self.idle_updates = 0
        self.updates = 0

    def update_idletasks(self):
        self.idle_updates += 1

    def update(self):
        self.updates += 1


class FakeHiddenRoot(FakeRoot):
    def __init__(self):
        super().__init__()
        self.withdrawn = False
        self.destroyed = False

    def withdraw(self):
        self.withdrawn = True

    def destroy(self):
        self.destroyed = True


class DialogWrapperTests(unittest.TestCase):
    def test_get_file_does_not_parent_to_hidden_root(self):
        hidden_root = FakeHiddenRoot()

        with tempfile.TemporaryDirectory() as tmpdir:
            chosen_path = str(Path(tmpdir) / "map.png")
            with patch.object(dialogs.tk, "Tk", return_value=hidden_root), patch.object(
                dialogs.filedialog,
                "askopenfilename",
                return_value=chosen_path,
            ) as askopen:
                result = dialogs.get_file(title="Pick file")

        askopen.assert_called_once()
        self.assertNotIn("parent", askopen.call_args.kwargs)
        self.assertTrue(hidden_root.withdrawn)
        self.assertTrue(hidden_root.destroyed)
        self.assertEqual(result, chosen_path)

    def test_get_file_reuses_supplied_parent(self):
        parent = FakeRoot()

        with tempfile.TemporaryDirectory() as tmpdir:
            chosen_path = str(Path(tmpdir) / "points.csv")
            with patch.object(dialogs.tk, "Tk") as tk_ctor, patch.object(
                dialogs.filedialog,
                "askopenfilename",
                return_value=chosen_path,
            ) as askopen:
                result = dialogs.get_file(title="Pick file", parent=parent)

        tk_ctor.assert_not_called()
        askopen.assert_called_once()
        self.assertEqual(askopen.call_args.kwargs["parent"], parent)
        self.assertEqual(result, chosen_path)
        self.assertGreaterEqual(parent.idle_updates, 2)
        self.assertGreaterEqual(parent.updates, 2)


if __name__ == "__main__":
    unittest.main()
