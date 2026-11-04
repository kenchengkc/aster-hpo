"""GPU-free behavior checks for the remote runner's argument/output boundary."""

import pathlib
import subprocess
import tempfile
import unittest

RUNNER = pathlib.Path(__file__).resolve().parents[1] / "run_tests.sh"


class RunnerTests(unittest.TestCase):
    def run_runner(self, *args):
        return subprocess.run(
            ["bash", str(RUNNER), *map(str, args)], text=True, capture_output=True
        )

    def test_help_does_not_require_cuda(self):
        result = self.run_runner("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-dir", result.stdout)

    def test_existing_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary) / "results"
            result_dir.mkdir()
            marker = result_dir / "keep.txt"
            marker.write_text("existing evidence\n")
            result = self.run_runner("--output-dir", result_dir, "--sanitizer", "off")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(marker.read_text(), "existing evidence\n")
            self.assertEqual(list(result_dir.iterdir()), [marker])

    def test_invalid_sanitizer_does_not_create_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary) / "results"
            result = self.run_runner("--output-dir", result_dir, "--sanitizer", "maybe")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--sanitizer", result.stderr)
            self.assertFalse(result_dir.exists())

    def test_unknown_flag_is_rejected(self):
        result = self.run_runner("--not-a-real-option")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown option", result.stderr)


if __name__ == "__main__":
    unittest.main()
