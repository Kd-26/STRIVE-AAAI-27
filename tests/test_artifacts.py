import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityArtifactTests(unittest.TestCase):
    def test_sample_generation_archive(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_generation_artifact.py"),
                str(ROOT / "data" / "sample" / "batch01_generation.zip"),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("trajectories: 600", result.stdout)

    def test_notebooks_are_sanitized_and_structured(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_notebooks.py")],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
