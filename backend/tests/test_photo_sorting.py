"""Opt-in acceptance check against the manually checked carpet-photo sample.

Run with SHREDDEDMAN_PHOTO_REGRESSIONS=1. The large user-provided images and
Tesseract are deliberately not requirements for the fast unit test suite.
"""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sort_strips import sort_strips


class PhotoSortingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("SHREDDEDMAN_PHOTO_REGRESSIONS") == "1", "Opt-in local photo regression")
    def test_carpet_photo_order_and_complete_page_verification(self):
        source = Path(__file__).resolve().parents[1] / "img/351eb45f764d40c2b9c9682366f479ed/normalized_strips"
        if not source.is_dir() or not shutil.which("tesseract"):
            self.skipTest("Local sample and Tesseract are required")
        # Independently checked from intact headings and multiple text lines;
        # never supplied to the production sorter or used as an OCR prompt.
        expected = [14, 16, 31, 29, 18, 13, 15, 24, 19, 25, 23, 26, 34, 27,
                    33, 21, 28, 2, 22, 30, 32, 4, 9, 10, 8, 5, 7, 12]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final_document"
            report = sort_strips(source, output, ocr_mode="required")
            actual = [entry["source"] for entry in report["order"] if entry["position"] is not None]
            self.assertEqual(actual, [f"strip{number}.png" for number in expected])
            self.assertEqual(report["verification"]["ocr_pairs_checked"], 28 * 27)
            check = report["verification"]["document_check"]
            self.assertTrue(check["changed_from_pairwise"])
            self.assertGreater(check["final_score"], check["baseline_score"])
            self.assertEqual(check["selected_order"], actual)
            evidence = json.loads((output / "join_report.json").read_text())
            self.assertEqual(len(evidence["joins"]), 27)
            self.assertTrue(all("full_page_evidence" in item for item in evidence["joins"]))
            self.assertTrue((output / "document.png").is_file())
            print("Uncached photo sorting timings:", report["sorting_timings_seconds"], flush=True)


if __name__ == "__main__":
    unittest.main()
