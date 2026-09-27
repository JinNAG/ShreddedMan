"""Opt-in acceptance checks against independently checked photo samples.

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
from detect_paper import detect_submission
from normalize_strips import normalize_submission
from submission import create_submission


class PhotoSortingTests(unittest.TestCase):
    def assert_photo_pair_recovers_checked_text_order(self, photos):
        if not all(photo.is_file() for photo in photos) or not shutil.which("tesseract"):
            self.skipTest("Both source photos and Tesseract are required")

        # Checked independently against a readable capture of the same strips.
        # Sparse strips are omitted because their positions are not established.
        expected = [24, 17, 9, 7, 25, 20, 4, 6, 14, 18, 1, 5, 3, 11,
                    2, 15, 16, 21, 12, 10, 8, 27, 29, 31, 30, 26, 32]
        with tempfile.TemporaryDirectory() as directory:
            submission = create_submission(photos, Path(directory))
            detect_submission(submission.directory)
            normalize_submission(submission.directory)
            report = sort_strips(submission.normalized_strips, submission.final_document,
                                 ocr_mode="required")
            checked = set(expected)
            actual = [int(entry["source"][5:-4]) for entry in report["order"]
                      if entry["position"] is not None and int(entry["source"][5:-4]) in checked]
            self.assertEqual(actual, expected)

    @unittest.skipUnless(os.environ.get("SHREDDEDMAN_PHOTO_REGRESSIONS") == "1", "Opt-in local photo regression")
    def test_photo8_pair_recovers_checked_text_strip_order(self):
        source = Path(__file__).resolve().parents[1] / "img/error_images"
        self.assert_photo_pair_recovers_checked_text_order(
            [source / "photo8_1.jpg", source / "photo8_2.jpg"])

    @unittest.skipUnless(os.environ.get("SHREDDEDMAN_PHOTO_REGRESSIONS") == "1", "Opt-in local photo regression")
    def test_photo9_pair_recovers_checked_text_strip_order(self):
        source = Path(__file__).resolve().parents[1] / "img/good_images/4"
        self.assert_photo_pair_recovers_checked_text_order(
            [source / "photo9_1.jpg", source / "photo9_2.jpg"])

    @unittest.skipUnless(os.environ.get("SHREDDEDMAN_PHOTO_REGRESSIONS") == "1", "Opt-in local photo regression")
    def test_uneven_photo_heights_recover_the_checked_page_order(self):
        source = Path(__file__).resolve().parents[1] / "img/69c263930fa84e2daa99720b9f3eb93a/normalized_strips"
        if not source.is_dir() or not shutil.which("tesseract"):
            self.skipTest("Local sample and Tesseract are required")
        # Matched to the independently checked capture of the same physical
        # strips using hundreds of SIFT correspondences per text-bearing piece.
        expected = [24, 17, 9, 7, 25, 20, 4, 6, 14, 18, 1, 5, 3, 11,
                    2, 15, 16, 21, 12, 10, 8, 27, 29, 31, 30, 26, 32, 34]
        with tempfile.TemporaryDirectory() as directory:
            report = sort_strips(source, Path(directory) / "final_document", ocr_mode="required")
            actual = [int(entry["source"][5:-4]) for entry in report["order"]
                      if entry["position"] is not None]
            self.assertEqual([number for number in actual if number in expected], expected)
            self.assertTrue(report["matching"]["vertical_spacing_preserved"])
            self.assertTrue(report["review_required"])
            self.assertGreater(report["verification"]["document_check"]["repair_candidates_checked"], 0)

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
