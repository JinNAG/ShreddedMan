from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from join_report import confidence_for_join
from join_verification import JoinVerifier, measure_seam, refine_orders
from strip_ocr import TesseractOCR


class JoinVerificationTests(unittest.TestCase):
    def test_composed_alignment_does_not_repeat_text_at_strip_tips(self):
        images = [np.full((50, 20, 4), 255, np.uint8) for _ in range(3)]
        images[2][-1, :, :3] = 0
        rows = np.arange(50, dtype=np.float32)
        warps = np.broadcast_to(rows, (3, 3, 50)).copy()
        warps[0, 1] += 10
        verifier = JoinVerifier(images, SimpleNamespace(warps=warps), 50, TesseractOCR("off"))
        ink, seams = verifier.render((0, 1, 2))
        self.assertGreater(ink[39, seams[-1]:].sum(), 0)
        self.assertEqual(ink[40:, seams[-1]:].sum(), 0)

    def test_ignoring_broken_text_cannot_improve_ocr_score(self):
        ink = np.zeros((100, 80), np.float32)
        ink[15:25, 30:50] = 1
        ink[65:75, 30:50] = 1
        incomplete = [{"text": "good", "confidence": 1.0, "box": [25, 10, 30, 20]}]
        complete = [{"text": "first", "confidence": 0.8, "box": [25, 10, 30, 20]},
                    {"text": "second", "confidence": 0.8, "box": [25, 60, 30, 20]}]
        partial, full = measure_seam(ink, 40, incomplete), measure_seam(ink, 40, complete)
        self.assertGreater(partial["ocr_confidence"], full["ocr_confidence"])
        self.assertLess(partial["ocr_score"], full["ocr_score"])
        self.assertAlmostEqual(partial["ink_coverage"], 0.5)
        self.assertEqual(measure_seam(ink, 40, [])["ocr_score"], 0)

    def test_ocr_away_from_the_cut_is_not_join_evidence(self):
        ink = np.zeros((100, 80), np.float32)
        ink[15:25, 30:50] = 1
        words = [{"text": "unrelated", "confidence": 1.0, "box": [0, 10, 20, 20]}]
        evidence = measure_seam(ink, 40, words)
        self.assertEqual(evidence["recognized_tokens"], 0)
        self.assertEqual(evidence["ocr_score"], 0)

    def test_blank_join_never_gets_high_confidence(self):
        evidence = measure_seam(np.zeros((100, 80), np.float32), 40, [])
        result = confidence_for_join(1.0, evidence, 1.0, 1.0, True)
        self.assertEqual(result["level"], "low")
        self.assertLessEqual(result["score"], 39)

    def test_better_competing_neighbor_caps_confidence(self):
        evidence = {"stroke_score": 1, "ocr_score": 1, "text_lines": 20,
                    "recognized_lines": 20, "ink_coverage": 1}
        confidence = confidence_for_join(1, evidence, -0.01, 1, True)
        self.assertEqual(confidence["level"], "low")
        self.assertTrue(any("Another neighbor" in reason for reason in confidence["reasons"]))

    def test_missing_ocr_is_explicit_and_required_mode_fails(self):
        with patch("strip_ocr.shutil.which", return_value=None):
            automatic = TesseractOCR("auto")
            self.assertFalse(automatic.enabled)
            self.assertEqual(automatic.metadata()["status"], "unavailable")
            with self.assertRaisesRegex(ValueError, "Install Tesseract"):
                TesseractOCR("required")

    def test_context_search_can_correct_a_swap(self):
        class Oracle:
            ocr = SimpleNamespace(enabled=True)

            def __init__(self):
                self.cache = {}

            def evaluate_many(self, orders):
                for order in orders:
                    value = 1.0 if tuple(order) in ((0, 2, 1), (2, 1, 3)) else 0.2
                    self.cache[tuple(order)] = {"seams": [{"ocr_score": value}] * 2}

        scores = np.zeros((4, 4))
        np.fill_diagonal(scores, -np.inf)
        verifier = Oracle()
        ranked, metadata = refine_orders([(0, (0, 1, 2, 3))], scores, verifier)
        self.assertEqual(ranked[0][1], (0, 2, 1, 3))
        self.assertEqual(metadata["accepted_changes"][0]["positions"], [2, 3])
        self.assertFalse(metadata["globally_optimal"])

    def test_swap_that_damages_other_joins_is_rejected(self):
        class Oracle:
            ocr = SimpleNamespace(enabled=True)

            def __init__(self):
                self.cache = {}

            def evaluate_many(self, orders):
                for order in orders:
                    value = {(0, 1, 2): 0.7, (1, 2, 3): 0.7, (0, 2, 1): 1.0}.get(tuple(order), 0.0)
                    self.cache[tuple(order)] = {"seams": [{"ocr_score": value}] * 2}

        scores = np.zeros((4, 4))
        np.fill_diagonal(scores, -np.inf)
        ranked, metadata = refine_orders([(0, (0, 1, 2, 3))], scores, Oracle())
        self.assertEqual(ranked[0][1], (0, 1, 2, 3))
        self.assertEqual(metadata["accepted_changes"], [])

    @unittest.skipUnless(shutil.which("tesseract"), "Tesseract is optional outside OCR integration tests")
    def test_real_ocr_and_cache_invalidation(self):
        image = np.full((90, 440), 255, np.uint8)
        cv2.putText(image, "HELLO WORLD", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 0, 3, cv2.LINE_AA)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            engine = TesseractOCR("required", cache_path=cache)
            words = engine.recognize(image)
            self.assertIn("HELLO", " ".join(word["text"] for word in words))
            self.assertEqual(engine.recognize(image), words)
            self.assertEqual(engine.calls, 1)
            engine.flush()
            reloaded = TesseractOCR("required", cache_path=cache)
            self.assertEqual(reloaded.recognize(image), words)
            self.assertEqual(reloaded.calls, 0)
            changed = image.copy()
            changed[0, 0] = 0
            reloaded.recognize(changed)
            self.assertEqual(reloaded.calls, 1)


if __name__ == "__main__":
    unittest.main()
