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
from join_verification import (JoinVerifier, english_fragments, measure_seam, refine_orders,
                               text_evidence, verify_document_orders)
from strip_ocr import TesseractOCR
from sort_strips import rank_orders


class JoinVerificationTests(unittest.TestCase):
    def test_high_resolution_analysis_rescales_row_maps_without_changing_sources(self):
        height = 4800
        images = [np.full((height, 80, 4), 255, np.uint8) for _ in range(2)]
        warps = np.broadcast_to(np.arange(height, dtype=np.float32), (2, 2, height)).copy()
        warps[0, 1] += 20
        verifier = JoinVerifier(images, SimpleNamespace(warps=warps), height, TesseractOCR("off"))
        self.assertEqual(verifier.height, 2400)
        np.testing.assert_allclose(verifier.warps[0, 1], np.arange(2400) + 10)
        np.testing.assert_array_equal(warps[0, 1], np.arange(height) + 20)
        self.assertEqual(images[0].shape, (4800, 80, 4))
        self.assertEqual(verifier.render((0, 1))[0].shape[0], 2400)

    def test_visual_shortlist_cannot_exclude_readable_neighbors(self):
        size, height = 13, 100
        images = [np.full((height, 20, 4), 255, np.uint8) for _ in range(size)]
        visual = np.full((size, size), 0.6)
        np.fill_diagonal(visual, -np.inf)
        visual[0, 12] = 0.05  # Outside the former ten-neighbor shortlist.
        matches = SimpleNamespace(scores=visual, warps=np.broadcast_to(np.arange(height, dtype=np.float32), (size, size, height)))
        verifier = JoinVerifier(images, matches, height, SimpleNamespace(enabled=True))

        def readings(pairs):
            for pair in pairs:
                verifier.cache[pair] = {"seams": [{"ocr_score": 1.0 if pair == (0, 12) else 0.0}]}

        with patch.object(verifier, "evaluate_many", side_effect=readings):
            scores = verifier.score_pairs()
        self.assertEqual(np.isfinite(scores).sum(), size * (size - 1))
        self.assertEqual(int(np.argmax(scores[0])), 12)

    def test_context_search_can_relocate_an_intact_group(self):
        target = (0, 1, 2, 3, 4, 5)
        correct_triples = {target[i:i + 3] for i in range(4)}

        class Oracle:
            ocr = SimpleNamespace(enabled=True)

            def __init__(self):
                self.cache = {}

            def evaluate_many(self, orders):
                for order in orders:
                    value = 1.0 if tuple(order) in correct_triples else 0.0
                    self.cache[tuple(order)] = {"seams": [{"ocr_score": value}] * 2}

        scores = np.zeros((6, 6))
        np.fill_diagonal(scores, -np.inf)
        for a, b in zip(target, target[1:]):
            scores[a, b] = 0.1
        ranked, metadata = refine_orders([(0, (0, 1, 4, 5, 2, 3))], scores, Oracle())
        self.assertEqual(ranked[0][1], target)
        self.assertEqual(metadata["accepted_changes"][0]["operation"], "move_block")

    def test_full_page_check_rejects_local_improvement_that_breaks_other_text(self):
        baseline, changed = (0, 1, 2, 3), (0, 2, 1, 3)

        class Oracle:
            ocr = SimpleNamespace(enabled=True)
            workers = 1

            def evaluate_document(self, order):
                score = 0.85 if order == baseline else 0.6
                return {"score": score, "ink_coverage": score, "text": "page", "analysis_height": 100,
                        "seams": [{"ocr_score": score, "ink_coverage": score, "recognized_lines": 8}] * 3}

        ranked, report = verify_document_orders([(3, changed), (2, baseline)], [(1, baseline)], Oracle())
        self.assertEqual(ranked[0][1], baseline)
        self.assertFalse(report["changed_from_pairwise"])
        self.assertGreaterEqual(report["final_score"], report["baseline_score"])
        self.assertEqual(report["evaluated_orders"], 2)

    def test_page_rereading_noise_cannot_move_the_first_strip_to_the_end(self):
        # Only the first join differs: a weak real join versus blank margins
        # wrapped together. Shifting the layout rereads shared joins higher.
        baseline, wrapped = (0, 1, 2, 3), (1, 2, 3, 0)

        class Oracle:
            ocr = SimpleNamespace(enabled=True)
            workers = 1

            def evaluate_document(self, order):
                values = {(0, 1): 0.3, (1, 2): 0.7, (2, 3): 0.7} if order == baseline else \
                         {(1, 2): 0.9, (2, 3): 0.9, (3, 0): 0.0}
                seams = [{"ocr_score": values[join], "ink_coverage": 1, "recognized_lines": 8}
                         for join in zip(order, order[1:])]
                return {"score": float(np.mean(list(values.values()))), "ink_coverage": 1, "text": "page",
                        "analysis_height": 100, "seams": seams}

        ranked, report = verify_document_orders([(3, wrapped)], [(2, baseline)], Oracle())
        self.assertEqual(ranked[0][1], baseline)
        self.assertFalse(report["changed_from_pairwise"])
        self.assertGreater(report["candidates"][1]["page_score"], report["candidates"][0]["page_score"])

    def test_compact_ocr_covers_both_page_ends_and_restores_coordinates(self):
        class ReadingOracle:
            enabled = True
            calls = 0

            def recognize(self, image):
                self.calls += 1
                self.height = image.shape[0]
                active = (image < 100).any(1)
                changes = np.diff(np.pad(active.astype(int), (1, 1)))
                return [{"text": "joined", "confidence": 1.0,
                         "box": [20, int(a), image.shape[1] - 40, int(b - a)]}
                        for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]

        height = 2000
        images = [np.full((height, 20, 4), 255, np.uint8) for _ in range(2)]
        positions = np.linspace(100, 1900, 40).round().astype(int)
        for image in images:
            for y in positions:
                image[y:y + 6, 2:-2, :3] = 0
        matches = SimpleNamespace(warps=np.broadcast_to(np.arange(height, dtype=np.float32), (2, 2, height)))
        oracle = ReadingOracle()
        verifier = JoinVerifier(images, matches, height, oracle)
        result = verifier.evaluate((0, 1))
        self.assertEqual(result["available_text_lines"], 40)
        self.assertEqual(result["sampled_text_lines"], 24)
        self.assertLess(oracle.height, 1500)
        lines = result["seams"][0]["lines"]
        self.assertEqual(lines[0]["y_start"], positions[0])
        self.assertEqual(lines[-1]["y_start"], positions[-1])
        self.assertTrue(all(line["text"] == "joined" for line in lines))
        verifier.evaluate((0, 1))
        self.assertEqual(oracle.calls, 1)

    def test_similar_blank_edges_cannot_wrap_end_of_page_to_start(self):
        images = [np.full((100, 20, 4), 255, np.uint8) for _ in range(3)]
        scores = np.full((3, 3), 0.1)
        np.fill_diagonal(scores, -np.inf)
        scores[0, 1] = scores[1, 2] = 0.7
        scores[2, 0] = 0.95  # A misleading match between the outer margins.
        matches = SimpleNamespace(scores=scores, warps=np.broadcast_to(np.arange(100, dtype=np.float32), (3, 3, 100)))
        verifier = JoinVerifier(images, matches, 100, SimpleNamespace(enabled=True))

        def readings(pairs):
            for pair in pairs:
                readable = pair in ((0, 1), (1, 2))
                verifier.cache[pair] = {"seams": [{"ocr_score": 0.9 if readable else 0.0,
                                                  "stroke_score": 0.7, "text_lines": 20 if readable else 0}]}

        with patch.object(verifier, "evaluate_many", side_effect=readings):
            self.assertEqual(rank_orders(verifier.score_pairs())[0][1], (0, 1, 2))

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
        complete, _ = verifier.render((0, 1, 2), full_page=True)
        self.assertEqual(complete.shape[0], 60)
        self.assertGreater(complete[49, seams[-1]:].sum(), 0)
        self.assertEqual(complete[50:, seams[-1]:].sum(), 0)

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

    def test_confident_ocr_of_a_wrong_join_is_not_plausible_english(self):
        # Tesseract reads both joins confidently; only the letters differ.
        ink = np.zeros((100, 80), np.float32)
        ink[15:25, 30:50] = 1
        ink[65:75, 30:50] = 1

        def words(first, second):
            return [{"text": first, "confidence": 0.9, "box": [25, 10, 30, 20]},
                    {"text": second, "confidence": 0.9, "box": [25, 60, 30, 20]}]

        fragments = english_fragments()
        correct = measure_seam(ink, 40, words("ledge", "versity"), fragments)
        wrong = measure_seam(ink, 40, words("ledgne", "versiMary"), fragments)
        self.assertEqual(correct["ocr_score"], wrong["ocr_score"])
        self.assertGreater(correct["lexical_score"], 0.8)
        self.assertLess(wrong["lexical_score"], 0.4)
        self.assertGreater(text_evidence(correct), text_evidence(wrong))

    def test_blank_margins_joined_together_earn_no_english_bonus(self):
        # The page's right margin wrapped onto its left margin: no text crosses.
        ink = np.zeros((100, 80), np.float32)
        ink[15:25, 60:70] = 1
        blank = measure_seam(ink, 40, [], english_fragments())
        garbled = measure_seam(ink, 65, [{"text": "sweihe", "confidence": 0.5, "box": [55, 10, 20, 20]}],
                               english_fragments())
        self.assertEqual(blank["lexical_score"], 0)
        self.assertEqual(text_evidence(blank), 0)
        self.assertGreater(text_evidence(garbled), text_evidence(blank))

    def test_ink_evidence_is_unchanged_without_an_english_word_list(self):
        ink = np.zeros((100, 80), np.float32)
        ink[15:25, 30:50] = 1
        evidence = measure_seam(ink, 40, [{"text": "ledgne", "confidence": 0.9, "box": [25, 10, 30, 20]}])
        self.assertIsNone(evidence["lexical_score"])
        self.assertEqual(text_evidence(evidence), evidence["ocr_score"])
        images = [np.full((100, 20, 4), 255, np.uint8) for _ in range(2)]
        matches = SimpleNamespace(warps=np.broadcast_to(np.arange(100, dtype=np.float32), (2, 2, 100)))
        german = SimpleNamespace(enabled=True, language="deu")
        self.assertIsNone(JoinVerifier(images, matches, 100, german).fragments)
        self.assertIsNotNone(JoinVerifier(images, matches, 100, SimpleNamespace(enabled=True, language="eng")).fragments)

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
