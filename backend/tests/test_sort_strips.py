from pathlib import Path
import json
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sort_strips import prepare_matching_images, rank_orders, sort_strips
from strip_matching import _affine_matches, ink_profiles, match_profiles, stabilize_row_maps


class StripSortingTests(unittest.TestCase):
    def test_nearby_reciprocal_matches_correct_an_outlier_row_offset(self):
        rows = np.arange(200, dtype=np.float32)
        offsets = [0, 10, 20, 30, 40]
        warps = np.array([[rows + offsets[j] - offsets[i]
                           for j in range(5)] for i in range(5)])
        warps[2, 3] = rows + 180
        warps[3, 2] = rows + 100
        direct = [rows + value for value in (0, 10, 20, 200, 210)]
        scores = np.full((5, 5), 0.8)
        adjusted = stabilize_row_maps(tuple(range(5)), direct, warps, scores)
        self.assertEqual([round(mapping[100] - 100) for mapping in adjusted], offsets)

    def test_disconnected_weak_tail_does_not_disable_other_row_corrections(self):
        rows = np.arange(200, dtype=np.float32)
        offsets = [0, 10, 20, 30, 40, 50]
        warps = np.array([[rows + offsets[j] - offsets[i]
                           for j in range(6)] for i in range(6)])
        warps[2, 3] = rows + 180
        warps[3, 2] = rows + 100
        for index in range(2, 5):
            warps[index, 5] = rows + 180
            warps[5, index] = rows + 100
        direct = [rows + value for value in (0, 10, 20, 200, 210, 260)]
        scores = np.full((6, 6), 0.8)
        adjusted = stabilize_row_maps(tuple(range(6)), direct, warps, scores)
        self.assertEqual([round(mapping[100] - 100) for mapping in adjusted[:5]], offsets[:5])
        self.assertEqual(round(adjusted[5][100] - 100), 260)

    def test_large_crop_height_difference_preserves_text_row_spacing(self):
        short = np.full((100, 20, 4), 255, np.uint8)
        tall = np.full((120, 20, 4), 255, np.uint8)
        short[40:44, :, :3] = 0
        prepared, height, preserved = prepare_matching_images([short, tall])
        self.assertTrue(preserved)
        self.assertEqual(height, 120)
        np.testing.assert_array_equal(prepared[0][10:110], short)
        self.assertTrue(np.all(prepared[0][:10, :, 3] == 0))
        self.assertTrue(np.all(prepared[0][110:, :, 3] == 0))
        opaque_ink = (prepared[0][:, :, :3] == 0).any(axis=2) & (prepared[0][:, :, 3] > 0)
        self.assertEqual(np.flatnonzero(opaque_ink.any(axis=1)).tolist(),
                         [50, 51, 52, 53])

    def test_shared_fft_matches_direct_full_height_correlation(self):
        # Check both directions and positive/negative shifts independently of
        # FFT implementation details, including the zero padding at the tips.
        profiles = np.random.default_rng(41).uniform(0, 1, (4, 3, 128)).astype(np.float32)
        actual, scales, offsets = _affine_matches(profiles, 17, 0)
        rows = np.arange(128)
        for a in range(4):
            for b in range(4):
                if a == b:
                    continue
                candidates = []
                for shift in range(-17, 18):
                    score = 0
                    for weight, left, right in ((0.8, profiles[a, 1], profiles[b, 0]),
                                                (0.2, profiles[a, 2], profiles[b, 2])):
                        aligned = np.interp(rows - shift, rows, right, left=0, right=0)
                        score += weight * 2 * np.sum(left * aligned) / (np.sum(left ** 2) + np.sum(right ** 2))
                    candidates.append((score, shift))
                expected, shift = max(candidates)
                self.assertAlmostEqual(actual[a, b], expected, places=6)
                self.assertEqual(offsets[a, b], shift)
                self.assertEqual(scales[a, b], 1)

    def test_global_order_can_reject_the_highest_individual_match(self):
        scores = np.zeros((4, 4))
        np.fill_diagonal(scores, -np.inf)
        scores[0, 1] = 0.9
        scores[1, 0] = scores[0, 2] = scores[2, 3] = 0.8
        self.assertEqual(rank_orders(scores)[0][1], (1, 0, 2, 3))

    def test_global_solver_uses_each_strip_once(self):
        scores = np.zeros((9, 9))
        np.fill_diagonal(scores, -np.inf)
        for i in range(8):
            scores[i, i + 1] = 1
        self.assertEqual(rank_orders(scores)[0][1], tuple(range(9)))

    def test_blank_paper_has_no_matching_evidence_or_arbitrary_offset(self):
        profiles = [ink_profiles(np.full((200, 40, 4), 255, np.uint8), 200)] * 2
        matches = match_profiles(profiles)
        self.assertEqual(matches.scores[0, 1], 0)
        self.assertEqual(matches.scales[0, 1], 1)
        self.assertEqual(matches.offsets[0, 1], 0)

    def test_hidden_background_colors_do_not_change_ink_profiles(self):
        image = np.full((200, 40, 4), 255, np.uint8)
        image[50:60, 5:35, :3] = 20
        image[:, :3, 3] = 0
        image[:, -3:, 3] = 0
        changed = image.copy()
        changed[changed[:, :, 3] == 0, :3] = (255, 0, 255)
        np.testing.assert_array_equal(ink_profiles(image, 200), ink_profiles(changed, 200))

    def test_shuffled_offset_strips_are_recovered_without_changing_sources(self):
        # Distinct strokes cross each cut in a known synthetic page. Offset
        # the pieces vertically before shuffling, just as photo crops can be.
        height, width = 640, 64
        page = np.full((height, width * 4, 4), 255, dtype=np.uint8)
        rng = np.random.default_rng(72)
        for boundary in range(1, 4):
            for y in sorted(rng.choice(np.arange(70, 570, 16), 12, replace=False)):
                cv2.line(
                    page, (boundary * width - 20, int(y) - 2),
                    (boundary * width + 20, int(y) + 2), (20, 20, 20, 255), 5,
                )
        pieces = []
        for i, shift in enumerate((10, -8, 7, -4)):
            piece = page[:, i * width : (i + 1) * width]
            shifted = cv2.warpAffine(
                piece, np.float32([[1, 0, 0], [0, 1, shift]]), (width, height),
                borderValue=(255, 255, 255, 255),
            )
            pieces.append(shifted)

        with tempfile.TemporaryDirectory() as directory:
            input_dir, output_dir = Path(directory) / "input", Path(directory) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for number, original in enumerate((2, 0, 3, 1), 1):
                self.assertTrue(cv2.imwrite(str(input_dir / f"strip{number}.png"), pieces[original]))
            originals = {path.name: path.read_bytes() for path in input_dir.glob("*.png")}
            report = sort_strips(input_dir, output_dir, ocr_mode="off")
            self.assertEqual(
                [item["source"] for item in report["order"]],
                ["strip2.png", "strip4.png", "strip1.png", "strip3.png"],
            )
            for item in report["order"]:
                self.assertEqual((input_dir / item["source"]).read_bytes(), originals[item["source"]])
            self.assertEqual(report["search"], "exhaustive")
            self.assertIsNotNone(cv2.imread(str(output_dir / "document.png")))
            self.assertTrue((output_dir.parent / "order.json").is_file())
            self.assertEqual({p.name for p in output_dir.iterdir()}, {"document.png", "join_report.html", "join_report.json"})
            self.assertEqual([item["position"] for item in report["order"]], [1, 2, 3, 4])
            verification = json.loads((output_dir / "join_report.json").read_text())
            self.assertEqual(len(verification["joins"]), 3)
            self.assertEqual(verification["ocr"]["status"], "disabled")
            for index, join in enumerate(verification["joins"]):
                self.assertEqual((join["left_position"], join["right_position"]), (index + 1, index + 2))
                self.assertEqual(join["left_strip"], report["order"][index]["source"])
                self.assertEqual(join["right_strip"], report["order"][index + 1]["source"])
                self.assertEqual(join["confidence"]["level"], "low")
                self.assertIsNone(join["scores"]["ocr_ink_evidence"])

    def test_full_height_matching_handles_warp_and_keeps_blanks_out_of_text(self):
        height, width, count = 1200, 64, 5
        page = np.full((height, width * count, 4), 255, np.uint8)
        rng = np.random.default_rng(49)
        for boundary in range(1, count):
            # A large identical mark near the top is a misleading match.
            cv2.rectangle(page, (boundary * width - 18, 90), (boundary * width + 18, 130), (20, 20, 20, 255), -1)
            # Distinct smaller strokes across the rest of the page establish
            # the real neighbors; no single short patch identifies the order.
            for y in sorted(rng.choice(np.arange(280, 1100, 15), 24, replace=False)):
                cv2.line(page, (boundary * width - 19, int(y) - 2),
                         (boundary * width + 19, int(y) + 2), (20, 20, 20, 255), 4)
        pieces = []
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        for index in range(count):
            rows = (yy - (index - 2) * 4) / (1 + (index - 2) * 0.005)
            rows += 2.5 * np.sin(yy / 180 + index)
            pieces.append(cv2.remap(
                page[:, index * width:(index + 1) * width], xx, rows, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 255),
            ))
        with tempfile.TemporaryDirectory() as directory:
            inputs, output = Path(directory) / "input", Path(directory) / "output"
            inputs.mkdir()
            for number, index in enumerate((3, 1, 4, 0, 2), 1):
                cv2.imwrite(str(inputs / f"strip{number}.png"), pieces[index])
            blank = np.full((height, width, 4), 255, np.uint8)
            blank[:, :2, :3] = 90  # Cut-edge shadows must not count as text.
            cv2.imwrite(str(inputs / "strip6.png"), blank)
            report = sort_strips(inputs, output, ocr_mode="off")
            self.assertEqual([entry["source"] for entry in report["order"]],
                             ["strip4.png", "strip2.png", "strip5.png", "strip1.png", "strip3.png", "strip6.png"])
            self.assertEqual(report["matching"]["ordered_pairs_compared"], 20)
            self.assertEqual(report["text_strip_count"], 5)
            self.assertEqual(report["unplaced_low_ink_count"], 1)
            self.assertEqual(report["order"][-1]["position_status"], "unplaced_low_ink")
            self.assertIsNone(report["order"][-1]["preview"])
            self.assertIsNone(report["order"][-1]["position"])
            np.testing.assert_array_equal(cv2.imread(str(inputs / "strip6.png"), -1), blank)
            preview = cv2.imread(str(output / "document.png"))
            self.assertEqual(preview.shape[1], count * width)

    def test_all_blank_input_leaves_existing_output_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, output = Path(directory) / "input", Path(directory) / "output"
            inputs.mkdir()
            output.mkdir()
            cv2.imwrite(str(inputs / "strip1.png"), np.full((200, 40, 4), 255, np.uint8))
            (output.parent / "order.json").write_text("previous result")
            with self.assertRaisesRegex(ValueError, "blank or too faint"):
                sort_strips(inputs, output, ocr_mode="off")
            self.assertEqual((output.parent / "order.json").read_text(), "previous result")


if __name__ == "__main__":
    unittest.main()
