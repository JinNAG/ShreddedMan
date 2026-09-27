from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detect_paper import detect_strips, save_strips
from normalize_strips import main as normalize_main


def interrupted_strips(width=40):
    # Four strips with only three pixels between them. Each has two dark rules
    # crossing its entire width, which divide a brightness mask into segments.
    image = np.zeros((width * 10, 4 * (width + 3) + 20, 3), dtype=np.uint8)
    for number in range(4):
        x = 10 + number * (width + 3)
        image[width : width * 9, x : x + width] = 240
        for y in (width * 3, width * 6):
            image[y : y + width // 3, x : x + width] = 30
    return image


class PaperDetectionTests(unittest.TestCase):
    def test_bright_background_fibres_are_rejected_at_different_resolutions(self):
        for width in (20, 40, 80):
            with self.subTest(width=width):
                paper = interrupted_strips(width)
                image = np.zeros((paper.shape[0] + 2 * width, paper.shape[1] + 2 * width, 3), np.uint8)
                image[width:width + paper.shape[0], :paper.shape[1]] = paper
                # Long thin fibres have enough area to pass a size-only test.
                x = paper.shape[1] + 3
                image[width:7 * width, x:x + max(3, round(0.15 * width))] = 240
                # A separate bright speck exceeds the old fixed area cutoff.
                image[3:8, 3:width + 15] = 240
                strips = detect_strips(image)
                self.assertEqual(len(strips), 4)
                for number, strip in enumerate(strips):
                    self.assertEqual(cv2.boundingRect(strip),
                                     (10 + number * (width + 3), 2 * width, width, 8 * width))

    def test_short_full_width_piece_is_not_mistaken_for_a_fibre(self):
        image = np.zeros((500, 180, 3), np.uint8)
        image[30:460, 20:60] = 240
        image[100:140, 100:140] = 240
        self.assertEqual([cv2.boundingRect(c) for c in detect_strips(image)],
                         [(20, 30, 40, 430), (100, 100, 40, 40)])

    def test_joins_ink_gaps_without_merging_neighbors_at_different_resolutions(self):
        for width in (20, 40, 80):
            with self.subTest(width=width):
                strips = detect_strips(interrupted_strips(width))
                self.assertEqual(len(strips), 4)
                for number, strip in enumerate(strips):
                    self.assertEqual(
                        cv2.boundingRect(strip),
                        (10 + number * (width + 3), width, width, width * 8),
                    )

    def test_does_not_bridge_large_gaps_between_distinct_pieces(self):
        image = np.zeros((300, 100, 3), dtype=np.uint8)
        image[20:120, 20:60] = 240
        image[180:280, 20:60] = 240
        self.assertEqual(len(detect_strips(image)), 2)

    def test_dark_photo_has_no_strips(self):
        self.assertEqual(detect_strips(np.zeros((100, 100, 3), np.uint8)), [])

    def test_regeneration_preserves_ink_and_removes_only_obsolete_numbered_files(self):
        image = interrupted_strips()
        strips = detect_strips(image)
        with tempfile.TemporaryDirectory() as directory:
            crops = Path(directory) / "crops"
            normalized = Path(directory) / "normalized"
            for folder in (crops, normalized):
                folder.mkdir()
                (folder / "strip10.png").write_bytes(b"old generated crop")
                (folder / "strip_notes.png").write_bytes(b"keep this file")
                (folder / "comparison.png").write_bytes(b"keep this too")

            save_strips(image, strips, crops)
            crop = cv2.imread(str(crops / "strip1.png"), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(crop[:, :, :3], image[40:360, 10:50])
            self.assertTrue(np.all(crop[:, :, 3] == 255))

            # Only numbered strip PNGs are input; unrelated files are retained.
            with patch.object(
                sys, "argv", ["normalize_strips.py", str(crops), "--output-dir", str(normalized)]
            ), redirect_stdout(io.StringIO()):
                normalize_main()

            for folder in (crops, normalized):
                self.assertFalse((folder / "strip10.png").exists())
                self.assertEqual((folder / "strip_notes.png").read_bytes(), b"keep this file")
                self.assertTrue((folder / "comparison.png").exists())
                for number in range(1, 5):
                    self.assertTrue((folder / f"strip{number}.png").exists())


if __name__ == "__main__":
    unittest.main()
