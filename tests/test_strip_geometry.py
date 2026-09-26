from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from normalize_strips import normalize_strip
from strip_geometry import estimate_strip_edges, repair_strip_mask


class StripGeometryTests(unittest.TestCase):
    def test_edge_text_is_preserved_without_horizontal_tearing(self):
        # A straight white strip with black strokes touching both boundaries.
        image = np.zeros((240, 100, 4), dtype=np.uint8)
        image[:, 20:80] = 255
        image[98:110, 20:40, :3] = 0
        image[150:163, 62:80, :3] = 0
        expected = image[:, 20:80].copy()

        # Brightness segmentation mistakes the strokes for missing paper.
        image[:, :, 3] = np.where(image[:, :, 0] > 200, 255, 0)
        original_colors = image[:, :, :3].copy()
        image[:, :, 3] = repair_strip_mask(image[:, :, 3])
        np.testing.assert_array_equal(image[:, :, :3], original_colors)
        np.testing.assert_array_equal(normalize_strip(image), expected)

    def test_smooth_edges_follow_a_curve_despite_text_notches(self):
        rows = np.arange(500)
        true_left = 25 + 10 * np.sin(rows / 100)
        mask = np.zeros((500, 110), dtype=np.uint8)
        for y, left in enumerate(np.rint(true_left).astype(int)):
            mask[y, left : left + 60] = 255
        for y in range(180, 192):
            left = round(true_left[y])
            mask[y, left : left + 18] = 0

        left, right = estimate_strip_edges(mask)
        np.testing.assert_allclose(left[40:-40], true_left[40:-40], atol=1)
        np.testing.assert_allclose(right[40:-40], true_left[40:-40] + 59, atol=1)
        self.assertLess(np.abs(np.diff(left[40:-40])).max(), 0.3)

    def test_missing_rows_stay_transparent_without_hidden_color_bleeding(self):
        image = np.full((100, 50, 4), (255, 0, 255, 0), dtype=np.uint8)
        image[:, 10:40] = (80, 100, 120, 255)
        image[50] = (255, 0, 255, 0)
        image[:, :, 3] = repair_strip_mask(image[:, :, 3])
        normalized = normalize_strip(image, width=45)
        self.assertEqual(normalized.shape, (100, 45, 4))
        self.assertTrue(np.all(normalized[50, :, 3] == 0))
        self.assertTrue(np.all(normalized[49, :, :3] == (80, 100, 120)))
        np.testing.assert_array_equal(normalized[51], normalized[49])


if __name__ == "__main__":
    unittest.main()
