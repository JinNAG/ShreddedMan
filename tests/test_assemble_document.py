from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assemble_document import assemble_document, process_submission, orient_photo
from detect_paper import detect_submission
from normalize_strips import normalize_submission
from sort_strips import sort_strips, sort_submission
from submission import create_submission, Submission


def source_photo(seed):
    image = np.zeros((320, 150, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for x in (10, 95):
        image[20:300, x:x + 40] = 240
        for y in rng.choice(np.arange(60, 260, 12), 8, replace=False):
            cv2.line(image, (x, int(y)), (x + 39, int(y) + 2), (30, 30, 30), 3)
    return image


class DocumentAssemblyTests(unittest.TestCase):
    def test_auto_rotation_makes_horizontal_strips_vertical(self):
        upright = source_photo(1)
        sideways = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
        rotated, degrees = orient_photo(sideways)
        self.assertEqual(degrees, 90)
        np.testing.assert_array_equal(rotated, upright)
        unchanged, degrees = orient_photo(upright)
        self.assertEqual(degrees, 0)
        np.testing.assert_array_equal(unchanged, upright)

    def test_duplicate_upload_names_pool_all_strips_into_one_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "upload1/photo.png", root / "upload2/photo.png"]
            for number, path in enumerate(paths, 1):
                path.parent.mkdir()
                image = source_photo(number)
                if number == 2:
                    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                self.assertTrue(cv2.imwrite(str(path), image))
            originals = [path.read_bytes() for path in paths]
            with redirect_stdout(io.StringIO()):
                report = assemble_document(paths, img_root=root / "img")
            submission = Submission(Path(report["submission_dir"]))
            self.assertEqual(uuid.UUID(submission.directory.name).version, 4)
            self.assertEqual(len(report["order"]), 4)
            self.assertEqual([item["strip_count"] for item in report["source_images"]], [2, 2])
            self.assertEqual([item["rotation_ccw"] for item in report["source_images"]], [90, 90])
            self.assertEqual({item["source_image"] for item in report["order"]},
                             {"source_images/photo.png", "source_images/photo_2.png"})
            widths = [item["preview"]["width"] for item in report["order"]]
            self.assertLessEqual(max(widths) - min(widths), 3)
            for folder in (submission.cropped_strips, submission.normalized_strips):
                self.assertEqual({p.name for p in folder.iterdir()}, {f"strip{n}.png" for n in range(1, 5)})
            self.assertEqual([path.read_bytes() for path in paths], originals)
            self.assertEqual([p.read_bytes() for p in submission.sources()], originals)
            self.assertEqual([p.name for p in submission.final_document.iterdir()], ["document.png"])
            self.assertIsNotNone(cv2.imread(str(submission.directory / report["result"])))
            saved = json.loads(submission.order_path.read_text())
            for item in saved["order"]:
                self.assertTrue((submission.directory / item["source_path"]).is_file())
                self.assertTrue((submission.directory / item["source_image"]).is_file())
            self.assertEqual(submission.read_manifest()["status"], "complete")

    def test_submissions_are_isolated_and_uuid_collision_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), source_photo(1))
            first = create_submission([source], root / "img")
            before = first.manifest_path.read_bytes()
            with patch("submission.uuid.uuid4", side_effect=[uuid.UUID(first.directory.name), uuid.uuid4(), uuid.uuid4()]):
                second = create_submission([source], root / "img")
            self.assertNotEqual(first.directory, second.directory)
            self.assertEqual(first.manifest_path.read_bytes(), before)
            self.assertEqual((first.source_images / source.name).read_bytes(), source.read_bytes())
            self.assertEqual((second.source_images / source.name).read_bytes(), source.read_bytes())

    def test_new_uploads_and_removed_uploads_are_reflected_on_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), source_photo(1))
            with redirect_stdout(io.StringIO()):
                first = assemble_document([source], root / "img")
                submission = Submission(Path(first["submission_dir"]))
                cv2.imwrite(str(submission.source_images / "added.PNG"), source_photo(2))
                second = process_submission(submission.directory)
                self.assertEqual(len(second["order"]), 4)
                (submission.source_images / "added.PNG").unlink()
                third = process_submission(submission.directory)
            self.assertEqual(len(third["order"]), 2)
            for folder in (submission.cropped_strips, submission.normalized_strips):
                self.assertEqual({p.name for p in folder.iterdir()}, {"strip1.png", "strip2.png"})
            self.assertEqual([p.name for p in submission.final_document.iterdir()], ["document.png"])
            self.assertEqual(len(list((root / "img").iterdir())), 1)
            (submission.source_images / "source.png").unlink()
            with self.assertRaisesRegex(ValueError, "No source images"):
                process_submission(submission.directory)
            self.assertEqual(submission.read_manifest()["status"], "failed")
            self.assertNotIn("result", submission.read_manifest())

    def test_each_stage_and_relative_paths_work_after_moving_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            cv2.imwrite(str(source), source_photo(1))
            submission = create_submission([source], root / "img")
            with redirect_stdout(io.StringIO()):
                detect_submission(submission.directory)
                normalize_submission(submission.directory)
            moved = root / "moved" / submission.directory.name
            moved.parent.mkdir()
            shutil.move(str(submission.directory), moved)
            with redirect_stdout(io.StringIO()):
                report = sort_submission(moved)
            self.assertTrue((moved / report["result"]).is_file())
            for item in report["order"]:
                self.assertTrue((moved / item["source_path"]).is_file())
                self.assertTrue((moved / item["source_image"]).is_file())

    def test_bad_later_upload_does_not_publish_partial_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "a.png"
            cv2.imwrite(str(source), source_photo(1))
            submission = create_submission([source], root / "img")
            with redirect_stdout(io.StringIO()):
                detect_submission(submission.directory)
            before = {p.name: p.read_bytes() for p in submission.cropped_strips.iterdir()}
            (submission.source_images / "z.png").write_bytes(b"not an image")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Could not open"):
                process_submission(submission.directory)
            self.assertEqual({p.name: p.read_bytes() for p in submission.cropped_strips.iterdir()}, before)
            self.assertEqual(submission.read_manifest()["status"], "failed")
            self.assertEqual(list(submission.final_document.iterdir()), [])
            self.assertFalse(any(p.name.startswith(".detect-") for p in submission.directory.iterdir()))

    def test_missing_normalized_strip_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "a.png"
            cv2.imwrite(str(source), source_photo(1))
            submission = create_submission([source], root / "img")
            with redirect_stdout(io.StringIO()):
                detect_submission(submission.directory)
                normalize_submission(submission.directory)
            (submission.normalized_strips / "strip2.png").unlink()
            with self.assertRaisesRegex(ValueError, "rerun normalization"):
                sort_submission(submission.directory)

    def test_repeated_input_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "distinct"):
                sort_strips([path, path / "."], path / "output")

    def test_empty_or_missing_sources_do_not_create_a_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for paths in ([], [root / "missing.jpg"]):
                with self.assertRaises(ValueError):
                    assemble_document(paths, root / "img")
            self.assertFalse((root / "img").exists())


if __name__ == "__main__":
    unittest.main()
