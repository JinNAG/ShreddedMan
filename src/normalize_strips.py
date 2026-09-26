"""Unbend mostly vertical strip crops using their alpha masks.

Usage:
    python src/normalize_strips.py img/<submission_id>

All submission crops go to normalized_strips/strip<number>.png.
Rows are resampled horizontally; this does not correct folds, strong bends,
or vertical perspective distortion.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from strip_geometry import estimate_strip_edges
from submission import Submission


def normalize_strip(image: np.ndarray, width: int | None = None) -> np.ndarray:
    """Return a straight BGRA strip, preserving vertical pixel spacing.

    The default output width is the median estimated paper width. Smooth edges
    prevent dark letters and pixel-sized mask notches from distorting the text.
    Transparent padding above and below the strip is removed. Missing pixels
    inside the strip stay transparent instead of becoming invented content.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected an 8-bit PNG with an alpha channel.")
    if width is not None and width < 2:
        raise ValueError("Output width must be at least 2 pixels.")

    mask = image[:, :, 3] > 0
    occupied_rows = np.flatnonzero(mask.any(axis=1))
    if occupied_rows.size == 0:
        raise ValueError("The strip contains no visible paper.")

    image = image[occupied_rows[0] : occupied_rows[-1] + 1]
    mask = image[:, :, 3] > 0
    height = mask.shape[0]
    left, right = estimate_strip_edges(mask)

    if width is None:
        width = max(2, int(np.rint(np.median(right - left + 1))))

    # Sample smooth paper boundaries, retaining zero alpha for missing pixels.
    rows = np.arange(height, dtype=np.float32)
    across = np.linspace(0, 1, width, dtype=np.float32)
    map_x = left[:, None] + (right - left)[:, None] * across[None, :]
    map_y = np.broadcast_to(rows[:, None], (height, width)).copy()

    # Transparent pixels still contain carpet RGB in the source crops. Multiply
    # by alpha before interpolation so that hidden colors cannot bleed in.
    source = image.astype(np.float32)
    source[:, :, :3] *= source[:, :, 3:4] / 255.0
    result = cv2.remap(
        source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    alpha = result[:, :, 3:4] / 255.0
    colors = np.full_like(result[:, :, :3], 255.0)
    np.divide(result[:, :, :3], alpha, out=colors, where=alpha > 0)
    result[:, :, :3] = colors
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def normalize_directory(
    input_dir: Path, output_dir: Path | None = None, width: int | None = None
) -> list[Path]:
    """Normalize pooled crops, keeping submission-wide strip filenames."""
    if width is not None and width < 2:
        raise ValueError("--width must be at least 2 pixels")
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    paths = sorted(
        (path for path in input_dir.glob("strip*.png") if path.stem[5:].isdigit()),
        key=lambda path: int(path.stem[5:]),
    )
    if not paths:
        raise ValueError(f"No strip PNGs found in {input_dir}")

    if output_dir is None:
        if input_dir.name != "cropped_strips":
            raise ValueError("Specify output_dir or use a submission's cropped_strips directory.")
        output_dir = input_dir.parent / "normalized_strips"
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Output directory must differ from the input directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Could not open {path}")
        try:
            normalized = normalize_strip(image, width=width)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        output_path = output_dir / path.name
        if not cv2.imwrite(str(output_path), normalized):
            raise OSError(f"Could not save {output_path}")
        print(
            f"{output_path}: {image.shape[1]}x{image.shape[0]} -> "
            f"{normalized.shape[1]}x{normalized.shape[0]}"
        )

    # A corrected detection may produce fewer strips than the previous run.
    saved_names = {path.name for path in paths}
    for path in output_dir.glob("strip*.png"):
        if path.stem[5:].isdigit() and path.name not in saved_names:
            path.unlink()

    print(f"Normalized {len(paths)} strips into {output_dir}")
    return [output_dir / path.name for path in paths]


def normalize_submission(submission_dir: Path, width: int | None = None) -> list[Path]:
    submission = Submission(Path(submission_dir))
    paths = normalize_directory(submission.cropped_strips, submission.normalized_strips, width)
    submission.order_path.unlink(missing_ok=True)
    (submission.final_document / "document.png").unlink(missing_ok=True)
    manifest = submission.read_manifest()
    manifest.update(status="normalized")
    manifest.pop("error", None)
    manifest.pop("result", None)
    submission.save_manifest(manifest)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", type=Path,
        help="Submission folder, or a crop directory when using --output-dir.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--width", type=int, help="Output width in pixels (default: median per strip)."
    )
    args = parser.parse_args()
    try:
        if args.output_dir is None and (args.input_dir / "cropped_strips").is_dir():
            normalize_submission(args.input_dir, args.width)
        else:
            normalize_directory(args.input_dir, args.output_dir, args.width)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
