import argparse
from pathlib import Path
import tempfile

import cv2
import numpy as np

from strip_geometry import repair_strip_mask
from submission import Submission


def orient_photo(image, rotation: str = "auto"):
    """Make strip length vertical; return the image and applied CCW degrees."""
    if rotation == "auto":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, 0
        _, _, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        degrees = 90 if width > height else 0
    else:
        degrees = int(rotation)
    rotations = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    if degrees not in (0, 90, 180, 270):
        raise ValueError("Rotation must be auto, 0, 90, 180, or 270.")
    return (cv2.rotate(image, rotations[degrees]) if degrees else image), degrees


def detect_strips(image: np.ndarray, min_strip_area: float = 100) -> list[np.ndarray]:
    """Find mostly vertical paper strips, bridging interruptions caused by ink."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    initial, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not initial:
        return []

    # Estimate paper width from the largest bright region, using row widths
    # rather than its bounding box, which also includes lateral bending.
    largest = max(initial, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest)
    sample = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(sample, [largest], -1, 255, cv2.FILLED, offset=(-x, -y))
    visible = sample > 0
    row_widths = width - visible.argmax(axis=1) - visible[:, ::-1].argmax(axis=1)
    paper_width = float(np.median(row_widths[visible.any(axis=1)]))

    # A 5x5 close cannot bridge large lettering or rules across a whole strip.
    # Scale the gap limit with resolution, and close vertically only so nearby
    # parallel strips remain separate even when their horizontal gap is small.
    gap_kernel_height = max(5, int(np.ceil(paper_width / 2)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_kernel_height))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strips = [c for c in contours if cv2.contourArea(c) >= min_strip_area]
    return sorted(strips, key=lambda contour: cv2.boundingRect(contour)[:2])


def save_strips(
    image: np.ndarray, strips: list[np.ndarray], output_dir: Path,
    start_number: int = 1, remove_stale: bool = True,
) -> list[Path]:
    """Save complete strips and remove obsolete numbered crops after success."""
    if start_number < 1:
        raise ValueError("Strip numbering must start at 1 or greater.")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for number, strip in enumerate(strips, start=start_number):
        x, y, width, height = cv2.boundingRect(strip)
        cropped = image[y : y + height, x : x + width]
        strip_mask = np.zeros((height, width), dtype=np.uint8)

        # Fill the whole outline—including the black text inside it.
        cv2.drawContours(
            strip_mask, [strip], -1, 255, thickness=cv2.FILLED, offset=(-x, -y)
        )
        # Recover edge letters using the surrounding paper's geometry.
        strip_mask = repair_strip_mask(strip_mask)

        result = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = strip_mask

        output_path = output_dir / f"strip{number}.png"
        if not cv2.imwrite(str(output_path), result):
            raise OSError(f"Could not save {output_path}")
        saved.append(output_path)

    if remove_stale:
        saved_names = {path.name for path in saved}
        for path in output_dir.glob("strip*.png"):
            if path.stem[5:].isdigit() and path.name not in saved_names:
                path.unlink()
    return saved


def detect_submission(submission_dir: Path, rotation: str = "auto") -> list[Path]:
    """Pool crops from every source photo using one submission-wide sequence."""
    submission = Submission(Path(submission_dir))
    paths = submission.sources()
    submission.ensure_layout()
    manifest = submission.read_manifest()
    originals = {source["filename"]: source.get("original_name", source["filename"])
                 for source in manifest.get("sources", [])}
    sources, mapping = [], []
    # Finish all photos before replacing existing crops. A bad later upload
    # must not silently leave a submission containing only the earlier photos.
    with tempfile.TemporaryDirectory(prefix=".detect-", dir=submission.directory) as directory:
        staged = Path(directory)
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not open {path}")
            image, degrees = orient_photo(image, rotation)
            strips = detect_strips(image)
            if not strips:
                raise ValueError(f"No paper strips detected in {path}")
            saved = save_strips(image, strips, staged, start_number=len(mapping) + 1, remove_stale=False)
            for number, crop in enumerate(saved, 1):
                mapping.append({
                    "filename": crop.name, "source_image": f"source_images/{path.name}",
                    "source_strip_number": number,
                })
            sources.append({
                "filename": path.name, "original_name": originals.get(path.name, path.name),
                "rotation_ccw": degrees, "strip_count": len(strips),
            })
            print(f"{path.name}: {len(strips)} strips, rotation {degrees} degrees CCW")
        names = {entry["filename"] for entry in mapping}
        for name in names:
            (staged / name).replace(submission.cropped_strips / name)
        for path in submission.cropped_strips.glob("strip*.png"):
            if path.stem[5:].isdigit() and path.name not in names:
                path.unlink()

    # Detection changes the meaning of strip IDs. Invalidate downstream output.
    for path in submission.normalized_strips.glob("strip*.png"):
        if path.stem[5:].isdigit():
            path.unlink()
    submission.order_path.unlink(missing_ok=True)
    (submission.final_document / "document.png").unlink(missing_ok=True)
    manifest.update(status="detected", sources=sources, strips=mapping)
    manifest.pop("error", None)
    manifest.pop("result", None)
    submission.save_manifest(manifest)
    return [submission.cropped_strips / entry["filename"] for entry in mapping]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect strips from all source_images in an existing submission."
    )
    parser.add_argument(
        "submission_dir", type=Path, help="Submission folder: img/<submission_id>.",
    )
    parser.add_argument("--rotation", choices=("auto", "0", "90", "180", "270"), default="auto")
    args = parser.parse_args()
    try:
        paths = detect_submission(args.submission_dir, args.rotation)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved {len(paths)} strips to {args.submission_dir / 'cropped_strips'}")


if __name__ == "__main__":
    main()
