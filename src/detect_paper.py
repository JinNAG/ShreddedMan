import argparse
from pathlib import Path

import cv2
import numpy as np

from strip_geometry import repair_strip_mask

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


def save_strips(image: np.ndarray, strips: list[np.ndarray], output_dir: Path) -> None:
    """Save complete strips and remove obsolete numbered crops after success."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_names = set()
    for number, strip in enumerate(strips, start=1):
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
        saved_names.add(output_path.name)

    for path in output_dir.glob("strip*.png"):
        if path.stem[5:].isdigit() and path.name not in saved_names:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect paper strips in a photo and save transparent PNG crops."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=Path("img/source_images/photo2.jpg"),
        help="Original photo, including its background (default: img/source_images/photo2.jpg).",
    )
    image_path = parser.parse_args().image_path
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}")

    strips = detect_strips(image)
    if not strips:
        raise ValueError("No paper detected. Try lowering the threshold.")

    output_dir = Path("img/cropped_images") / image_path.stem
    save_strips(image, strips, output_dir)
    print(f"Saved {len(strips)} strips to {output_dir}")


if __name__ == "__main__":
    main()
