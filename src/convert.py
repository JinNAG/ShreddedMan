"""Inspect a source image without assuming a shared upload directory."""

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="For example: img/<submission_id>/source_images/photo.jpg")
    args = parser.parse_args()
    image = cv2.imread(str(args.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        parser.error(f"Could not load {args.image_path}")
    print(image.shape)
    print(image)


if __name__ == "__main__":
    main()
