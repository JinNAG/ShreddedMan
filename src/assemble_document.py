"""Create and process one submission from one or more uploaded photos.

Examples:
    python src/assemble_document.py /path/photo1.jpg /path/photo2.jpg
    python src/assemble_document.py --submission img/<submission_id>

New submissions get UUID directories in img/. An existing submission can be
reprocessed in place; every supported image in its source_images is included.
"""

import argparse
from pathlib import Path

from detect_paper import detect_submission, orient_photo
from normalize_strips import normalize_submission
from sort_strips import sort_submission
from submission import DEFAULT_IMG_ROOT, Submission, create_submission


def process_submission(submission_dir: Path, rotation: str = "auto") -> dict:
    """Run all stages in one existing submission, recording its status."""
    submission = Submission(Path(submission_dir))
    if not submission.source_images.is_dir():
        raise ValueError(f"Submission has no source_images directory: {submission.directory}")
    try:
        submission.sources()
        manifest = submission.read_manifest()
        manifest.update(status="processing")
        manifest.pop("error", None)
        manifest.pop("result", None)
        submission.save_manifest(manifest)
        detect_submission(submission.directory, rotation)
        normalize_submission(submission.directory)
        report = sort_submission(submission.directory)
    except Exception as error:
        manifest = submission.read_manifest()
        manifest.update(status="failed", error=str(error))
        manifest.pop("result", None)
        submission.save_manifest(manifest)
        raise
    return {**report, "submission_dir": str(submission.directory)}


def assemble_document(
    source_images: list[Path], img_root: Path = DEFAULT_IMG_ROOT, rotation: str = "auto",
) -> dict:
    """Application entry point: copy uploads, reserve a UUID, and assemble."""
    submission = create_submission(source_images, img_root)
    return process_submission(submission.directory, rotation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_images", nargs="*", type=Path, help="One or more photos for a new submission.")
    parser.add_argument("--submission", type=Path, help="Reprocess an existing submission instead of creating one.")
    parser.add_argument("--img-root", type=Path, default=DEFAULT_IMG_ROOT, help="Parent directory for new submissions.")
    parser.add_argument(
        "--rotation", choices=("auto", "0", "90", "180", "270"), default="auto",
        help="Counterclockwise rotation before detection (default: auto).",
    )
    args = parser.parse_args()
    if bool(args.source_images) == bool(args.submission):
        parser.error("Provide source photos OR --submission with an existing submission folder.")
    try:
        if args.submission:
            report = process_submission(args.submission, args.rotation)
        else:
            report = assemble_document(args.source_images, args.img_root, args.rotation)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Submission: {report['submission_dir']}")
    print(f"Saved {Path(report['submission_dir']) / report['result']}")
    print(f"Ordered {report['text_strip_count']} text strips; {report['unplaced_low_ink_count']} low-ink positions unresolved.")


if __name__ == "__main__":
    main()
