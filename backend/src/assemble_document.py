"""Create and process one submission from one or more uploaded photos.

Examples:
    python backend/src/main.py /path/photo1.jpg /path/photo2.jpg
    python backend/src/main.py --submission backend/img/<submission_id>

New submissions get UUID directories in backend/img/. An existing submission can be
reprocessed in place; every supported image in its source_images is included.
"""

import argparse
from pathlib import Path
from time import perf_counter

from detect_paper import detect_submission
from normalize_strips import normalize_submission
from sort_strips import sort_submission
from strip_ocr import DEFAULT_OCR_WORKERS
from submission import DEFAULT_IMG_ROOT, Submission, create_submission, write_json


def process_submission(
    submission_dir: Path, rotation: str = "auto", *,
    ocr_mode: str = "auto", ocr_language: str = "eng", ocr_workers: int = DEFAULT_OCR_WORKERS,
) -> dict:
    """Run all stages in one existing submission, recording its status."""
    submission = Submission(Path(submission_dir))
    if not submission.source_images.is_dir():
        raise ValueError(f"Submission has no source_images directory: {submission.directory}")
    started = perf_counter()
    timings = {}
    try:
        submission.sources()
        manifest = submission.read_manifest()
        manifest.update(status="processing")
        manifest.pop("error", None)
        manifest.pop("result", None)
        manifest.pop("join_report", None)
        manifest.pop("review_required", None)
        manifest.pop("timings_seconds", None)
        manifest.pop("sorting_timings_seconds", None)
        submission.save_manifest(manifest)
        for name, operation in (("detection", lambda: detect_submission(submission.directory, rotation)),
                                ("normalization", lambda: normalize_submission(submission.directory))):
            stage = perf_counter()
            operation()
            timings[name] = round(perf_counter() - stage, 3)
        stage = perf_counter()
        report = sort_submission(submission.directory, ocr_mode=ocr_mode,
                                 ocr_language=ocr_language, ocr_workers=ocr_workers)
        timings["sorting"] = round(perf_counter() - stage, 3)
        timings["total"] = round(perf_counter() - started, 3)
        report["timings_seconds"] = timings
        write_json(submission.order_path, report)
        manifest = submission.read_manifest()
        manifest["timings_seconds"] = timings
        submission.save_manifest(manifest)
    except Exception as error:
        manifest = submission.read_manifest()
        manifest.update(status="failed", error=str(error))
        manifest.pop("result", None)
        manifest.pop("join_report", None)
        manifest.pop("review_required", None)
        submission.save_manifest(manifest)
        raise
    return {**report, "submission_dir": str(submission.directory)}


def assemble_document(
    source_images: list[Path], img_root: Path = DEFAULT_IMG_ROOT, rotation: str = "auto",
    *, ocr_mode: str = "auto", ocr_language: str = "eng", ocr_workers: int = DEFAULT_OCR_WORKERS,
) -> dict:
    """Application entry point: copy uploads, reserve a UUID, and assemble."""
    submission = create_submission(source_images, img_root)
    return process_submission(submission.directory, rotation, ocr_mode=ocr_mode,
                              ocr_language=ocr_language, ocr_workers=ocr_workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_images", nargs="*", type=Path, help="One or more photos for a new submission.")
    parser.add_argument("--submission", type=Path, help="Reprocess an existing submission instead of creating one.")
    parser.add_argument("--img-root", type=Path, default=DEFAULT_IMG_ROOT, help="Parent directory for new submissions.")
    parser.add_argument(
        "--rotation", choices=("auto", "0", "90", "180", "270"), default="auto",
        help="Counterclockwise rotation before detection (default: auto).",
    )
    parser.add_argument("--ocr", choices=("auto", "required", "off"), default="auto")
    parser.add_argument("--ocr-language", default="eng")
    parser.add_argument("--ocr-workers", type=int, default=DEFAULT_OCR_WORKERS)
    args = parser.parse_args()
    if bool(args.source_images) == bool(args.submission):
        parser.error("Provide source photos OR --submission with an existing submission folder.")
    try:
        if args.submission:
            report = process_submission(args.submission, args.rotation, ocr_mode=args.ocr,
                                        ocr_language=args.ocr_language, ocr_workers=args.ocr_workers)
        else:
            report = assemble_document(args.source_images, args.img_root, args.rotation, ocr_mode=args.ocr,
                                       ocr_language=args.ocr_language, ocr_workers=args.ocr_workers)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Submission: {report['submission_dir']}")
    print(f"Saved {Path(report['submission_dir']) / report['result']}")
    print(f"Join report: {Path(report['submission_dir']) / report['verification']['html_report']}")
    print(f"Ordered {report['text_strip_count']} text strips; {report['unplaced_low_ink_count']} low-ink positions unresolved.")


if __name__ == "__main__":
    main()
