"""Estimate the order of upright, normalized strips from a single page.

Usage:
    python backend/src/sort_strips.py backend/img/<submission_id>

Writes document.png and join_report.html/json to final_document. Source mappings,
scores, and alternatives go to order.json at the submission root. Normalized
PNGs keep their original filenames; blank/low-ink positions remain unresolved.
"""

import argparse
import heapq
from itertools import permutations
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from strip_matching import ink_fraction, ink_profiles, level_strips, match_profiles, seam_row_shifts
from strip_order import optimize_orders
from submission import Submission, write_json
from strip_ocr import DEFAULT_OCR_WORKERS, TesseractOCR
from join_verification import JoinVerifier, refine_orders, verify_document_orders
from join_report import build_join_report, write_join_reports


# Joins between leveled strips: search range (fractions of page height and of
# scale), and the score cost of each 1% shift or scale change (quadratic).
MAX_ROW_SHIFT = 0.04
MAX_RELATIVE_SCALE = 0.02
LEVEL_PRIOR = 0.05


def rank_orders(
    scores: np.ndarray, count: int = 5
) -> list[tuple[float, tuple[int, ...]]]:
    """Find complete optimal orders, enumerating only the smallest samples."""
    size = len(scores)
    if size == 0:
        raise ValueError("No strips to sort.")

    def ranking(item):
        return (-item[0], item[1])

    if size <= 8:
        candidates = (
            (sum(float(scores[a, b]) for a, b in zip(order, order[1:])), order)
            for order in permutations(range(size))
        )
        return heapq.nsmallest(count, candidates, key=ranking)

    return optimize_orders(scores, count=count)


def _extrapolate(points, coordinates, values):
    """Linear extension preserves the strip tips when composing row maps."""
    result = np.interp(points, coordinates, values)
    for outside, endpoint, neighbor in (
        (points < coordinates[0], 0, 1), (points > coordinates[-1], -1, -2)
    ):
        slope = (values[neighbor] - values[endpoint]) / (coordinates[neighbor] - coordinates[endpoint])
        result[outside] = values[endpoint] + (points[outside] - coordinates[endpoint]) * slope
    return result


def level_row_maps(
    height: int, steps: list[np.ndarray], weights: list[float], stiffness: float = 0.01,
) -> list[np.ndarray]:
    """Line up text at each join while keeping every strip level.

    steps[k][y] is the row of strip k+1 that continues row y of strip k,
    minus y. Chaining those steps would pass every small error on to all later
    strips, so the page would drift or step. Instead, all corrections are
    solved together: each join pulls its right strip toward its left strip
    (weighted by the text found along the cut), and a weak spring holds every
    strip at its leveled placement. Returns page row -> strip row maps.
    """
    rows = np.arange(height, dtype=np.float32)
    system = stiffness * np.eye(len(steps) + 1)
    targets = np.zeros((len(steps) + 1, height))
    for position, (step, weight) in enumerate(zip(steps, weights)):
        system[position:position + 2, position:position + 2] += weight * np.array([[1, -1], [-1, 1]])
        targets[position] -= weight * step
        targets[position + 1] += weight * step
    return [rows + correction for correction in np.linalg.solve(system, targets).astype(np.float32)]


def assemble_preview(
    images: list[np.ndarray], order: tuple[int, ...], maps: list[np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    """Render strips through their row maps; only the preview is resized and aligned."""
    height = len(maps[0])
    rows = np.arange(height, dtype=np.float32)
    # Bound the canvas by paper, not by the transparent rows around leveled strips.
    paper = [np.flatnonzero(images[index][:, :, 3].any(1)) for index in order]
    bounds = [_extrapolate(np.array([found[0], found[-1]] if len(found) else [0, height - 1], np.float32),
                           mapping, rows) for found, mapping in zip(paper, maps)]
    top = int(np.floor(min(bound[0] for bound in bounds)))
    bottom = int(np.ceil(max(bound[1] for bound in bounds))) + 1
    output_rows = np.arange(top, bottom, dtype=np.float32)
    widths = [max(1, round(images[i].shape[1] * height / images[i].shape[0])) for i in order]
    preview = np.full((bottom - top, sum(widths), 3), 255, np.uint8)
    placements, x = [], 0
    for index, width, mapping, bound in zip(order, widths, maps, bounds):
        image = images[index]
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        rgb = np.rint(image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        source_rows = (_extrapolate(output_rows, rows, mapping) + 0.5) * image.shape[0] / height - 0.5
        source_columns = (np.arange(width, dtype=np.float32) + 0.5) * image.shape[1] / width - 0.5
        shape = (len(output_rows), width)
        preview[:, x:x + width] = cv2.remap(
            rgb, np.broadcast_to(source_columns, shape).copy(),
            np.broadcast_to(source_rows[:, None], shape).astype(np.float32).copy(),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
        y = int(np.floor(bound[0])) - top
        placements.append({
            "x": x, "y": y, "width": width, "height": int(np.ceil(bound[1])) - top - y + 1,
            "alignment": "affine plus smooth local row offsets",
        })
        x += width
    return preview, placements


def sort_strips(
    input_dirs: Path | list[Path], output_dir: Path, *,
    ocr_mode: str = "auto", ocr_language: str = "eng", ocr_workers: int = DEFAULT_OCR_WORKERS,
) -> dict:
    """Render one document and write its order report beside the output folder."""
    started = perf_counter()
    timings = {}
    directories = [input_dirs] if isinstance(input_dirs, Path) else list(input_dirs)
    if not directories:
        raise ValueError("At least one input directory is required.")
    if len({directory.resolve() for directory in directories}) != len(directories):
        raise ValueError("Input directories must be distinct; do not include a photo twice.")
    paths = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Input directory does not exist: {directory}")
        if directory.resolve() == output_dir.resolve():
            raise ValueError("Output directory must differ from every input directory.")
        found = sorted(
            (path for path in directory.glob("strip*.png") if path.stem[5:].isdigit()),
            key=lambda path: int(path.stem[5:]),
        )
        if not found:
            raise ValueError(f"No numbered strip PNGs found in {directory}")
        paths.extend(found)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("The same strip file was included more than once.")
    labels = [str(path) if len(directories) > 1 else path.name for path in paths]
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not open {path}")
        images.append(image)
    fractions = [ink_fraction(image) for image in images]
    active = [index for index, fraction in enumerate(fractions) if fraction >= 0.0005]
    low_ink = [index for index in range(len(paths)) if index not in active]
    if not active:
        raise ValueError("No usable ink found; all strips are blank or too faint to order.")
    stage = perf_counter()
    # Matching, OCR, and the preview all use copies placed level on one page grid.
    active_images, leveling = level_strips([images[index] for index in active])
    common_height = active_images[0].shape[0]
    timings["leveling"] = round(perf_counter() - stage, 3)
    profiles = [ink_profiles(image, common_height) for image in active_images]
    if len(active) > 1 and not any(np.any(profile[:2] > 0) for profile in profiles):
        raise ValueError("No usable ink near the strip edges; cannot estimate an order.")

    ocr = TesseractOCR(ocr_mode, ocr_language, output_dir.parent / ".ocr_cache.json")
    stage = perf_counter()
    # Leveled neighbors differ by a few rows. Without a cost on shifting,
    # repeating text lines can pull a join onto the wrong line.
    matches = match_profiles(profiles, max_shift=max(1, round(common_height * MAX_ROW_SHIFT)),
                             scale_range=MAX_RELATIVE_SCALE, level_prior=LEVEL_PRIOR)
    timings["visual_matching"] = round(perf_counter() - stage, 3)
    scales, offsets = matches.scales, matches.offsets
    active_labels = [labels[index] for index in active]
    visual_order = rank_orders(matches.scores, count=1)[0][1]
    verifier = JoinVerifier(active_images, matches, common_height, ocr, workers=ocr_workers)
    print(f"Compared all {len(active) * (len(active) - 1)} directed joins. OCR verification: {ocr.status}.", flush=True)
    try:
        stage = perf_counter()
        scores = verifier.score_pairs()
        timings["pair_ocr"] = round(perf_counter() - stage, 3)
        pair_ranked = rank_orders(scores)
        print("Refining the order using text across three neighboring strips..." if ocr.enabled
              else f"Visual-only ordering: {ocr.reason}", flush=True)
        stage = perf_counter()
        ranked, refinement = refine_orders(pair_ranked, scores, verifier)
        timings["context_verification"] = round(perf_counter() - stage, 3)
        if ocr.enabled:
            print("Checking the complete reconstructed page with OCR...", flush=True)
        stage = perf_counter()
        ranked, document_check = verify_document_orders(ranked, pair_ranked, verifier)
        if document_check["enabled"]:
            for key in ("baseline_order", "selected_order"):
                document_check[key] = [active_labels[index] for index in document_check[key]]
            for candidate in document_check["candidates"]:
                candidate["order"] = [active_labels[index] for index in candidate["order"]]
        timings["document_verification"] = round(perf_counter() - stage, 3)
        join_report = build_join_report(ranked[0][1], ranked, scores, verifier, active_labels, refinement,
                                        document_check=document_check)
    finally:
        ocr.flush()
    best_score, active_order = ranked[0]
    stage = perf_counter()
    # Edge traces separate neighbors well, but cut shadows make them noisy
    # for exact height. Text rows just inside each chosen cut set the final rows.
    seams = [seam_row_shifts(active_images[a], active_images[b]) for a, b in zip(active_order, active_order[1:])]
    maps = level_row_maps(common_height, [step for step, _ in seams], [coverage for _, coverage in seams])
    timings["final_alignment"] = round(perf_counter() - stage, 3)
    preview, placements = assemble_preview(active_images, active_order, maps)
    order = tuple(active[index] for index in active_order) + tuple(low_ink)
    placements.extend([None] * len(low_ink))
    joins = max(1, len(active) - 1)
    alternative_edges = [set(zip(candidate, candidate[1:])) for _, candidate in ranked]
    join_details = []
    for a, b in zip(active_order, active_order[1:]):
        competitors = [j for j in range(len(active)) if j not in (a, b)]
        runner_up = max(competitors, key=lambda j: scores[a, j]) if competitors else None
        margin = float(scores[a, b] - scores[a, runner_up]) if runner_up is not None else None
        support = sum((a, b) in edges for edges in alternative_edges)
        join_details.append({
            "left": labels[active[a]], "right": labels[active[b]], "score": float(scores[a, b]),
            "next_best_right": labels[active[runner_up]] if runner_up is not None else None,
            "margin_over_next_best_right": margin,
            "present_in_top_orders": support,
            "ambiguous": (margin is not None and margin < 0.02) or support < len(ranked),
            "confidence": join_report["joins"][len(join_details)]["confidence"],
        })
    report = {
        "estimated": True,
        "input_directories": [str(directory) for directory in directories],
        "assumptions": "Upright strips from one page; missing strips are not identified.",
        "search": "exhaustive" if len(active) <= 8 else "mixed_integer_global",
        "optimal_for_pairwise_scores": active_order == pair_ranked[0][1],
        "matching": {
            "method": "strips leveled on one page grid by their shared text rows; full-height edge ink and "
                      "text-row traces; shared FFT registration and constrained local alignment",
            "ordered_pairs_compared": len(active) * (len(active) - 1),
            "edge_weight": 0.8, "text_row_weight": 0.2,
            "affine_weight": 0.6, "locally_aligned_weight": 0.4,
            "low_ink_threshold": 0.0005,
            "max_row_shift": MAX_ROW_SHIFT, "max_relative_scale": MAX_RELATIVE_SCALE,
            "level_prior_per_percent": LEVEL_PRIOR,
        },
        "verification": {
            "ocr": ocr.metadata(),
            "ocr_workers": verifier.workers,
            "visual_order": [active_labels[index] for index in visual_order],
            "pairwise_order": [active_labels[index] for index in pair_ranked[0][1]],
            "order_changed_by_verification": active_order != visual_order,
            "pair_score_weights": (
                {"visual": 0.25, "ocr_ink_evidence": 0.25, "english_plausibility": 0.5} if verifier.fragments is not None
                else {"visual": 0.5, "ocr_ink_evidence": 0.5} if ocr.enabled else {"visual": 1.0}
            ),
            "pair_score_fallback": "All directed pairs remain available. Missing OCR evidence earns no text bonus. OCR-off mode uses all visual scores.",
            "ocr_pairs_checked": sum(len(order) == 2 for order in verifier.cache) if ocr.enabled else 0,
            "ocr_sampling": "Up to 24 text lines distributed over the full height; analysis height capped at 2400 pixels. Final-page OCR reads the complete analysis image.",
            "candidate_selection": "Every directed pair is checked. Context search tests swaps, single moves, and moves of groups of up to six strips.",
            "context_score_weights": {"pair_score": 0.25, "three_strip_ocr": 0.75} if refinement["enabled"] else {"pair_score": 1.0},
            "refinement": refinement,
            "document_check": document_check,
            "html_report": f"{output_dir.name}/join_report.html", "json_report": f"{output_dir.name}/join_report.json",
        },
        "text_strip_count": len(active),
        "unplaced_low_ink_count": len(low_ink),
        "document_contents": "Text-bearing strips only. Low-ink strips stay in normalized_strips with unresolved positions.",
        "common_height": common_height,
        "mean_score": best_score / joins,
        "score_kind": "full_page_ocr_ink_evidence" if document_check["enabled"] else "pairwise_or_context",
        "runner_up_gap": (best_score - ranked[1][0]) / joins if len(ranked) > 1 else None,
        "order": [
            {
                "position": None if index in low_ink else number, "source": labels[index],
                "source_path": str(paths[index]), "preview": placement,
                # Scale and top row of the strip on the leveled page grid.
                "leveling": None if index in low_ink else leveling[active.index(index)],
                "ink_fraction": fractions[index],
                "position_status": "unplaced_low_ink" if index in low_ink else "estimated",
            }
            for number, (index, placement) in enumerate(zip(order, placements), 1)
        ],
        "alternatives": [
            {"sources": [labels[active[index]] for index in candidate], "mean_score": value / joins}
            for value, candidate in ranked
        ],
        "joins": join_details,
        "matrix_sources": [labels[index] for index in active],
        "pairwise_scores": [
            [float(value) if np.isfinite(value) else None for value in row] for row in scores
        ],
        "pairwise_scales": scales.tolist(),
        "pairwise_offsets": offsets.tolist(),
    }
    if len(directories) == 1:
        report["input_directory"] = str(directories[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    document_path = output_dir / "document.png"
    temporary = output_dir / ".document.tmp.png"
    try:
        if not cv2.imwrite(str(temporary), preview):
            raise OSError(f"Could not save document to {output_dir}")
        temporary.replace(document_path)
    finally:
        temporary.unlink(missing_ok=True)
    report["result"] = f"{output_dir.name}/document.png"
    write_join_reports(output_dir, join_report, verifier, active_order)
    timings["total"] = round(perf_counter() - started, 3)
    report["sorting_timings_seconds"] = timings
    write_json(output_dir.parent / "order.json", report)
    return report


def sort_submission(
    submission_dir: Path, *, ocr_mode: str = "auto", ocr_language: str = "eng", ocr_workers: int = DEFAULT_OCR_WORKERS,
) -> dict:
    """Sort a submission's pooled strips and attach per-photo provenance."""
    submission = Submission(Path(submission_dir))
    manifest = submission.read_manifest()
    provenance = {entry["filename"]: entry for entry in manifest.get("strips", [])}
    normalized_names = {
        path.name for path in submission.normalized_strips.glob("strip*.png") if path.stem[5:].isdigit()
    }
    if provenance and normalized_names != set(provenance):
        raise ValueError("Normalized strips do not match detection; rerun normalization for this submission.")
    report = sort_strips(submission.normalized_strips, submission.final_document,
                         ocr_mode=ocr_mode, ocr_language=ocr_language, ocr_workers=ocr_workers)
    report.update(
        submission_id=submission.directory.name, input_directory="normalized_strips",
        input_directories=["normalized_strips"], source_images=manifest.get("sources", []),
    )
    for entry in report["order"]:
        name = Path(entry["source_path"]).name
        entry["source_path"] = f"normalized_strips/{name}"
        if name in provenance:
            entry["source_image"] = provenance[name]["source_image"]
            entry["source_strip_number"] = provenance[name]["source_strip_number"]
    write_json(submission.order_path, report)
    manifest.update(status="complete", result="final_document/document.png", join_report="final_document/join_report.html")
    manifest.pop("timings_seconds", None)
    manifest["sorting_timings_seconds"] = report["sorting_timings_seconds"]
    manifest.pop("error", None)
    submission.save_manifest(manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission_dir", type=Path, help="Submission folder: backend/img/<submission_id> (from the repository root).",
    )
    parser.add_argument("--ocr", choices=("auto", "required", "off"), default="auto")
    parser.add_argument("--ocr-language", default="eng", help="Installed Tesseract language(s), e.g. eng or eng+deu.")
    parser.add_argument("--ocr-workers", type=int, default=DEFAULT_OCR_WORKERS)
    args = parser.parse_args()
    try:
        report = sort_submission(args.submission_dir, ocr_mode=args.ocr,
                                 ocr_language=args.ocr_language, ocr_workers=args.ocr_workers)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print("Estimated text order: " + " -> ".join(
        entry["source"] for entry in report["order"] if entry["position_status"] == "estimated"
    ))
    print(f"Saved {args.submission_dir / report['result']}")
    print(f"Join report: {args.submission_dir / report['verification']['html_report']}")
    print(f"{report['unplaced_low_ink_count']} blank/low-ink strips have unresolved positions.")
    print("Review the preview; scores rank candidates and are not confidence probabilities.")


if __name__ == "__main__":
    main()
