"""Estimate the order of upright, normalized strips from a single page.

Usage:
    python src/sort_strips.py img/<submission_id>

Writes one assembled image to final_document/document.png. Source mappings,
scores, and alternatives go to order.json at the submission root. Normalized
PNGs keep their original filenames; blank/low-ink positions remain unresolved.
"""

import argparse
import heapq
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

from strip_matching import ink_fraction, ink_profiles, match_profiles, score_pairs
from strip_order import optimize_orders
from submission import Submission, write_json


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


def assemble_preview(
    images: list[np.ndarray], order: tuple[int, ...], scales: np.ndarray,
    offsets: np.ndarray, common_height: int, warps: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Render the estimated joins; only the preview is resized and aligned."""
    if warps is not None:
        return _assemble_warped_preview(images, order, warps, common_height)
    vertical_scales = [1.0]
    vertical_offsets = [0.0]
    for a, b in zip(order, order[1:]):
        vertical_offsets.append(vertical_offsets[-1] + vertical_scales[-1] * offsets[a, b])
        vertical_scales.append(vertical_scales[-1] * scales[a, b])
    y_positions = [round(float(y)) for y in vertical_offsets]
    heights = [max(1, round(common_height * float(scale))) for scale in vertical_scales]
    top = min(y_positions)
    bottom = max(y + height for y, height in zip(y_positions, heights))
    # Photos taken at different distances need a common horizontal scale too.
    widths = [
        max(1, round(images[index].shape[1] * common_height / images[index].shape[0]))
        for index in order
    ]
    total_width = sum(widths)
    preview = np.full((bottom - top, total_width, 3), 255, dtype=np.uint8)
    placements = []
    x = 0
    for index, y, height, width in zip(order, y_positions, heights, widths):
        image = images[index]
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        rgb = np.rint(image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        preview[y - top : y - top + height, x : x + width] = rgb
        placements.append({"x": x, "y": y - top, "width": width, "height": height})
        x += width
    return preview, placements


def _extrapolate(points, coordinates, values):
    """Linear extension preserves the strip tips when composing row maps."""
    result = np.interp(points, coordinates, values)
    for outside, endpoint, neighbor in (
        (points < coordinates[0], 0, 1), (points > coordinates[-1], -1, -2)
    ):
        slope = (values[neighbor] - values[endpoint]) / (coordinates[neighbor] - coordinates[endpoint])
        result[outside] = values[endpoint] + (points[outside] - coordinates[endpoint]) * slope
    return result


def _assemble_warped_preview(images, order, warps, height):
    rows = np.arange(height, dtype=np.float32)
    maps = [rows]
    for a, b in zip(order, order[1:]):
        maps.append(_extrapolate(maps[-1], rows, warps[a, b]))
    bounds = [_extrapolate(np.array([0, height - 1]), mapping, rows) for mapping in maps]
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


def sort_strips(input_dirs: Path | list[Path], output_dir: Path) -> dict:
    """Render one document and write its order report beside the output folder."""
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
    common_height = int(np.median([image.shape[0] for image in images]))
    fractions = [ink_fraction(image) for image in images]
    active = [index for index, fraction in enumerate(fractions) if fraction >= 0.0005]
    low_ink = [index for index in range(len(paths)) if index not in active]
    if not active:
        raise ValueError("No usable ink found; all strips are blank or too faint to order.")
    profiles = [ink_profiles(images[index], common_height) for index in active]
    if len(active) > 1 and not any(np.any(profile[:2] > 0) for profile in profiles):
        raise ValueError("No usable ink near the strip edges; cannot estimate an order.")

    matches = match_profiles(profiles)
    scores, scales, offsets = matches.scores, matches.scales, matches.offsets
    ranked = rank_orders(scores)
    best_score, active_order = ranked[0]
    preview, placements = assemble_preview(
        [images[index] for index in active], active_order, scales, offsets, common_height, matches.warps
    )
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
        })
    report = {
        "estimated": True,
        "input_directories": [str(directory) for directory in directories],
        "assumptions": "Upright strips from one page; missing strips are not identified.",
        "search": "exhaustive" if len(active) <= 8 else "mixed_integer_global",
        "optimal_for_pairwise_scores": True,
        "matching": {
            "method": "full-height edge ink and text-row traces; affine refinement and constrained local alignment",
            "ordered_pairs_compared": len(active) * (len(active) - 1),
            "edge_weight": 0.8, "text_row_weight": 0.2,
            "affine_weight": 0.6, "locally_aligned_weight": 0.4,
            "low_ink_threshold": 0.0005,
        },
        "text_strip_count": len(active),
        "unplaced_low_ink_count": len(low_ink),
        "document_contents": "Text-bearing strips only. Low-ink strips stay in normalized_strips with unresolved positions.",
        "common_height": common_height,
        "mean_score": best_score / joins,
        "runner_up_gap": (best_score - ranked[1][0]) / joins if len(ranked) > 1 else None,
        "order": [
            {
                "position": None if index in low_ink else number, "source": labels[index],
                "source_path": str(paths[index]), "preview": placement,
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
    write_json(output_dir.parent / "order.json", report)
    return report


def sort_submission(submission_dir: Path) -> dict:
    """Sort a submission's pooled strips and attach per-photo provenance."""
    submission = Submission(Path(submission_dir))
    manifest = submission.read_manifest()
    provenance = {entry["filename"]: entry for entry in manifest.get("strips", [])}
    normalized_names = {
        path.name for path in submission.normalized_strips.glob("strip*.png") if path.stem[5:].isdigit()
    }
    if provenance and normalized_names != set(provenance):
        raise ValueError("Normalized strips do not match detection; rerun normalization for this submission.")
    report = sort_strips(submission.normalized_strips, submission.final_document)
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
    manifest.update(status="complete", result="final_document/document.png")
    manifest.pop("error", None)
    submission.save_manifest(manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission_dir", type=Path, help="Submission folder: img/<submission_id>.",
    )
    args = parser.parse_args()
    try:
        report = sort_submission(args.submission_dir)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print("Estimated text order: " + " -> ".join(
        entry["source"] for entry in report["order"] if entry["position_status"] == "estimated"
    ))
    print(f"Saved {args.submission_dir / report['result']}")
    print(f"{report['unplaced_low_ink_count']} blank/low-ink strips have unresolved positions.")
    print("Review the preview; scores rank candidates and are not confidence probabilities.")


if __name__ == "__main__":
    main()
