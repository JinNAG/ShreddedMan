"""Score reconstructed cuts using visible strokes and local OCR evidence."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gzip
from pathlib import Path
import re
import threading

import cv2
import numpy as np

from strip_matching import refine_row_maps_from_ink, stabilize_row_maps, strip_ink
from strip_ocr import DEFAULT_OCR_WORKERS


@lru_cache(maxsize=1)
def _english_words() -> frozenset[str]:
    """A local public-domain word list for reviewing complete English words."""
    path = Path(__file__).with_name("english_words.txt.gz")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return frozenset(line.strip() for line in source
                         if line.strip() and not line.startswith("#"))


def _lexical_ratio(words: list[dict]) -> tuple[float | None, int]:
    tokens = [re.sub("[^a-z]", "", word["text"].lower()) for word in words]
    tokens = [token for token in tokens if len(token) >= 4]
    if not tokens:
        return None, 0
    vocabulary = _english_words()
    return sum(token in vocabulary for token in tokens) / len(tokens), len(tokens)


def measure_seam(ink: np.ndarray, seam: int, words: list[dict]) -> dict:
    """Account for unread ink, rather than averaging only successful OCR hits."""
    height, width = ink.shape
    left, right = max(0, seam - 14), min(width, seam + 14)
    foreground = ink[:, left:right] > 0.2
    credit = np.zeros(foreground.shape, np.float32)
    recognized = np.zeros(foreground.shape, bool)
    crossing = []
    for word in words:
        x, y, w, h = word["box"]
        if not x + 2 < seam < x + w - 2:
            continue
        x0, x1 = max(0, round(x) - left), min(right - left, round(x + w) - left)
        y0, y1 = max(0, round(y)), min(height, round(y + h))
        if x0 >= x1 or y0 >= y1:
            continue
        crossing.append(word)
        credit[y0:y1, x0:x1] = np.maximum(credit[y0:y1, x0:x1], word["confidence"])
        recognized[y0:y1, x0:x1] = True

    pixel_count = int(foreground.sum())
    evidence_score = float(np.sum(foreground * credit) / max(1, pixel_count))
    coverage = float(np.sum(foreground & recognized) / max(1, pixel_count))
    # Adjacent ink traces provide a language-independent consistency check.
    a = ink[:, max(0, seam - 4):seam].mean(1)
    b = ink[:, seam:min(width, seam + 4)].mean(1)
    a = cv2.GaussianBlur(a[:, None], (1, 0), 0, sigmaY=1).ravel()
    b = cv2.GaussianBlur(b[:, None], (1, 0), 0, sigmaY=1).ravel()
    stroke = float(2 * np.sum(a * b) / max(float(np.sum(a * a) + np.sum(b * b)), 1e-9))

    active_rows = (foreground.sum(1) >= 2).astype(np.uint8)
    active_rows = cv2.morphologyEx(active_rows[:, None], cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)).ravel()
    changes = np.diff(np.pad(active_rows.astype(int), (1, 1)))
    lines = []
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        pixels = int(foreground[start:end].sum())
        if end - start < 3 or pixels < 8:
            continue
        score = float(np.sum(foreground[start:end] * credit[start:end]) / pixels)
        line_words = [word for word in crossing if word["box"][1] < end and word["box"][1] + word["box"][3] > start]
        lines.append({"y_start": int(start), "y_end": int(end), "ocr_score": score,
                      "text": " ".join(word["text"] for word in line_words)})
    return {
        "ocr_score": evidence_score, "ocr_confidence": float(np.mean([w["confidence"] for w in crossing])) if crossing else 0.0,
        "ink_coverage": coverage, "stroke_score": float(np.clip(stroke, 0, 1)),
        "ink_pixels": pixel_count, "text_lines": len(lines), "recognized_tokens": len(crossing),
        "recognized_lines": sum(bool(line["text"]) for line in lines),
        "fragments": crossing, "lines": lines,
    }


class JoinVerifier:
    def __init__(self, images, matches, height, ocr, workers=DEFAULT_OCR_WORKERS):
        self.matches, self.ocr = matches, ocr
        # Keep letter sizes in Tesseract's useful range on high-resolution
        # phone photos. This changes analysis only, never the saved strips.
        self.height = min(height, 2400)
        self.warps = matches.warps
        if self.height != height:
            rows = (np.arange(self.height) + 0.5) * height / self.height - 0.5
            self.warps = np.empty((*matches.warps.shape[:2], self.height), np.float32)
            for a in range(len(images)):
                for b in range(len(images)):
                    self.warps[a, b] = (np.interp(rows, np.arange(height), matches.warps[a, b]) + 0.5) * self.height / height - 0.5
        height = self.height
        self.workers = max(1, min(int(workers), 8))
        self.cache, self.document_cache, self.lock = {}, {}, threading.Lock()
        self.inks = []
        for image in images:
            width = max(4, round(image.shape[1] * height / image.shape[0]))
            ink = cv2.resize(strip_ink(image), (width, height))
            # Physical cut shadows are not letter strokes. Trim only the OCR
            # analysis copy; keep every original pixel in document.png.
            trim = min(2, (width - 2) // 2)
            self.inks.append(ink[:, trim:width - trim] if trim else ink)

    def render(self, order, *, full_page=False):
        rows = np.arange(self.height, dtype=np.float32)
        maps = [rows]
        for a, b in zip(order, order[1:]):
            previous, warp = maps[-1], self.warps[a, b]
            mapping = np.interp(previous, rows, warp).astype(np.float32)
            # Extend the row mapping, not the last ink row: clamping would
            # repeat text at the tips when composing several aligned strips.
            for endpoint, neighbor, outside in ((0, 1, previous < 0), (-1, -2, previous > rows[-1])):
                slope = (warp[neighbor] - warp[endpoint]) / (rows[neighbor] - rows[endpoint])
                mapping[outside] = warp[endpoint] + (previous[outside] - rows[endpoint]) * slope
            maps.append(mapping)
        if full_page:
            if hasattr(self.matches, "scores"):
                maps = stabilize_row_maps(order, maps, self.warps, self.matches.scores)
                maps = refine_row_maps_from_ink(order, maps, self.inks)
            def extend(points, coordinates, values):
                result = np.interp(points, coordinates, values)
                for endpoint, neighbor, outside in ((0, 1, points < coordinates[0]),
                                                     (-1, -2, points > coordinates[-1])):
                    slope = (values[neighbor] - values[endpoint]) / (coordinates[neighbor] - coordinates[endpoint])
                    result[outside] = values[endpoint] + (points[outside] - coordinates[endpoint]) * slope
                return result

            # The preview canvas includes all strip tips after vertical
            # alignment; the final OCR check must inspect those rows too.
            bounds = [extend(np.array([0, self.height - 1]), mapping, rows) for mapping in maps]
            output_rows = np.arange(np.floor(min(b[0] for b in bounds)),
                                    np.ceil(max(b[1] for b in bounds)) + 1, dtype=np.float32)
            maps = [extend(output_rows, rows, mapping).astype(np.float32) for mapping in maps]
        parts = []
        for index, mapping in zip(order, maps):
            ink = self.inks[index]
            shape = (len(mapping), ink.shape[1])
            xx = np.broadcast_to(np.arange(ink.shape[1], dtype=np.float32), shape).copy()
            parts.append(cv2.remap(ink, xx, np.broadcast_to(mapping[:, None], shape).copy(), cv2.INTER_LINEAR))
        return np.hstack(parts), np.cumsum([part.shape[1] for part in parts])[:-1]

    def evaluate(self, order):
        """Read separated text lines, with a fixed budget across the full height.

        Feeding a tall, narrow strip pair directly to OCR makes its layout
        detector skip readable fragments. Compact the lines without resizing
        individual letters or joining unrelated fragments into one text row.
        """
        order = tuple(order)
        with self.lock:
            if order in self.cache:
                return self.cache[order]
        ink, seams = self.render(order)
        words, sampled_lines, available_lines = [], 0, 0
        analyzed_ink = ink
        if self.ocr.enabled:
            active = ((ink > 0.2).sum(1) >= max(3, ink.shape[1] * 0.04)).astype(np.uint8)
            active = cv2.morphologyEx(active[:, None], cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)).ravel()
            changes = np.diff(np.pad(active.astype(int), (1, 1)))
            # Very tall components are usually logos or cut shadows. They
            # still contribute to full-height visual matching, not text OCR.
            max_text_height = max(45, round(self.height * 0.018))
            bounds = [(max(0, int(a) - 3), min(self.height, int(b) + 3))
                      for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
                      if 5 <= b - a <= max_text_height]
            available_lines = len(bounds)
            selected = np.linspace(0, len(bounds) - 1, min(24, len(bounds))).round().astype(int)
            regions, panels, y = [], [], 10
            analyzed_ink = np.zeros_like(ink)
            for index in selected:
                top, bottom = bounds[index]
                panel = np.rint(255 * (1 - ink[top:bottom])).astype(np.uint8)
                panels.append(cv2.copyMakeBorder(panel, 0, 8, 0, 0, cv2.BORDER_CONSTANT, value=255))
                regions.append((y, y + bottom - top, top))
                analyzed_ink[top:bottom] = ink[top:bottom]
                y += bottom - top + 8
            sampled_lines = len(regions)
            if panels:
                gray = cv2.copyMakeBorder(np.vstack(panels), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                for word in self.ocr.recognize(gray):
                    x, y, width, height = np.asarray(word["box"]) / 2
                    for start, end, top in regions:
                        if start <= y + height / 2 < end:
                            words.append({**word, "box": [float(x - 10), float(y - start + top), float(width), float(height)]})
                            break
        result = {"seams": [measure_seam(analyzed_ink, int(seam), words) for seam in seams],
                  "sampled_text_lines": sampled_lines, "available_text_lines": available_lines}
        with self.lock:
            self.cache[order] = result
        return result

    def evaluate_many(self, orders):
        needed = sorted({tuple(order) for order in orders} - self.cache.keys())
        if needed:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(self.evaluate, needed))

    def score_pairs(self):
        """Keep every possible neighbor available to the global order search.

        A weak visual match can be a correct join (for example, a cut through
        a wide letter). Pruning on visual rank alone excludes real solutions.
        """
        size = len(self.inks)
        visual = self.matches.scores
        if not self.ocr.enabled:
            return visual.copy()
        pairs = [(a, b) for a in range(size) for b in range(size) if a != b]
        self.evaluate_many(pairs)
        scores = np.full_like(visual, -np.inf)
        for pair in pairs:
            scores[pair] = 0.5 * visual[pair] + 0.5 * self.cache[pair]["seams"][0]["ocr_score"]
        return scores

    def evaluate_document(self, order):
        """Read the complete reconstructed page, accounting for unread ink.

        This uses a consistently sized analysis rendering of the same strip
        order and row maps as the preview. No text is synthesized or corrected.
        """
        order = tuple(order)
        if order in self.document_cache:
            return self.document_cache[order]
        ink, seams = self.render(order, full_page=True)
        gray = np.rint(255 * (1 - ink)).astype(np.uint8)
        gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        words = [{**word, "box": [word["box"][0] - 10, word["box"][1] - 10, *word["box"][2:]]}
                 for word in self.ocr.recognize(gray)]
        evidence = [measure_seam(ink, int(seam), words) for seam in seams]
        lexical_joins = []
        if self.ocr.language == "eng":
            for position, item in enumerate(evidence, 1):
                ratio, count = _lexical_ratio(item["fragments"])
                item["lexical_word_ratio"] = ratio
                item["lexical_word_count"] = count
                if count >= 10 and ratio is not None and ratio < 0.3:
                    lexical_joins.append(position)
        lexical_score, lexical_count = _lexical_ratio(words) if self.ocr.language == "eng" else (None, 0)
        ink_score = float(np.mean([item["ocr_score"] for item in evidence])) if evidence else 0.0
        use_lexicon = lexical_score is not None and lexical_count >= max(30, len(seams) * 5)
        result = {
            "score": 0.6 * ink_score + 0.4 * lexical_score if use_lexicon else ink_score,
            "ocr_ink_score": ink_score,
            "ink_coverage": float(np.mean([item["ink_coverage"] for item in evidence])) if evidence else 0.0,
            "recognized_tokens": len(words), "text": " ".join(word["text"] for word in words),
            "seams": evidence, "analysis_height": ink.shape[0],
            "lexical_word_ratio": lexical_score, "lexical_word_count": lexical_count,
            "lexical_suspect_joins": lexical_joins, "lexicon_used": use_lexicon,
        }
        self.document_cache[order] = result
        return result


def order_neighbors(order):
    """Swaps, single moves, and moves of intact groups of neighboring strips."""
    candidates = {}
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            swapped = list(order)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            candidates[tuple(swapped)] = {"operation": "swap", "positions": [i + 1, j + 1]}
        for j in range(len(order)):
            if i == j:
                continue
            moved = list(order)
            moved.insert(j, moved.pop(i))
            candidates.setdefault(tuple(moved), {"operation": "move", "positions": [i + 1, j + 1]})
    for length in range(2, min(7, len(order))):
        for start in range(len(order) - length + 1):
            block = order[start:start + length]
            rest = order[:start] + order[start + length:]
            for destination in range(len(rest) + 1):
                if destination != start:
                    candidates.setdefault(rest[:destination] + block + rest[destination:], {
                        "operation": "move_block", "positions": [start + 1, start + length, destination + 1],
                    })
    return candidates


def refine_orders(ranked, scores, verifier, rounds=4, proposals_per_round=24):
    """Bounded context search; only accept improvements to the complete score.

    Context scores include every triple in each order. Cached unaffected triples
    are reused, so a swap cannot improve one cut while hiding damage elsewhere.
    This search does not claim a global optimum for the context objective.
    """
    size = len(ranked[0][1])
    metadata = {"enabled": bool(verifier.ocr.enabled and size >= 3), "rounds": 0,
                "accepted_changes": [], "evaluated_orders": 0, "globally_optimal": False,
                "stop_reason": "OCR disabled or fewer than three text strips."}
    if not metadata["enabled"]:
        return ranked, metadata
    evaluated = {}

    def pair_mean(order):
        return sum(float(scores[a, b]) for a, b in zip(order, order[1:])) / (size - 1)

    def evaluate(orders):
        verifier.evaluate_many(order[i:i + 3] for order in orders for i in range(size - 2))
        for order in orders:
            context = np.mean([seam["ocr_score"] for i in range(size - 2)
                               for seam in verifier.cache[order[i:i + 3]]["seams"]])
            evaluated[order] = 0.25 * pair_mean(order) + 0.75 * float(context)

    evaluate([order for _, order in ranked])
    current = max(evaluated, key=evaluated.get)
    for _ in range(rounds):
        proposals = order_neighbors(current)
        candidates = sorted((order for order in proposals if order not in evaluated and np.isfinite(pair_mean(order))),
                            key=lambda order: (-pair_mean(order), order))[:proposals_per_round]
        if not candidates:
            metadata["stop_reason"] = "No untested proposals remain."
            break
        evaluate(candidates)
        metadata["rounds"] += 1
        best = max(evaluated, key=evaluated.get)
        if evaluated[best] <= evaluated[current] + 1e-9:
            metadata["stop_reason"] = "No tested change improved the complete score."
            break
        metadata["accepted_changes"].append({
            **proposals.get(best, {"operation": "alternative", "positions": []}),
            "order_before": list(current), "order_after": list(best),
            "score_before": evaluated[current], "score_after": evaluated[best],
        })
        current = best
    else:
        metadata["stop_reason"] = "Context-search round limit reached."
    metadata["evaluated_orders"] = len(evaluated)
    result = sorted(((value * (size - 1), order) for order, value in evaluated.items()),
                    key=lambda item: (-item[0], item[1]))[:5]
    return result, metadata


def verify_document_orders(ranked, pair_ranked, verifier, pair_scores=None):
    """Use whole-page OCR to choose among context finalists and the baseline.

    Keeping the pairwise baseline in the comparison prevents local context
    improvements from silently damaging the full page. This is a bounded
    verification step, not proof that every character or join is correct.
    """
    if not verifier.ocr.enabled or len(ranked[0][1]) < 2:
        return ranked, {"enabled": False, "reason": "OCR disabled or fewer than two text strips."}
    baseline = pair_ranked[0][1]
    orders = list(dict.fromkeys([baseline, *[order for _, order in ranked[:3]]]))
    with ThreadPoolExecutor(max_workers=verifier.workers) as pool:
        readings = list(pool.map(verifier.evaluate_document, orders))

    def weak_joins(reading):
        weak = {i for i, item in enumerate(reading["seams"], 1)
                if item["ocr_score"] < 0.55 or item["ink_coverage"] < 0.5
                or item["recognized_lines"] < 3}
        return sorted(weak | set(reading.get("lexical_suspect_joins", [])))

    # A high pair score can hide several broken cuts in an otherwise plausible
    # path. Move intact groups across the cuts flagged by complete-page OCR,
    # then let complete-page OCR judge the small set of resulting orders.
    repair_rounds = repair_candidates_checked = 0
    if pair_scores is not None:
        for _ in range(8):
            selected = min(range(len(orders)), key=lambda i: (-readings[i]["score"],
                                                              orders[i] != baseline, orders[i]))
            current, reading = orders[selected], readings[selected]
            lexical_suspect = reading.get("lexical_suspect_joins", [])
            suspect = lexical_suspect if len(lexical_suspect) >= 2 else weak_joins(reading)
            if len(suspect) < 2 or reading["score"] >= 0.7:
                break
            suspect_edges = {(current[i - 1], current[i]) for i in suspect}

            def priority(candidate):
                changed = len(suspect_edges - set(zip(candidate, candidate[1:])))
                pair_mean = sum(float(pair_scores[a, b]) for a, b in zip(candidate, candidate[1:])) / (len(candidate) - 1)
                return (-changed, -pair_mean, candidate)

            proposals = set(order_neighbors(current))
            # A true neighbor can have a poor two-strip OCR score when its
            # letters need context from both sides. Joining a promising pair
            # and moving it into a weak cut lets full-page OCR test that
            # coordinated change directly, without requiring a bad
            # intermediate page to win a greedy round.
            if reading["score"] < 0.55:
                for a in current:
                    successors = sorted((b for b in current if b != a),
                                        key=lambda b: (-pair_scores[a, b], b))[:4]
                    for b in successors:
                        remaining = [item for item in current if item not in (a, b)]
                        for position in suspect:
                            left = current[position - 1]
                            if left in (a, b):
                                continue
                            destination = remaining.index(left) + 1
                            proposals.add(tuple(remaining[:destination] + [a, b] + remaining[destination:]))
            proposals = [candidate for candidate in proposals
                         if candidate not in verifier.document_cache
                         and len(suspect_edges - set(zip(candidate, candidate[1:]))) >= 2]
            candidates = sorted(proposals, key=priority)[:48]
            if not candidates:
                break
            with ThreadPoolExecutor(max_workers=verifier.workers) as pool:
                checked = list(pool.map(verifier.evaluate_document, candidates))
            orders.extend(candidates)
            readings.extend(checked)
            repair_rounds += 1
            repair_candidates_checked += len(candidates)
            if max(item["score"] for item in checked) <= reading["score"] + 1e-9:
                break
    # On a tie, retain the pairwise baseline instead of making an unsupported
    # change. Every candidate includes every strip exactly once.
    selected = min(range(len(orders)), key=lambda i: (-readings[i]["score"], orders[i] != baseline, orders[i]))
    final_order, final = orders[selected], readings[selected]
    count = len(final_order) - 1
    result = sorted(((reading["score"] * count, order) for order, reading in zip(orders, readings)),
                    key=lambda item: (-item[0], item[1] != baseline, item[1]))[:5]
    summary = {key: value for key, value in final.items() if key != "seams"}
    return result, {
        "enabled": True, "method": "Full-page OCR ink evidence and English word plausibility when enough text is available.",
        "score_note": "Heuristic readability evidence, not an accuracy percentage or proof of a correct order. English word checks only rank candidates; source pixels and OCR text are not corrected.",
        "baseline_order": list(baseline), "selected_order": list(final_order),
        "baseline_score": readings[0]["score"], "final_score": final["score"],
        "changed_from_pairwise": final_order != baseline, "evaluated_orders": len(orders),
        "repair_rounds": repair_rounds, "repair_candidates_checked": repair_candidates_checked,
        "candidates": [{"order": list(order), "score": reading["score"], "ink_coverage": reading["ink_coverage"]}
                       for order, reading in zip(orders, readings)],
        "suspect_joins": weak_joins(final),
        "final_reading": summary,
    }
