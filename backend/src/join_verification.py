"""Score reconstructed cuts using visible strokes and local OCR evidence."""

from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np

from strip_matching import strip_ink


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
    def __init__(self, images, matches, height, ocr, workers=4):
        self.matches, self.height, self.ocr = matches, height, ocr
        self.workers = max(1, min(int(workers), 8))
        self.cache, self.lock = {}, threading.Lock()
        self.inks = []
        for image in images:
            width = max(4, round(image.shape[1] * height / image.shape[0]))
            ink = cv2.resize(strip_ink(image), (width, height))
            # Physical cut shadows are not letter strokes. Trim only the OCR
            # analysis copy; keep every original pixel in document.png.
            trim = min(2, (width - 2) // 2)
            self.inks.append(ink[:, trim:width - trim] if trim else ink)

    def render(self, order):
        rows = np.arange(self.height, dtype=np.float32)
        maps = [rows]
        for a, b in zip(order, order[1:]):
            previous, warp = maps[-1], self.matches.warps[a, b]
            mapping = np.interp(previous, rows, warp).astype(np.float32)
            # Extend the row mapping, not the last ink row: clamping would
            # repeat text at the tips when composing several aligned strips.
            for endpoint, neighbor, outside in ((0, 1, previous < 0), (-1, -2, previous > rows[-1])):
                slope = (warp[neighbor] - warp[endpoint]) / (rows[neighbor] - rows[endpoint])
                mapping[outside] = warp[endpoint] + (previous[outside] - rows[endpoint]) * slope
            maps.append(mapping)
        parts = []
        for index, mapping in zip(order, maps):
            ink = self.inks[index]
            xx = np.broadcast_to(np.arange(ink.shape[1], dtype=np.float32), ink.shape).copy()
            parts.append(cv2.remap(ink, xx, np.broadcast_to(mapping[:, None], ink.shape).copy(), cv2.INTER_LINEAR))
        return np.hstack(parts), np.cumsum([part.shape[1] for part in parts])[:-1]

    def evaluate(self, order):
        order = tuple(order)
        with self.lock:
            if order in self.cache:
                return self.cache[order]
        ink, seams = self.render(order)
        occupied = np.flatnonzero((ink > 0.2).any(1))
        words = []
        if occupied.size and self.ocr.enabled:
            top, bottom = max(0, int(occupied[0]) - 10), min(self.height, int(occupied[-1]) + 11)
            gray = np.rint(255 * (1 - ink[top:bottom])).astype(np.uint8)
            gray = cv2.copyMakeBorder(
                cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255,
            )
            for word in self.ocr.recognize(gray):
                x, y, width, height = word["box"]
                words.append({**word, "box": [(x - 20) / 2, (y - 20) / 2 + top, width / 2, height / 2]})
        result = {"seams": [measure_seam(ink, int(seam), words) for seam in seams]}
        with self.lock:
            self.cache[order] = result
        return result

    def evaluate_many(self, orders):
        needed = sorted({tuple(order) for order in orders} - self.cache.keys())
        if needed:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(self.evaluate, needed))

    def score_pairs(self):
        size = len(self.inks)
        pairs = [(i, j) for i in range(size) for j in range(size) if i != j]
        self.evaluate_many(pairs)
        scores = self.matches.scores.copy()
        if self.ocr.enabled:
            for a, b in pairs:
                seam = self.cache[a, b]["seams"][0]
                # No ink means no verification evidence, not a perfect match.
                if seam["text_lines"] >= 3:
                    scores[a, b] = (0.65 * scores[a, b] + 0.05 * seam["stroke_score"] + 0.30 * seam["ocr_score"])
        return scores


def order_neighbors(order):
    """Swaps and insertions, including non-adjacent strip corrections."""
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
    return candidates


def refine_orders(ranked, scores, verifier, rounds=2, proposals_per_round=24):
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
            evaluated[order] = 0.8 * pair_mean(order) + 0.2 * float(context)

    evaluate([order for _, order in ranked])
    current = max(evaluated, key=evaluated.get)
    for _ in range(rounds):
        proposals = order_neighbors(current)
        candidates = sorted((order for order in proposals if order not in evaluated),
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
