"""Full-height ink matching with affine and constrained local registration."""

from dataclasses import dataclass

import cv2
import numpy as np


def strip_ink(image: np.ndarray) -> np.ndarray:
    """Remove paper brightness and ignore RGB hidden behind transparency."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected an 8-bit normalized PNG with an alpha channel.")
    if image.shape[0] < 2 or image.shape[1] < 4:
        raise ValueError("The strip is too small to compare text.")
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    alpha = image[:, :, 3].astype(np.float32) / 255
    if not np.any(alpha > 0.95):
        raise ValueError("The strip contains no opaque paper.")
    white = float(np.percentile(gray[alpha > 0.95], 90))
    return np.clip((white - gray - 25) / max(white - 65, 1), 0, 1) * alpha


def ink_fraction(image: np.ndarray) -> float:
    """Estimate usable ink, excluding cut-edge shadows and ragged tips.

    This is a low-ink heuristic, not OCR. Low-ink strips remain in the output,
    with their position explicitly unresolved.
    """
    ink = strip_ink(image)
    margin = max(1, round(ink.shape[1] * 0.15))
    tip = max(1, round(ink.shape[0] * 0.02))
    interior = ink[tip:-tip, margin:-margin]
    return float(np.mean(interior > 0.2)) if interior.size else 0.0


def ink_profiles(image: np.ndarray, height: int) -> np.ndarray:
    """Extract complete left/right edge traces and a supporting text-row trace.

    A common analysis grid makes band widths comparable across photo scales.
    Only the analysis features are resized; saved strip PNGs stay unchanged.
    """
    ink = cv2.resize(strip_ink(image), (56, height))
    signals = (ink[:, 3:7].mean(1), ink[:, -7:-3].mean(1), ink[:, 5:-5].mean(1))
    profiles = []
    for index, signal in enumerate(signals):
        signal = cv2.GaussianBlur(
            signal[:, None], (1, 0), sigmaX=0, sigmaY=1.2 if index == 2 else 1.0
        ).ravel()
        # Give body text throughout the page a say, even beneath large logos.
        energy = cv2.blur(signal[:, None] ** 2, (1, max(3, height // 6))).ravel()
        profiles.append(signal / np.sqrt(energy + 0.005))
    return np.asarray(profiles, dtype=np.float32)


def _affine_match(a, b, max_shift, scale_range):
    height = a.shape[1]
    padding = max_shift + int(np.ceil(height * scale_range)) + 2
    best = (0.0, 1.0, 0)
    padded = [np.pad(signal, (padding, padding))[:, None] for signal in a]
    # Refine scale to sub-pixel changes in total height instead of accepting
    # the several-pixel error left by a coarse scale grid.
    tried = set()
    for stage in range(2):
        if stage == 0:
            search = np.linspace(1 - scale_range, 1 + scale_range, 25)
        else:
            radius = scale_range / 10
            search = np.linspace(
                max(1 - scale_range, best[1] - radius),
                min(1 + scale_range, best[1] + radius), 17,
            )
        for scale in search:
            scaled_height = max(2, round(height * scale))
            if scaled_height in tried:
                continue
            tried.add(scaled_height)
            values = []
            for signal, target, source in zip(a, padded, b):
                scaled = cv2.resize(source[:, None], (1, scaled_height))
                cross = cv2.matchTemplate(target, scaled, cv2.TM_CCORR).ravel()
                energy = float(np.sum(signal * signal) + np.sum(scaled * scaled))
                values.append(np.clip(2 * cross / max(energy, 1e-9), 0, 1))
            value = 0.8 * values[0] + 0.2 * values[1]
            offsets = np.arange(len(value)) - padding
            value[np.abs(offsets) > max_shift] = -np.inf
            index = int(np.argmax(value))
            if value[index] > best[0] + 1e-7:
                best = (float(value[index]), scaled_height / height, int(offsets[index]))
    return best


def _local_alignment(a, b, scale, offset):
    """Allow small, smooth drift without letting unrelated text lines jump.

    Dynamic programming chooses shifts along the whole edge, penalizing both
    displacement and changes between windows. Blank windows favor no change.
    """
    height = len(a)
    rows = np.arange(height, dtype=np.float32)
    affine_y = (rows - offset) / scale
    radius = max(2, round(height * 0.003))
    shifts = np.arange(-radius, radius + 1)
    step = max(16, round(height / 32))
    centers = np.arange(0, height, step)
    shifted = np.array([
        np.interp(affine_y + shift, rows, b, left=0, right=0) for shift in shifts
    ], dtype=np.float32)
    costs = []
    for center in centers:
        start, end = max(0, center - step), min(height, center + step)
        aa, bb = a[start:end], shifted[:, start:end]
        values = 2 * np.sum(aa * bb, axis=1) / (
            np.sum(aa * aa) + np.sum(bb * bb, axis=1) + 1e-9
        )
        costs.append(values - 0.002 * shifts ** 2)
    distance = np.abs(shifts[:, None] - shifts[None, :])
    transitions = np.where(distance <= 2, 0.025 * distance, 1e6)
    state, previous = costs[0], []
    for cost in costs[1:]:
        choices = state[:, None] - transitions
        predecessor = choices.argmax(0)
        previous.append(predecessor)
        state = choices[predecessor, np.arange(len(shifts))] + cost
    index = int(state.argmax())
    chosen = [index]
    for predecessor in reversed(previous):
        index = int(predecessor[index])
        chosen.append(index)
    displacement = np.interp(rows, centers, shifts[np.array(chosen[::-1])])
    displacement = cv2.GaussianBlur(
        displacement.astype(np.float32)[:, None], (1, 0), sigmaX=0, sigmaY=step / 3
    ).ravel()
    return affine_y + displacement


def _correlation(a, b):
    return float(2 * np.sum(a * b) / max(float(np.sum(a * a) + np.sum(b * b)), 1e-9))


@dataclass
class PairMatches:
    scores: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray
    warps: np.ndarray  # warps[A,B,y_A] is the corresponding row in B.


def match_profiles(
    profiles: list[np.ndarray], max_shift: int | None = None, scale_range: float = 0.04
) -> PairMatches:
    """Compare every ordered pair, using evidence over its entire height."""
    if not profiles:
        raise ValueError("No strip profiles to compare.")
    height = profiles[0].shape[1]
    if height < 2 or any(p.shape != (3, height) for p in profiles):
        raise ValueError("Profiles must have three traces at the same height.")
    if not 0 <= scale_range < 1 or (max_shift is not None and max_shift < 0):
        raise ValueError("Scale range must be in [0, 1) and max_shift must be nonnegative.")
    max_shift = max(1, round(height * 0.06)) if max_shift is None else max_shift
    count = len(profiles)
    scores = np.full((count, count), -np.inf)
    scales, offsets = np.ones((count, count)), np.zeros((count, count), dtype=int)
    rows = np.arange(height, dtype=np.float32)
    warps = np.broadcast_to(rows, (count, count, height)).copy()
    for i, left in enumerate(profiles):
        for j, right in enumerate(profiles):
            if i == j:
                continue
            coarse, scale, offset = _affine_match(left[[1, 2]], right[[0, 2]], max_shift, scale_range)
            if coarse == 0:
                scores[i, j] = 0
                continue
            warp = _local_alignment(
                0.8 * left[1] + 0.2 * left[2], 0.8 * right[0] + 0.2 * right[2], scale, offset
            )
            edge = _correlation(left[1], np.interp(warp, rows, right[0], left=0, right=0))
            lines = _correlation(left[2], np.interp(warp, rows, right[2], left=0, right=0))
            # Retain the affine evidence: flexible warping alone can overfit
            # similar-looking letters from two strips that are not neighbors.
            scores[i, j] = 0.6 * coarse + 0.4 * (0.8 * edge + 0.2 * lines)
            scales[i, j], offsets[i, j], warps[i, j] = scale, offset, warp
    return PairMatches(scores, scales, offsets, warps)


def score_pairs(profiles, max_shift=None, scale_range=0.04):
    """Compatibility wrapper for callers needing just the affine matrices."""
    matches = match_profiles(profiles, max_shift, scale_range)
    return matches.scores, matches.scales, matches.offsets
