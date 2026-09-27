"""Full-height ink matching with affine and constrained local registration."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


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


def _affine_matches(profiles, max_shift, scale_range):
    """Reuse Fourier transforms across pairs instead of repeating registration.

    This computes the same full-height edge/body correlations at every allowed
    offset. Scale refinement is grouped by height, and temporary FFT arrays are
    bounded in size so larger submissions do not need an N*N*height workspace.
    """
    profiles = np.asarray(profiles)
    count, _, height = profiles.shape
    padding = max_shift + int(np.ceil(height * scale_range)) + 2
    fft_size = next_fast_len(height * 2 + 2 * padding)
    left, right = profiles[:, [1, 2]], profiles[:, [0, 2]]
    left_fft = rfft(np.pad(left, ((0, 0), (0, 0), (padding, padding))), n=fft_size, axis=-1)
    left_energy = np.sum(left * left, axis=-1)
    best = np.zeros((count, count))
    heights = np.full((count, count), height)
    offsets = np.zeros((count, count), int)
    positions = np.arange(padding - max_shift, padding + max_shift + 1)
    channel_weights = np.array([0.8, 0.2], dtype=np.float32)[None, :, None]

    def score_height(scaled_height, pairs):
        scaled = cv2.resize(right.reshape(count * 2, height).T, (count * 2, scaled_height))
        scaled = scaled.T.reshape(count, 2, scaled_height).copy()
        right_fft = rfft(scaled, n=fft_size, axis=-1)
        right_energy = np.sum(scaled * scaled, axis=-1)
        for start in range(0, len(pairs), 128):
            ii, jj = pairs[start:start + 128].T
            energy = left_energy[ii] + right_energy[jj]
            products = np.sum(
                left_fft[ii] * np.conj(right_fft[jj])
                * (2 * channel_weights / np.maximum(energy[:, :, None], 1e-9)), axis=1,
            )
            values = irfft(products, n=fft_size, axis=-1)[:, positions]
            values[:, positions > height + 2 * padding - scaled_height] = -np.inf
            indices = values.argmax(1)
            scores = values[np.arange(len(ii)), indices]
            improved = scores > best[ii, jj] + 1e-7
            a, b = ii[improved], jj[improved]
            best[a, b] = scores[improved]
            heights[a, b] = scaled_height
            offsets[a, b] = positions[indices[improved]] - padding

    pairs = np.column_stack(np.where(~np.eye(count, dtype=bool)))
    for scaled_height in sorted({max(2, round(height * scale))
                                 for scale in np.linspace(1 - scale_range, 1 + scale_range, 25)}):
        score_height(scaled_height, pairs)
    refinements = {}
    for i, j in pairs:
        scale = heights[i, j] / height
        for value in np.linspace(max(1 - scale_range, scale - scale_range / 10),
                                 min(1 + scale_range, scale + scale_range / 10), 17):
            refinements.setdefault(max(2, round(height * value)), set()).add((i, j))
    for scaled_height, candidates in sorted(refinements.items()):
        score_height(scaled_height, np.asarray(sorted(candidates)))
    return best, heights / height, offsets


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


def stabilize_row_maps(order, maps, warps, scores):
    """Correct accumulated row offsets using consistent nearby comparisons.

    A single neighbor match can lock onto another text line and shift every
    strip that follows it. Nearby non-neighbors provide independent checks.
    Only reciprocal matches with a connected, low-residual consensus are used.
    A weak strip can be disconnected without disabling the consensus among
    the other strips. The local row bends of each strip stay intact.
    """
    count = len(order)
    if count < 4:
        return maps
    height = len(maps[0])
    center = (height - 1) // 2
    edges = []
    for i, left in enumerate(order):
        for j in range(i + 1, min(i + 4, count)):
            right = order[j]
            forward = float(warps[left, right, center] - center)
            reverse = float(warps[right, left, center] - center)
            if abs(forward + reverse) > max(25, height * 0.025):
                continue
            weight = max(0.05, min(float(scores[left, right]), float(scores[right, left])))
            edges.append((i, j, (forward - reverse) / 2, weight))
    neighbors = [set() for _ in range(count)]
    for i, j, _, _ in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    adjusted = list(maps)
    visited = set()
    for start in range(count):
        if start in visited:
            continue
        component = set()
        pending = [start]
        while pending:
            vertex = pending.pop()
            if vertex in component:
                continue
            component.add(vertex)
            pending.extend(neighbors[vertex] - component)
        visited.update(component)
        # A pair or triple has no independent check on a bad row match.
        if len(component) < 4:
            continue
        positions = sorted(component)
        columns = {position: index for index, position in enumerate(positions[1:])}
        component_edges = [edge for edge in edges if edge[0] in component]
        matrix = np.zeros((len(component_edges), len(component) - 1))
        expected = np.empty(len(component_edges))
        weights = np.empty(len(component_edges))
        for row, (i, j, delta, weight) in enumerate(component_edges):
            if i in columns:
                matrix[row, columns[i]] = -1
            if j in columns:
                matrix[row, columns[j]] = 1
            expected[row], weights[row] = delta, weight
        fit = np.linalg.lstsq(matrix * weights[:, None], expected * weights, rcond=None)[0]
        for _ in range(8):
            residual = matrix @ fit - expected
            robust_weights = weights / np.sqrt(1 + (residual / 25) ** 2)
            fit = np.linalg.lstsq(matrix * robust_weights[:, None], expected * robust_weights, rcond=None)[0]
        if np.percentile(abs(matrix @ fit - expected), 90) > max(30, height * 0.02):
            continue
        anchor = positions[0]
        anchor_offset = maps[anchor][center] - center
        for position, relative_offset in zip(positions[1:], fit):
            target_offset = anchor_offset + relative_offset
            adjusted[position] = maps[position] + (
                target_offset - (maps[position][center] - center)
            )
    return adjusted


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
    coarse_scores, scales, offsets = _affine_matches(profiles, max_shift, scale_range)
    for i, left in enumerate(profiles):
        for j, right in enumerate(profiles):
            if i == j:
                continue
            coarse, scale, offset = coarse_scores[i, j], scales[i, j], offsets[i, j]
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
