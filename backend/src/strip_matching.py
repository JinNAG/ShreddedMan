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


def _row_profile(image: np.ndarray, factor: int) -> np.ndarray:
    """Interior ink per row: the text lines a strip shares with the whole page.

    Cut shadows at the tips are not text; they would align crop ends instead.
    """
    ink = strip_ink(image)
    margin = max(1, round(ink.shape[1] * 0.15))
    rows = (ink[:, margin:-margin] > 0.2).mean(1).astype(np.float32)
    tip = max(1, round(len(rows) * 0.02))
    rows[:tip] = rows[-tip:] = 0
    length = max(2, round(len(rows) / factor))
    return cv2.resize(rows[:, None], (1, length), interpolation=cv2.INTER_AREA).ravel()


def _blur(profile: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(profile[:, None], (1, 0), sigmaX=0, sigmaY=sigma).ravel()


# Line-level blur, full scale/top search twice, then finer local refinement:
# (blur, +/- relative scale range, scale steps, +/- top range), page fractions.
LEVEL_SCHEDULE = ((0.003, 0.10, 41, 0.04), (0.003, 0.10, 41, 0.04),
                  (0.0015, 0.01, 17, 0.01), (0.0015, 0.004, 9, 0.004))
LEVEL_SCALE_LIMIT = 0.10
LEVEL_TOP_LIMIT = 0.04


def _register_profile(profile, consensus, pad, frame, prior, scale, top, scale_range, scale_count, top_range):
    """Best (scale, top) of one profile against the consensus of the others.

    Pearson correlation ignores the overall ink level, which otherwise makes
    every placement on a page of text look alike. A penalty keeps strips
    level and at the prior scale unless their text lines say otherwise.
    """
    sums = np.concatenate(([0], np.cumsum(consensus)))
    squares = np.concatenate(([0], np.cumsum(consensus * consensus)))
    reach = round(frame * LEVEL_TOP_LIMIT)
    rows = np.arange(len(profile))
    best = (-np.inf, scale, top)
    for candidate in scale * (1 + np.linspace(-scale_range, scale_range, scale_count)):
        if abs(candidate / prior - 1) > LEVEL_SCALE_LIMIT + 1e-9:
            continue
        length = int((len(profile) - 1) * candidate) + 1
        placed = np.interp(np.arange(length) / candidate, rows, profile)
        placed -= placed.mean()
        starts = pad + np.arange(max(-reach, round(top - top_range * frame)),
                                 min(reach, round(top + top_range * frame)) + 1)
        starts = starts[starts + length <= len(consensus)]
        if not len(starts):
            continue
        total = sums[starts + length] - sums[starts]
        spread = squares[starts + length] - squares[starts] - total * total / length
        windows = np.lib.stride_tricks.sliding_window_view(consensus, length)[starts]
        values = windows @ placed / np.sqrt(np.maximum(float(placed @ placed) * spread, 1e-12))
        values -= 0.05 * (((starts - pad) / (0.02 * frame)) ** 2 + ((candidate / prior - 1) / 0.1) ** 2)
        choice = int(values.argmax())
        if values[choice] > best[0]:
            best = (float(values[choice]), float(candidate), float(starts[choice] - pad))
    return best[1:]


def level_strips(images: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict]]:
    """Put every strip on one page grid, level with the others.

    All strips of a page share its text lines. Each strip's row profile is
    registered (scale and top position) against the mean of all the other
    strips, so a placement error cannot pass from one join to the next, and
    repeating text lines cannot drift a strip by whole lines. The search
    starts with every strip top on the page top and every crop stretched to
    the median height; weak evidence leaves a strip near that placement.
    Returns resampled BGRA copies with one height and each strip's placement;
    saved strip PNGs are unchanged.
    """
    if not images:
        raise ValueError("No strips to level.")
    median = float(np.median([image.shape[0] for image in images]))
    factor = max(1, round(median / 1200))
    profiles = [_row_profile(image, factor) for image in images]
    frame = median / factor
    prior = np.array([frame / len(profile) for profile in profiles])
    scales, tops = prior.copy(), np.zeros(len(images))
    # Margin strips cross too few text lines to place; a few marks or a tip
    # shadow would then decide. They keep the prior and stay out of the consensus.
    text_rows = np.array([np.mean(profile > 0.02) for profile in profiles])
    placeable = np.flatnonzero(text_rows >= 0.25 * np.median(text_rows))
    if len(placeable) > 1:
        pad = round(frame * LEVEL_TOP_LIMIT) + 2
        grid = np.arange(-pad, round(frame * (1.05 + LEVEL_SCALE_LIMIT)) + pad, dtype=np.float32)
        for sigma, scale_range, scale_count, top_range in LEVEL_SCHEDULE:
            blurred = {index: _blur(profiles[index], max(0.5, sigma * frame)) for index in placeable}
            placed = {
                index: np.interp((grid - tops[index]) / scales[index], np.arange(len(profile)), profile,
                                 left=0, right=0)
                for index, profile in blurred.items()
            }
            total = np.sum(list(placed.values()), axis=0)
            for index, (scale, top) in [
                (index, _register_profile(profile, (total - placed[index]) / (len(placeable) - 1), pad, frame,
                                          prior[index], scales[index], tops[index],
                                          scale_range, scale_count, top_range))
                for index, profile in blurred.items()
            ]:
                scales[index], tops[index] = scale, top
    # Analysis rows average `factor` page rows and len(image)/len(profile) strip rows.
    scales = scales * factor * np.array([len(p) / image.shape[0] for p, image in zip(profiles, images)])
    tops = tops * factor
    tops -= np.floor(tops.min())
    height = int(np.ceil(max(t + s * image.shape[0] for t, s, image in zip(tops, scales, images))))
    leveled, placements = [], []
    for index, (image, scale, offset) in enumerate(zip(images, scales, tops)):
        width = max(1, round(image.shape[1] * scale))
        # Premultiply so hidden background colors cannot bleed into paper edges.
        source = image.astype(np.float32)
        source[:, :, :3] *= source[:, :, 3:4] / 255
        matrix = np.float32([[scale, 0, 0], [0, scale, offset]])
        result = cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = result[:, :, 3:4] / 255
        colors = np.full_like(result[:, :, :3], 255)
        np.divide(result[:, :, :3], alpha, out=colors, where=alpha > 0)
        result[:, :, :3] = colors
        leveled.append(np.clip(np.rint(result), 0, 255).astype(np.uint8))
        placements.append({"scale": float(scale), "top": float(offset),
                           "placed_by_text": bool(index in placeable and len(placeable) > 1)})
    return leveled, placements


def seam_row_shifts(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    """Row shift that lines up text beside one cut between two leveled strips.

    Text lines continue across a cut, so the ink rows just inside both edges
    agree when the strips are aligned. Cut shadows and torn fibres at the edge
    itself are not text and are skipped. Short windows down the cut follow
    gentle paper bends; windows without text on both sides do not vote.
    Returns, for every row y of the left strip, the matching right-strip row
    minus y, and the fraction of windows that found text.
    """
    traces = []
    for ink, columns in ((strip_ink(left), slice(-(left.shape[1] // 2), -max(1, round(left.shape[1] * 0.1)))),
                         (strip_ink(right), slice(max(1, round(right.shape[1] * 0.1)), right.shape[1] // 2))):
        traces.append(_blur(ink[:, columns].mean(1).astype(np.float32), 1.5))
    a, b = traces
    height = len(a)
    window, radius = max(16, round(height * 0.08)), max(2, round(height * 0.006))
    starts = range(radius, height - window - radius + 1, window // 2)
    centers, shifts = [], []
    for start in starts:
        aa = a[start:start + window] - a[start:start + window].mean()
        candidates = np.lib.stride_tricks.sliding_window_view(b[start - radius:start + window + radius], window)
        candidates = candidates - candidates.mean(1, keepdims=True)
        values = candidates @ aa / np.sqrt(np.maximum((candidates ** 2).sum(1) * float(aa @ aa), 1e-12))
        values -= 0.1 * (np.arange(-radius, radius + 1) / radius) ** 2  # Prefer the level placement.
        best = int(values.argmax())
        if values[best] >= 0.5:
            centers.append(start + window / 2)
            shifts.append(best - radius)
    if not centers:
        return np.zeros(height, np.float32), 0.0
    step = np.interp(np.arange(height), centers, shifts).astype(np.float32)
    return _blur(step, window / 4), len(centers) / len(starts)


def _affine_matches(profiles, max_shift, scale_range, level_prior=0.0):
    """Reuse Fourier transforms across pairs instead of repeating registration.

    This computes the same full-height edge/body correlations at every allowed
    offset. Scale refinement is grouped by height, and temporary FFT arrays are
    bounded in size so larger submissions do not need an N*N*height workspace.
    level_prior is the score cost of a 1% shift or 1% scale change (growing
    quadratically) for strips that are already level.
    """
    profiles = np.asarray(profiles)
    count, _, height = profiles.shape
    padding = max_shift + int(np.ceil(height * scale_range)) + 2
    fft_size = next_fast_len(height * 2 + 2 * padding)
    left, right = profiles[:, [1, 2]], profiles[:, [0, 2]]
    left_fft = rfft(np.pad(left, ((0, 0), (0, 0), (padding, padding))), n=fft_size, axis=-1)
    left_energy = np.sum(left * left, axis=-1)
    best = np.full((count, count), -np.inf if level_prior else 0.0)
    heights = np.full((count, count), height)
    offsets = np.zeros((count, count), int)
    positions = np.arange(padding - max_shift, padding + max_shift + 1)
    channel_weights = np.array([0.8, 0.2], dtype=np.float32)[None, :, None]
    shift_cost = level_prior * ((positions - padding) / (0.01 * height)) ** 2

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
            values -= shift_cost + level_prior * ((scaled_height / height - 1) / 0.01) ** 2
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


def match_profiles(
    profiles: list[np.ndarray], max_shift: int | None = None, scale_range: float = 0.04,
    level_prior: float = 0.0,
) -> PairMatches:
    """Compare every ordered pair, using evidence over its entire height.

    For leveled strips, level_prior makes shifts and scale changes cost score,
    so repeating text lines cannot pull a join onto a neighboring line.
    """
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
    coarse_scores, scales, offsets = _affine_matches(profiles, max_shift, scale_range, level_prior)
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
