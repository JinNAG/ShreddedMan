"""Estimate gently curved paper edges without following individual letters."""

import cv2
import numpy as np


def estimate_strip_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth left/right coordinates for each row of a vertical strip.

    A median filter rejects short notches caused by dark ink at the boundary.
    A Gaussian filter then removes pixel-sized steps before image resampling.
    The filter scale follows paper width so it works at different resolutions.
    """
    visible = mask > 0
    occupied = visible.any(axis=1)
    if not occupied.any():
        raise ValueError("The strip contains no visible paper.")

    height, width = visible.shape
    left = visible.argmax(axis=1)
    right = width - 1 - visible[:, ::-1].argmax(axis=1)
    widths = right - left + 1
    typical_width = float(np.median(widths[occupied]))

    # Slanted tips and deep text notches should not determine the paper width.
    reliable = occupied & (widths >= 0.6 * typical_width)
    rows = np.arange(height)
    window = max(3, int(round(typical_width)) | 1)
    window = min(window, height if height % 2 else height - 1)

    edges = []
    for edge in (left, right):
        edge = np.interp(rows, rows[reliable], edge[reliable]).astype(np.float32)
        if window > 1:
            neighborhoods = np.lib.stride_tricks.sliding_window_view(
                np.pad(edge, window // 2, mode="edge"), window
            )
            edge = np.median(neighborhoods, axis=1)
            edge = cv2.GaussianBlur(
                edge[:, None],
                (1, window),
                sigmaX=0,
                sigmaY=window / 6,
                borderType=cv2.BORDER_REPLICATE,
            ).ravel()
        edges.append(np.clip(edge, 0, width - 1))

    return edges[0], edges[1]


def repair_strip_mask(mask: np.ndarray) -> np.ndarray:
    """Restore mask coverage over ink between the estimated paper edges.

    Use this while original photo pixels are still available. It changes only
    alpha coverage, never the photographed text. Preserve the original tips
    and fully missing rows, where extending the mask could include background.
    """
    left, right = estimate_strip_edges(mask)
    columns = np.arange(mask.shape[1])[None, :]
    interior = (columns >= np.floor(left[:, None])) & (
        columns <= np.ceil(right[:, None])
    )
    occupied_rows = np.flatnonzero((mask > 0).any(axis=1))
    margin = max(1, int(round(np.median(right - left + 1) / 2)))
    interior[: occupied_rows[0] + margin] = False
    interior[max(0, occupied_rows[-1] - margin + 1) :] = False
    interior &= (mask > 0).any(axis=1)[:, None]

    repaired = mask.copy()
    repaired[interior] = 255
    return repaired
