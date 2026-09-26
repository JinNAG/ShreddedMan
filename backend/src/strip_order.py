"""Optimize the complete strip order with one predecessor and successor each."""

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


def optimize_orders(
    scores: np.ndarray, count: int = 5, time_limit: float = 90
) -> list[tuple[float, tuple[int, ...]]]:
    """Return globally optimal paths for the supplied compatibility scores.

    A dummy node turns an open path into a cycle. Subtour constraints exclude
    disconnected groups, and exclusion constraints yield distinct alternatives.
    A solver timeout raises an error rather than claiming an optimal result.
    """
    size = len(scores)
    if size == 0:
        raise ValueError("No strips to sort.")
    if size == 1:
        return [(0.0, (0,))]
    nodes = size + 1
    weights = np.zeros((nodes, nodes))
    weights[:size, :size] = np.where(np.isfinite(scores), scores, 0)
    upper = np.ones((nodes, nodes))
    upper[:size, :size] = np.isfinite(scores)
    np.fill_diagonal(upper, 0)
    rows, lower_bounds, upper_bounds = [], [], []
    for i in range(nodes):
        for incoming in (False, True):
            row = np.zeros((nodes, nodes))
            if incoming:
                row[:, i] = 1
            else:
                row[i, :] = 1
            rows.append(row.ravel())
            lower_bounds.append(1)
            upper_bounds.append(1)

    deadline = time.monotonic() + time_limit
    ranked = []
    cut_sets = set()
    while len(ranked) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Ordering solver timed out before proving the requested results.")
        result = milp(
            -weights.ravel(), integrality=np.ones(nodes * nodes),
            bounds=Bounds(0, upper.ravel()),
            constraints=LinearConstraint(csr_matrix(rows), lower_bounds, upper_bounds),
            options={"time_limit": remaining, "mip_rel_gap": 0},
        )
        if result.status == 2 and ranked:
            break  # Fewer distinct orders exist than requested.
        if not result.success:
            raise RuntimeError(f"Ordering solver did not prove an optimum: {result.message}")
        successor = np.argmax(result.x.reshape(nodes, nodes), axis=1)
        unseen = set(range(nodes))
        cycles = []
        while unseen:
            cycle = []
            node = min(unseen)
            while node in unseen:
                unseen.remove(node)
                cycle.append(node)
                node = int(successor[node])
            cycles.append(cycle)

        if len(cycles) > 1:
            for cycle in cycles:
                key = frozenset(cycle)
                if key in cut_sets:
                    continue
                cut_sets.add(key)
                row = np.zeros((nodes, nodes))
                row[np.ix_(cycle, cycle)] = 1
                rows.append(row.ravel())
                lower_bounds.append(-np.inf)
                upper_bounds.append(len(cycle) - 1)
            continue

        order = []
        node = int(successor[size])
        while node != size:
            order.append(node)
            node = int(successor[node])
        value = sum(float(scores[a, b]) for a, b in zip(order, order[1:]))
        ranked.append((value, tuple(order)))
        row = np.zeros((nodes, nodes))
        row[np.arange(nodes), successor] = 1
        rows.append(row.ravel())
        lower_bounds.append(-np.inf)
        upper_bounds.append(nodes - 1)
    return ranked
