"""
Cost-matrix utilities and Hungarian assignment for DeepSORT matching.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


INF_COST = 1e5


def cosine_distance(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    """
    Cosine distance = 1 - cosine similarity, assuming L2-normalized inputs.

    Args:
        features1: (N, D)
        features2: (M, D)
    Returns:
        (N, M) distance matrix in [0, 2].
    """
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        return np.zeros((features1.shape[0], features2.shape[0]), dtype=np.float32)
    similarity = features1 @ features2.T
    return 1.0 - similarity


def iou_distance(track_bboxes: list, detection_bboxes: list) -> np.ndarray:
    """
    1 - IoU between all (track, detection) pairs.

    Returns (N, M) matrix in [0, 1].
    """
    N = len(track_bboxes)
    M = len(detection_bboxes)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    tb = np.asarray(track_bboxes, dtype=np.float32)
    db = np.asarray(detection_bboxes, dtype=np.float32)

    area_t = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
    area_d = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])

    x1 = np.maximum(tb[:, None, 0], db[None, :, 0])
    y1 = np.maximum(tb[:, None, 1], db[None, :, 1])
    x2 = np.minimum(tb[:, None, 2], db[None, :, 2])
    y2 = np.minimum(tb[:, None, 3], db[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_t[:, None] + area_d[None, :] - inter
    iou = inter / np.maximum(union, 1e-6)
    return 1.0 - iou


def gate_cost_matrix(cost_matrix: np.ndarray, max_distance: float) -> np.ndarray:
    """Set any entry above ``max_distance`` to INF_COST so Hungarian won't pick it."""
    gated = cost_matrix.copy()
    gated[gated > max_distance] = INF_COST
    return gated


def hungarian_match(cost_matrix: np.ndarray, max_cost: float = INF_COST - 1) -> tuple:
    """
    Solve assignment and drop matches whose cost is above ``max_cost``.

    Returns:
        matches:              list of (row_idx, col_idx)
        unmatched_rows:       list of row indices not matched
        unmatched_cols:       list of col indices not matched
    """
    N, M = cost_matrix.shape
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    # scipy's solver minimizes; we want the cheapest (best) match per row.
    # Passing -cost_matrix turns it into a maximization of "closeness".
    row_idx, col_idx = linear_sum_assignment(-cost_matrix)

    matches = []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] > max_cost:
            continue
        matches.append((int(r), int(c)))
        matched_rows.add(int(r))
        matched_cols.add(int(c))

    unmatched_rows = [r for r in range(N) if r not in matched_rows]
    unmatched_cols = [c for c in range(M) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
