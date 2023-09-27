import numpy as np
from line_profiler import profile

from util import l2_distance

NDIM = 2


@profile
def build_kdtree(points: np.array, depth: int = 0) -> dict:
    n = points.shape[0]

    assert points.shape[1] == NDIM

    if n <= 0:
        return None

    col = depth % NDIM
    sorted_points = points[np.argsort(points[:, col])]
    pivot = n // 2

    return {
        "point": sorted_points[pivot, :],
        "left": build_kdtree(sorted_points[:pivot, :], depth + 1),
        "right": build_kdtree(sorted_points[pivot + 1 :, :], depth + 1),
    }


def closer_distance(query: np.array, p1: np.array, p2: np.array) -> np.array:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    d1 = l2_distance(query, p1)[0]
    d2 = l2_distance(query, p2)[0]

    if d1 < d2:
        return p1

    return p2


@profile
def kdtree_search(root, query, depth=0):
    if root is None:
        return None

    col = depth % NDIM

    next_branch = None
    opposite_branch = None

    if query[col] < root["point"][col]:
        next_branch = root["left"]
        opposite_branch = root["right"]
    else:
        next_branch = root["right"]
        opposite_branch = root["left"]

    best = closer_distance(
        query, kdtree_search(next_branch, query, depth + 1), root["point"]
    )

    if l2_distance(query, best)[0] > abs(query[col] - root["point"][col]):
        best = closer_distance(
            query, kdtree_search(opposite_branch, query, depth + 1), best
        )

    return best
