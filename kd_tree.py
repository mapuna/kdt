import numpy as np
from line_profiler import profile

from util import l2_distance

NDIM = 2


@profile
def build_kd_tree(points: np.array, depth: int = 0) -> dict:
    n = points.shape[0]

    assert points.shape[1] == NDIM

    if n <= 0:
        return None

    col = depth % NDIM
    sorted_points = points[np.argsort(points[:, col])]
    pivot = n // 2

    return {
        "point": sorted_points[pivot, :],
        "left": build_kd_tree(sorted_points[:pivot, :], depth + 1),
        "right": build_kd_tree(sorted_points[pivot + 1 :, :], depth + 1),
    }


@profile
def kd_tree_search(root, query: np.array, depth: int = 0, best: np.array = None):
    assert query.shape[0] == NDIM

    if root is None:
        return best

    axis = depth % NDIM

    next_best = None
    next_branch = None

    if (
        best is None
        or l2_distance(query, best)[0] > l2_distance(query, root["point"])[0]
    ):
        next_best = root["point"]
    else:
        next_best = best

    if query[axis] < root["point"][axis]:
        next_branch = root["left"]
    else:
        next_branch = root["right"]

    return kd_tree_search(next_branch, query, depth + 1, next_best)
