import numpy as np
from data import generate_data
from rich import print
from line_profiler import profile

NDIM = 2


@profile
def build_kd_tree(points: np.array, depth: int = 0) -> dict:
    n = points.shape[0]

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


if __name__ == "__main__":
    points = generate_data(n_points=31)
    print(build_kd_tree(points, 0))
