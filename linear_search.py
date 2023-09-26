import numpy as np

from data import generate_data, generate_query_point
from util import l2_distance
from line_profiler import profile


@profile
def closest_point(all_p: np.array, new_p: np.array) -> tuple:
    assert all_p.shape[1] == new_p.shape[0]
    distances = l2_distance(all_p, new_p)
    min_dist_index = np.argmin(distances)
    return (
        min_dist_index,
        all_p[min_dist_index, :],
        distances[min_dist_index],
    )


if __name__ == "__main__":
    data = generate_data()
    query = generate_query_point()

    print(query)
    print(closest_point(data, query))
