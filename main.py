from all_pair_search import all_pair_search
from kd_tree import kdtree_search, build_kdtree

from data import generate_2d_data, generate_2d_query_point

from rich import print

import time


if __name__ == "__main__":
    points = generate_2d_data(n_points=100000)
    query = generate_2d_query_point()
    kdt = build_kdtree(points)

    t1 = time.time_ns()
    all_pair_closest = all_pair_search(points, query)[1]
    t2 = time.time_ns()
    kdtree_closest = kdtree_search(kdt, query, 0)
    t3 = time.time_ns()

    print(f"Closest point (all_pair): {all_pair_closest}, time taken: {t2 - t1} ns")
    print(f"Closest point (kdtree): {kdtree_closest}, time taken: {t3 - t2} ns")
