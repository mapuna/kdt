from linear_search import closest_point as linear_closet_point
from kd_tree import kd_tree_search, build_kd_tree

from data import generate_data, generate_query_point

from rich import print


if __name__ == "__main__":
    points = generate_data(n_points=5)
    query = generate_query_point()
    kdt = build_kd_tree(points)

    print(f"Query:\n{query}\nData:\n{points}")
    print(
        f"Closest point (correct, brute-force): {linear_closet_point(points, query)[1]}"
    )
    print(f"Closest point (kd_tree): {kd_tree_search(kdt, query, 0)}")
