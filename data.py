import random
import numpy as np


IMAGE_SIZE = (1920, 1080)


def generate_data(n_points: int = 10) -> np.array:
    return np.vstack(
        [
            np.random.uniform(0, IMAGE_SIZE[0], size=(n_points,)),
            np.random.uniform(0, IMAGE_SIZE[1], size=(n_points,)),
        ]
    ).transpose()


def generate_query_point() -> np.array:
    return np.array(
        [random.uniform(0, IMAGE_SIZE[0]), random.uniform(0, IMAGE_SIZE[1])]
    )


if __name__ == "__main__":
    from rich import print

    print(generate_data())
    print(generate_query_point())
