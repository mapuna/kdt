import numpy as np


def l2_distance(all_p: np.array, new_p: np.array) -> np.array:
    return np.sqrt(np.sum((all_p - new_p) ** 2, axis=1))
