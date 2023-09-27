import numpy as np


def l2_distance(all_p: np.array, new_p: np.array) -> tuple["float | np.array", int]:
    assert len(new_p.shape) == 1

    axis = 0
    if len(all_p.shape) == 2:
        axis = 1
        assert all_p.shape[1] == new_p.shape[0]
    else:
        assert new_p.shape[0] == all_p.shape[0]

    return np.sqrt(np.sum((all_p - new_p) ** 2, axis=axis)), axis
