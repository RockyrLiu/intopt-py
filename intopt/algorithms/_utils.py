import numpy as np


def clamp_to_bounds(
    population: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """将群体每个维度钳制到 ``[lower, upper]`` 区间内。

    Parameters
    ----------
    population:
        形状 ``(size, dim)`` 或 ``(dim,)``。
    lower, upper:
        形状 ``(dim,)``。
    """
    return np.clip(population, lower, upper)
