import numpy as np


def velStd(
    X: np.ndarray,
    V: np.ndarray,
    pbest: np.ndarray,
    gbest: np.ndarray,
    w: float,
    c1: float,
    c2: float,
) -> np.ndarray:
    """标准粒子群速度更新。

    .. math::
        V_{new} = w \\cdot V + c_1 \\cdot r_1 \\cdot (pbest - X)
                + c_2 \\cdot r_2 \\cdot (gbest - X)

    Parameters
    ----------
    X:
        当前粒子位置，shape ``(pop_size, dim)``。
    V:
        当前粒子速度，shape ``(pop_size, dim)``。
    pbest:
        个体历史最优位置，shape ``(pop_size, dim)``。
    gbest:
        全局最优位置，shape ``(1, dim)``。
    w:
        惯性权重。
    c1:
        个体学习因子。
    c2:
        社会学习因子。

    Returns
    -------
    np.ndarray
        更新后的速度，shape ``(pop_size, dim)``。
    """
    popsize, dim = X.shape
    r1 = np.random.rand(popsize, dim)
    r2 = np.random.rand(popsize, dim)
    gbest_matrix = np.broadcast_to(gbest, (popsize, dim))
    return w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest_matrix - X)
