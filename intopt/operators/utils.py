import numpy as np


def chaos_sequence(length: int) -> np.ndarray:
    """生成 Logistic 映射混沌序列，值域 (0, 1)。

    Parameters
    ----------
    length : int
        序列长度。

    Returns
    -------
    np.ndarray
        混沌序列，shape ``(length,)``，值域 ``(0, 1)``。
    """
    seq = np.zeros(length)
    seq[0] = np.random.rand()
    for i in range(1, length):
        seq[i] = 4 * seq[i - 1] * (1 - seq[i - 1])
    return seq


def fitnessSharing(
    population: np.ndarray,
    fitness: np.ndarray,
    distance_threshold: float,
    sharing_extent: float,
) -> np.ndarray:
    """适应度共享：距离近的个体互相抑制适应度，维护种群多样性。

    适用于**最小化**问题。对每个个体，计算与其他个体的欧氏距离。
    距离小于 ``distance_threshold`` 的邻居对该个体的共享和贡献：

        ``1 - distance / (sharing_extent * distance_threshold)``

    先将适应度转为最大化风格，除以共享和，再转回最小化。
    等价于使拥挤个体适应度变差。

    Parameters
    ----------
    population : np.ndarray
        种群，shape ``(n, d)``。
    fitness : np.ndarray
        原始适应度数组，shape ``(n,)``。
    distance_threshold : float
        共享距离阈值，距离小于此值的个体纳入共享计算。
    sharing_extent : float
        共享程度，控制共享半径 = extent * threshold。

    Returns
    -------
    np.ndarray
        调整后的适应度数组，shape ``(n,)``。
    """
    fitness = np.asarray(fitness, dtype=float)
    n = len(population)
    if n <= 1:
        return fitness.copy()

    diff = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff**2).sum(axis=-1))

    radius = sharing_extent * distance_threshold
    sharing = np.where(
        dist_matrix < distance_threshold,
        1.0 - dist_matrix / radius,
        0.0,
    )
    np.fill_diagonal(sharing, 0.0)
    sharing_sum = 1.0 + sharing.sum(axis=1)

    max_fit = np.max(fitness)
    return max_fit - (max_fit - fitness) / sharing_sum
