import numpy as np


def selTournament(
    population: np.ndarray, fitness: np.ndarray, k: int, tournsize: int = 3
) -> np.ndarray:
    """锦标赛选择：每次随机选 ``tournsize`` 个个体，保留最优。

    Parameters
    ----------
    population : np.ndarray
        种群，shape ``(n, *ind_shape)``。
    fitness : np.ndarray
        适应度数组，值越小越好，shape ``(n,)``。
    k : int
        需选择的个体数量。
    tournsize : int
        锦标赛规模，默认 3。

    Returns
    -------
    np.ndarray
        选出的 k 个个体，shape ``(k, *ind_shape)``。
    """
    n = len(population)
    indices = np.empty(k, dtype=int)
    for i in range(k):
        candidates = np.random.choice(n, tournsize, replace=False)
        indices[i] = candidates[np.argmin(fitness[candidates])]
    return population[indices]


def selRoulette(
    population: np.ndarray, fitness: np.ndarray, k: int
) -> np.ndarray:
    """轮盘赌选择：适应度越小被选中概率越大。

    Parameters
    ----------
    population : np.ndarray
        种群，shape ``(n, *ind_shape)``。
    fitness : np.ndarray
        适应度数组，值越小越好，shape ``(n,)``。
    k : int
        需选择的个体数量。

    Returns
    -------
    np.ndarray
        选出的 k 个个体，shape ``(k, *ind_shape)``。
    """
    eps = 1e-10
    weights = np.max(fitness) - fitness + eps
    if np.sum(weights) <= eps:
        weights = np.ones(len(fitness))
    probs = weights / np.sum(weights)
    indices = np.random.choice(len(population), k, p=probs, replace=True)
    return population[indices]


def selBest(
    population: np.ndarray, fitness: np.ndarray, k: int
) -> np.ndarray:
    """精英选择：返回适应度最优的 k 个个体。

    Parameters
    ----------
    population : np.ndarray
        种群，shape ``(n, *ind_shape)``。
    fitness : np.ndarray
        适应度数组，值越小越好，shape ``(n,)``。
    k : int
        需选择的个体数量。

    Returns
    -------
    np.ndarray
        适应度最优的 k 个个体，shape ``(k, *ind_shape)``。
    """
    indices = np.argsort(fitness)[:k]
    return population[indices]
