import numpy as np


def selTournament(
    population: np.ndarray, fitness: np.ndarray, k: int, tournsize: int = 3
) -> np.ndarray:
    """锦标赛选择：每次随机选 tournsize 个个体，保留最优。"""
    n = len(population)
    indices = np.empty(k, dtype=int)
    for i in range(k):
        candidates = np.random.choice(n, tournsize, replace=False)
        indices[i] = candidates[np.argmin(fitness[candidates])]
    return population[indices]


def selRoulette(
    population: np.ndarray, fitness: np.ndarray, k: int
) -> np.ndarray:
    """轮盘赌选择：适应度越小被选中概率越大。"""
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
    """精英选择：返回适应度最优的 k 个个体。"""
    indices = np.argsort(fitness)[:k]
    return population[indices]
