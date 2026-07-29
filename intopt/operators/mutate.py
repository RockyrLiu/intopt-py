import numpy as np

from intopt.operators.utils import chaos_sequence


def mutSwap(solution: np.ndarray) -> np.ndarray:
    """交换变异：随机交换两个位置。用于排列型解（如 TSP 路径）。

    Parameters
    ----------
    solution : np.ndarray
        待变异的排列解。

    Returns
    -------
    np.ndarray
        交换两个位置后的新解。
    """
    n = len(solution)
    i, j = np.random.choice(n, size=2, replace=False)
    new_sol = solution.copy()
    new_sol[i], new_sol[j] = solution[j], solution[i]
    return new_sol


def mutFlip(solution: np.ndarray) -> np.ndarray:
    """翻转变异：随机翻转一个 0/1 位。用于二值型解（如 0-1 背包）。

    Parameters
    ----------
    solution : np.ndarray
        待变异的二值解。

    Returns
    -------
    np.ndarray
        翻转一个随机位后的新解。
    """
    n = len(solution)
    idx = np.random.randint(n)
    new_sol = solution.copy()
    new_sol[idx] = 1.0 - new_sol[idx]
    return new_sol


def mutGaussian(
    solution: np.ndarray,
    mu: float = 0.0,
    sigma: float = 0.5,
    indpb: float = 0.5,
) -> np.ndarray:
    """高斯变异：每个维度以概率 ``indpb`` 变异，扰动值服从 N(mu, sigma²)。用于连续型解。

    Parameters
    ----------
    solution : np.ndarray
        待变异的连续解。
    mu : float
        扰动均值，默认 0.0。
    sigma : float
        扰动标准差，默认 0.5。
    indpb : float
        每个维度独立变异的概率，默认 0.5。

    Returns
    -------
    np.ndarray
        高斯变异后的新解。
    """
    n = len(solution)
    mask = np.random.random(n) < indpb
    new_sol = solution.copy()
    new_sol[mask] += mu + sigma * np.random.randn(np.sum(mask))
    return new_sol


def mutChaosSwap(solution: np.ndarray) -> np.ndarray:
    """混沌交换变异：交换位置由 Logistic 序列确定。用于排列型解。

    Parameters
    ----------
    solution : np.ndarray
        待变异的排列解。

    Returns
    -------
    np.ndarray
        混沌交换两个位置后的新解。
    """
    n = len(solution)
    ch = chaos_sequence(2)
    i, j = np.clip(np.floor(ch * n).astype(int), 0, n - 1)
    new_sol = solution.copy()
    new_sol[i], new_sol[j] = solution[j], solution[i]
    return new_sol


# ---------------------------------------------------------------------------
# DE 差分变异算子（单个个体）
# ---------------------------------------------------------------------------


def mutDERand1(
    solution: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    F: float = 0.5,
) -> np.ndarray:
    """DE/rand/1 差分变异。

    从种群中随机选 3 个互异个体，
    生成供体向量: v = x_r1 + F * (x_r2 - x_r3)。

    Parameters
    ----------
    solution : np.ndarray
        目标向量（用于排除自身）。
    population : np.ndarray
        种群，shape ``(pop_size, dim)``。
    fitness : np.ndarray
        适应度数组（未使用，保留以统一接口）。
    F : float
        缩放因子，默认 0.5。

    Returns
    -------
    np.ndarray
        供体向量。
    """
    popsize = len(population)
    for idx in range(popsize):
        if np.array_equal(population[idx], solution):
            break
    candidates = [i for i in range(popsize) if i != idx]
    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
    return population[r1] + F * (population[r2] - population[r3])


def mutDEBest1(
    solution: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    F: float = 0.5,
) -> np.ndarray:
    """DE/best/1 差分变异。

    供体向量: v = x_best + F * (x_r1 - x_r2)，
    其中 x_best 为当前种群最优个体。

    Parameters
    ----------
    solution : np.ndarray
        目标向量（用于排除自身）。
    population : np.ndarray
        种群，shape ``(pop_size, dim)``。
    fitness : np.ndarray
        适应度数组，用于确定最优个体。
    F : float
        缩放因子，默认 0.5。

    Returns
    -------
    np.ndarray
        供体向量。
    """
    popsize = len(population)
    for idx in range(popsize):
        if np.array_equal(population[idx], solution):
            break
    best_idx = np.argmin(fitness)
    current_best = population[best_idx]
    candidates = [i for i in range(popsize) if i != idx]
    r1, r2 = np.random.choice(candidates, 2, replace=False)
    return current_best + F * (population[r1] - population[r2])


def mutDERand2(
    solution: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    F: float = 0.5,
) -> np.ndarray:
    """DE/rand/2 差分变异。

    供体向量:
    v = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)。

    Parameters
    ----------
    solution : np.ndarray
        目标向量（用于排除自身）。
    population : np.ndarray
        种群，shape ``(pop_size, dim)``。
    fitness : np.ndarray
        适应度数组（未使用）。
    F : float
        缩放因子，默认 0.5。

    Returns
    -------
    np.ndarray
        供体向量。
    """
    popsize = len(population)
    for idx in range(popsize):
        if np.array_equal(population[idx], solution):
            break
    candidates = [i for i in range(popsize) if i != idx]
    r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
    return (
        population[r1]
        + F * (population[r2] - population[r3])
        + F * (population[r4] - population[r5])
    )


def mutDEBest2(
    solution: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    F: float = 0.5,
) -> np.ndarray:
    """DE/best/2 差分变异。

    供体向量:
    v = x_best + F*(x_r1 - x_r2) + F*(x_r3 - x_r4)。

    Parameters
    ----------
    solution : np.ndarray
        目标向量（用于排除自身）。
    population : np.ndarray
        种群，shape ``(pop_size, dim)``。
    fitness : np.ndarray
        适应度数组，用于确定最优个体。
    F : float
        缩放因子，默认 0.5。

    Returns
    -------
    np.ndarray
        供体向量。
    """
    popsize = len(population)
    for idx in range(popsize):
        if np.array_equal(population[idx], solution):
            break
    best_idx = np.argmin(fitness)
    current_best = population[best_idx]
    candidates = [i for i in range(popsize) if i != idx]
    r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
    return (
        current_best
        + F * (population[r1] - population[r2])
        + F * (population[r3] - population[r4])
    )


def mutDECurrentToRand1(
    solution: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    F: float = 0.5,
) -> np.ndarray:
    """DE/current-to-rand/1 差分变异。

    供体向量:
    v = x_i + K*(x_r1 - x_i) + F*(x_r2 - x_r3)，其中 K=0.5。

    Parameters
    ----------
    solution : np.ndarray
        目标向量。
    population : np.ndarray
        种群，shape ``(pop_size, dim)``。
    fitness : np.ndarray
        适应度数组（未使用）。
    F : float
        缩放因子，默认 0.5。

    Returns
    -------
    np.ndarray
        供体向量。
    """
    popsize = len(population)
    K = 0.5
    for idx in range(popsize):
        if np.array_equal(population[idx], solution):
            break
    candidates = [i for i in range(popsize) if i != idx]
    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
    return (
        solution
        + K * (population[r1] - solution)
        + F * (population[r2] - population[r3])
    )


def mutDECurrentToBest1(
    solution: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    F: float = 0.5,
) -> np.ndarray:
    """DE/current-to-best/1 差分变异。

    供体向量:
    v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)。

    Parameters
    ----------
    solution : np.ndarray
        目标向量。
    population : np.ndarray
        种群，shape ``(pop_size, dim)``。
    fitness : np.ndarray
        适应度数组，用于确定最优个体。
    F : float
        缩放因子，默认 0.5。

    Returns
    -------
    np.ndarray
        供体向量。
    """
    popsize = len(population)
    for idx in range(popsize):
        if np.array_equal(population[idx], solution):
            break
    best_idx = np.argmin(fitness)
    current_best = population[best_idx]
    candidates = [i for i in range(popsize) if i != idx]
    r1, r2 = np.random.choice(candidates, 2, replace=False)
    return (
        solution
        + F * (current_best - solution)
        + F * (population[r1] - population[r2])
    )
