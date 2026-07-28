import numpy as np

from intopt.operators.utils import chaos_sequence


def initRandom(problem, pop_size: int) -> np.ndarray:
    """随机初始化：调用 ``problem.random_solution()`` 生成种群。

    Parameters
    ----------
    problem : Problem
        优化问题实例。
    pop_size : int
        种群大小。

    Returns
    -------
    np.ndarray
        种群，shape ``(pop_size, *ind_shape)``。
    """
    return np.array([problem.random_solution() for _ in range(pop_size)])


def initChaosContinuous(problem, pop_size: int) -> np.ndarray:
    """混沌初始化（连续型）：Logistic 映射生成种群，映射到边界内。

    Parameters
    ----------
    problem : ContinuousProblem
        连续优化问题实例（需有 ``dim``、``lower``、``upper`` 属性）。
    pop_size : int
        种群大小。

    Returns
    -------
    np.ndarray
        种群，shape ``(pop_size, dim)``。
    """
    dim = problem.dim
    ch = chaos_sequence(pop_size * dim)
    ch = ch.reshape(pop_size, dim)
    return problem.lower + ch * (problem.upper - problem.lower)


def initChaosPermutation(problem, pop_size: int) -> np.ndarray:
    """混沌初始化（排列型）：Logistic 映射 → argsort 生成排列种群。

    Parameters
    ----------
    problem : TSPProblem
        TSP 问题实例（需有 ``coordinates`` 属性）。
    pop_size : int
        种群大小。

    Returns
    -------
    np.ndarray
        排列种群，每个元素为 ``range(n)`` 的排列。
    """
    n = len(problem.coordinates)
    pop = []
    for _ in range(pop_size):
        ch = chaos_sequence(n)
        pop.append(np.argsort(ch))
    return np.array(pop)


def initCustom(population: np.ndarray) -> np.ndarray:
    """自定义初始化：直接包装用户提供的种群。

    Parameters
    ----------
    population : np.ndarray
        用户提供的种群。

    Returns
    -------
    np.ndarray
        种群副本。
    """
    return np.array(population, copy=True)
