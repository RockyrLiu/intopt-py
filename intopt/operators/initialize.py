import numpy as np


def initRandom(problem, pop_size: int) -> np.ndarray:
    """随机初始化：调用 ``problem.random_solution()`` 生成种群。"""
    return np.array([problem.random_solution() for _ in range(pop_size)])


def initChaosContinuous(problem, pop_size: int) -> np.ndarray:
    """混沌初始化（连续型）：Logistic 映射生成种群，映射到边界内。"""
    dim = problem.dim
    ch = _chaos_sequence(pop_size * dim)
    ch = ch.reshape(pop_size, dim)
    return problem.lower + ch * (problem.upper - problem.lower)


def initChaosPermutation(problem, pop_size: int) -> np.ndarray:
    """混沌初始化（排列型）：Logistic 映射 → argsort 生成排列种群。"""
    n = len(problem.coordinates)
    pop = []
    for _ in range(pop_size):
        ch = _chaos_sequence(n)
        pop.append(np.argsort(ch))
    return np.array(pop)


def initCustom(population: np.ndarray) -> np.ndarray:
    """自定义初始化：直接包装用户提供的种群。"""
    return np.array(population, copy=True)


def _chaos_sequence(length: int) -> np.ndarray:
    """生成 Logistic 映射混沌序列，值域 (0, 1)。"""
    seq = np.zeros(length)
    seq[0] = np.random.rand()
    for i in range(1, length):
        seq[i] = 4 * seq[i - 1] * (1 - seq[i - 1])
    return seq
