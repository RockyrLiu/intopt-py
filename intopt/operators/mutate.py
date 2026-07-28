import numpy as np

from intopt.operators.utils import chaos_sequence


def mutSwap(solution: np.ndarray) -> np.ndarray:
    """交换变异：随机交换两个位置。用于排列型解（如 TSP 路径）。"""
    n = len(solution)
    i, j = np.random.choice(
        n, size=2, replace=False
    )  # 从 [0, n) 中无放回地选择 2 个不同的整数
    new_sol = solution.copy()
    new_sol[i], new_sol[j] = solution[j], solution[i]
    return new_sol


def mutFlip(solution: np.ndarray) -> np.ndarray:
    """翻转变异：随机翻转一个 0/1 位。用于二值型解（如 0-1 背包）。"""
    n = len(solution)
    idx = np.random.randint(n)  # 从 [0, n) 中随机选择一个整数
    new_sol = solution.copy()
    new_sol[idx] = 1.0 - new_sol[idx]
    return new_sol


def mutGaussian(
    solution: np.ndarray,
    mu: float = 0.0,
    sigma: float = 0.5,
    indpb: float = 0.5,
) -> np.ndarray:
    """高斯变异：每个维度以概率 ``indpb`` 变异，扰动值服从 N(mu, sigma²)。用于连续型解。"""
    n = len(solution)
    mask = np.random.random(n) < indpb
    new_sol = solution.copy()
    new_sol[mask] += mu + sigma * np.random.randn(np.sum(mask))
    return new_sol


def mutChaosSwap(solution: np.ndarray) -> np.ndarray:
    """混沌交换变异：交换位置由 Logistic 序列确定。用于排列型解。"""
    n = len(solution)
    ch = chaos_sequence(2)
    i, j = np.clip(np.floor(ch * n).astype(int), 0, n - 1)
    new_sol = solution.copy()
    new_sol[i], new_sol[j] = solution[j], solution[i]
    return new_sol
