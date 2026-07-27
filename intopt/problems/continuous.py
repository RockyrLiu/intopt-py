from collections.abc import Callable

import numpy as np

from intopt.operators.mutate import mutGaussian
from intopt.problems.base import Problem


class ContinuousProblem(Problem):
    """连续优化问题。

    Parameters
    ----------
    func:
        目标函数，签名 ``f(x: np.ndarray) -> float``。
        输入为 1D ndarray，返回标量。值越小越好。
    bounds:
        每个维度的 ``(min, max)`` 边界。

    如需自定义变异参数，继承此类并覆写 ``mutate`` 方法即可：
    ``return mutGaussian(solution, sigma=1.0)``。
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
    ):
        self.func = func
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.dim = len(bounds)
        self.lower = np.array([b[0] for b in self.bounds])
        self.upper = np.array([b[1] for b in self.bounds])

    def random_solution(self) -> np.ndarray:
        return np.random.uniform(self.lower, self.upper)

    def evaluate(self, solution: np.ndarray) -> float:
        return float(self.func(solution))

    def is_feasible(self, solution: np.ndarray) -> bool:
        return bool(np.all(solution >= self.lower) and np.all(solution <= self.upper))

    def mutate(self, solution: np.ndarray) -> np.ndarray:
        return mutGaussian(solution)

    def clamp(self, solution: np.ndarray) -> np.ndarray:
        """将解钳制到边界内。"""
        return np.clip(solution, self.lower, self.upper)

