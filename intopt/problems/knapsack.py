import numpy as np

from intopt.problems.base import Problem


class KnapsackProblem(Problem):
    """0-1 背包问题。

    Parameters
    ----------
    capacity:
        背包容量上限。
    weights:
        各物品重量。
    values:
        各物品价值。
    penalty:
        超重惩罚系数，默认 2.0。
    """

    def __init__(
        self,
        capacity: float,
        weights: list[float],
        values: list[float],
        penalty: float = 2.0,
    ):
        if len(weights) != len(values):
            raise ValueError("weights 与 values 长度必须相同")
        self.capacity = float(capacity)
        self.weights = np.array(weights, dtype=float)
        self.values = np.array(values, dtype=float)
        self.penalty = float(penalty)
        self.n_items = len(weights)

    def random_solution(self) -> np.ndarray:
        return np.random.randint(0, 2, size=self.n_items).astype(float)

    def evaluate(self, solution: np.ndarray) -> float:
        total_weight = float(solution @ self.weights)
        total_value = float(solution @ self.values)
        excess = max(total_weight - self.capacity, 0.0)
        return total_value - excess * self.penalty

    def is_feasible(self, solution: np.ndarray) -> bool:
        return bool(float(solution @ self.weights) <= self.capacity)

