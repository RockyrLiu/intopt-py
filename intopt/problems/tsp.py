import numpy as np

from intopt.problems.base import Problem


class TSPProblem(Problem):
    """旅行商问题（TSP）。

    Parameters
    ----------
    coordinates:
        城市坐标，shape ``(n, 2)``。
    """

    def __init__(self, coordinates: np.ndarray):
        self.coordinates = np.array(coordinates, dtype=float)
        self.n = len(coordinates)
        self.dist_matrix = self._compute_dist_matrix()

    def _compute_dist_matrix(self) -> np.ndarray:
        diff = self.coordinates[:, np.newaxis, :] - self.coordinates[np.newaxis, :, :]
        return np.sqrt((diff**2).sum(axis=-1))

    def random_solution(self) -> np.ndarray:
        return np.random.permutation(self.n)

    def evaluate(self, solution: np.ndarray) -> float:
        idx = solution.astype(int)
        return float(
            self.dist_matrix[idx[:-1], idx[1:]].sum()
            + self.dist_matrix[idx[-1], idx[0]]
        )

    def is_feasible(self, solution: np.ndarray) -> bool:
        return bool(len(solution) == self.n and len(set(solution)) == self.n)

    def clamp(self, solution: np.ndarray) -> np.ndarray:
        s = solution.astype(int)
        if len(s) == self.n and set(s) == set(range(self.n)):
            return s
        return np.random.permutation(self.n)

