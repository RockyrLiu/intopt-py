"""演示免疫算法 (IA) 在连续优化和 TSP 上的用法。"""

import numpy as np

from intopt.algorithms.ia import IA
from intopt.operators.initialize import initRandom
from intopt.operators.mutate import mutGaussian, mutSwap
from intopt.problems import ContinuousProblem, TSPProblem


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


# ---------------------------------------------------------------------------
# 连续 IA —— Rastrigin 10D
# ---------------------------------------------------------------------------


class ContinuousIA(IA):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)

    def mutate(self, solution):
        sigma = 0.8 / (1 + self.current_gen * 0.02)
        return self.problem.clamp(mutGaussian(solution, sigma=sigma))


def demo_continuous_ia():
    problem = ContinuousProblem(func=rastrigin, bounds=[(-5.12, 5.12)] * 10)
    ia = ContinuousIA(
        problem, pop_size=100, maxiter=500, mut_prob=0.7, Ncl=10, verbose=True
    )
    result = ia.run()
    print(f"连续 IA 最优值: {result.best_fitness:.6f}")


# ---------------------------------------------------------------------------
# TSP IA
# ---------------------------------------------------------------------------


class TspIA(IA):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)

    def mutate(self, solution):
        return mutSwap(solution)


def demo_tsp_ia():
    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)
    ia = TspIA(problem, pop_size=60, maxiter=50, mut_prob=0.7, Ncl=10, verbose=True)
    result = ia.run()
    print(f"TSP IA 最短路径: {result.best_fitness:.2f}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_continuous_ia()
    demo_tsp_ia()
