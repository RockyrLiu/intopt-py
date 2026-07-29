"""演示差分进化算法 (DE) 在连续优化上的用法。"""

import numpy as np

from intopt.algorithms.de import DE
from intopt.operators.crossover import cxBinomial
from intopt.operators.initialize import initRandom
from intopt.operators.mutate import mutDERand1
from intopt.problems import ContinuousProblem


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


# ---------------------------------------------------------------------------
# 连续 DE —— Rastrigin 10D
# ---------------------------------------------------------------------------


class ContinuousDE(DE):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)

    def mutate(self, solution):
        return mutDERand1(solution, self.population, self.fitness, self.F)

    def crossover(self, target, donor):
        return cxBinomial(target, donor, self.CR)


def demo():
    problem = ContinuousProblem(func=rastrigin, bounds=[(-5.12, 5.12)] * 10)
    de = ContinuousDE(problem, pop_size=100, maxiter=500, verbose=True)
    result = de.run()
    print(f"DE 最优值: {result.best_fitness:.6f}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo()
