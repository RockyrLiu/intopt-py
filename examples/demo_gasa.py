import numpy as np

from intopt.algorithms.gasa import GASA
from intopt.operators.crossover import cxArithmetic, cxOrdered
from intopt.operators.initialize import initRandom
from intopt.operators.mutate import mutGaussian, mutSwap
from intopt.operators.selection import selTournament
from intopt.problems import ContinuousProblem, TSPProblem
from intopt.visualize import plot_convergence, plot_tsp_path


def _rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def demo_continuous():
    print("=" * 50)
    print("遗传模拟退火 — 连续优化 (Rastrigin 10D)")
    print("=" * 50)

    problem = ContinuousProblem(
        func=_rastrigin, bounds=[(-5.12, 5.12)] * 10
    )

    class ContGASA(GASA):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def crossover(self, p1, p2):
            return cxArithmetic(p1, p2)

        def select(self, pop, fit, k):
            return selTournament(pop, fit, k)

        def mutate(self, solution):
            return self.problem.clamp(mutGaussian(solution))

    ga = ContGASA(
        problem,
        pop_size=100,
        generations=500,
        cxpb=0.8,
        mutpb=0.2,
    )
    result = ga.run()

    print(f"最优适应度 : {result.best_fitness:.6f}")
    print(f"最优解      : {result.best_solution}")
    plot_convergence(result, title="GASA — Rastrigin 10D 收敛曲线")


def demo_tsp():
    print("\n" + "=" * 50)
    print("遗传模拟退火 — TSP (101 城市)")
    print("=" * 50)

    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)

    class TspGASA(GASA):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def crossover(self, p1, p2):
            return cxOrdered(p1, p2)

        def select(self, pop, fit, k):
            return selTournament(pop, fit, k)

        def mutate(self, solution):
            return mutSwap(solution)

    ga = TspGASA(
        problem,
        pop_size=300,
        generations=500,
        cxpb=0.8,
        mutpb=0.2,
    )
    result = ga.run()

    print(f"最优路径长度 : {result.best_fitness:.2f}")
    print(f"最优路径     : {result.best_solution}")
    plot_tsp_path(result.best_solution, problem.coordinates, result.best_fitness)


if __name__ == "__main__":
    # demo_continuous()
    demo_tsp()
