"""演示蚁群算法 (ACO) 在连续优化上的用法（覆写 _initialize/build_solutions/update_pheromone）。"""
import numpy as np

from intopt.algorithms.aco import ACO
from intopt.problems import ContinuousProblem


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


class ContinuousACO(ACO):
    def init_population(self):
        return np.array([
            self.problem.random_solution() for _ in range(self.m)
        ])

    def initialize(self, population):
        self.Tau = np.ones(self.m)

    def build_solutions(self, population):
        lamda = 1 / (self.current_gen + 1)
        best_idx = int(np.argmin(self.Tau))
        Tau_best = self.Tau[best_idx]
        dim = self.problem.dim
        new_positions = population.copy()
        for i in range(self.m):
            P = (Tau_best - self.Tau[i]) / (Tau_best + 1e-10)
            if P < 0.2:
                delta = 2 * np.random.rand(dim) - 1
                new_positions[i] += delta * 0.1 * lamda
            else:
                for d in range(dim):
                    range_d = self.problem.upper[d] - self.problem.lower[d]
                    new_positions[i, d] += (np.random.rand() - 0.5) * range_d
            new_positions[i] = self.problem.clamp(new_positions[i])

        new_fit = np.array([self.problem.evaluate(p) for p in new_positions])
        old_fit = np.array([self.problem.evaluate(p) for p in population])
        for i in range(self.m):
            if new_fit[i] < old_fit[i]:
                population[i] = new_positions[i]
        return population

    def update_pheromone(self, population, fitness):
        self.Tau = (1 - self.rho) * self.Tau + self.Q / (fitness + 1e-10)


def demo():
    problem = ContinuousProblem(
        func=sphere, bounds=[(-5.12, 5.12), (-5.12, 5.12)]
    )
    aco = ContinuousACO(problem, m=30, max_iter=100, verbose=True)
    result = aco.run()
    print(f"最优值: {result.best_fitness:.6f}")


if __name__ == "__main__":
    demo()
