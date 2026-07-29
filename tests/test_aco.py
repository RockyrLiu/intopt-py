import numpy as np

from intopt.algorithms.base import OptimizeResult
from intopt.problems import ContinuousProblem, TSPProblem


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


# ------------------------------------------------------------------
# TSP 测试
# ------------------------------------------------------------------


def test_aco_tsp_returns_optimize_result():
    from intopt.algorithms.aco import ACO

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = TSPProblem(coords)
    aco = ACO(problem, m=10, max_iter=5, verbose=False)
    result = aco.run()

    assert isinstance(result, OptimizeResult)
    assert isinstance(result.best_solution, np.ndarray)
    assert result.best_solution.shape == (4,)
    assert result.best_fitness >= 0.0
    assert "best" in result.history
    assert "avg" in result.history
    assert len(result.history["best"]) > 0


def test_aco_tsp_path_is_valid_permutation():
    from intopt.algorithms.aco import ACO

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = TSPProblem(coords)
    aco = ACO(problem, m=20, max_iter=10, verbose=False)
    result = aco.run()

    assert set(result.best_solution) == set(range(4))
    assert len(result.best_solution) == 4


def test_aco_tsp_data_improves():
    from intopt.algorithms.aco import ACO

    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)
    aco = ACO(problem, m=30, max_iter=20, verbose=False)
    result = aco.run()

    assert result.best_fitness <= result.history["best"][0]


# ------------------------------------------------------------------
# 连续 ACO 测试（覆写方法）
# ------------------------------------------------------------------


def test_aco_continuous_returns_optimize_result():
    from intopt.algorithms.aco import ACO

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class ContACO(ACO):
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
            new_positions = population.copy()
            for i in range(self.m):
                P = (Tau_best - self.Tau[i]) / (Tau_best + 1e-10)
                if P < 0.2:
                    delta = 2 * np.random.rand(problem.dim) - 1
                    new_positions[i] += delta * 0.1 * lamda
                else:
                    for d in range(problem.dim):
                        range_d = problem.upper[d] - problem.lower[d]
                        new_positions[i, d] += (
                            np.random.rand() - 0.5
                        ) * range_d
                new_positions[i] = self.problem.clamp(new_positions[i])

            new_fit = np.array([
                self.problem.evaluate(p) for p in new_positions
            ])
            old_fit = np.array([
                self.problem.evaluate(p) for p in population
            ])
            for i in range(self.m):
                if new_fit[i] < old_fit[i]:
                    population[i] = new_positions[i]
            return population

        def update_pheromone(self, population, fitness):
            if self.strategy == "AS":
                self.Tau = (1 - self.rho) * self.Tau + self.Q / (
                    fitness + 1e-10
                )
            else:
                self.Tau = (1 - self.rho) * self.Tau + self.Q / (
                    fitness + 1e-10
                )

    aco = ContACO(problem, m=10, max_iter=5, verbose=False)
    result = aco.run()

    assert isinstance(result, OptimizeResult)
    assert isinstance(result.best_solution, np.ndarray)
    assert result.best_solution.shape == (2,)
    assert "best" in result.history
    assert "avg" in result.history
    assert len(result.history["best"]) > 0


def test_aco_continuous_sphere_2d_converges():
    from intopt.algorithms.aco import ACO

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class ContACO(ACO):
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
            new_positions = population.copy()
            for i in range(self.m):
                P = (Tau_best - self.Tau[i]) / (Tau_best + 1e-10)
                if P < 0.2:
                    delta = 2 * np.random.rand(problem.dim) - 1
                    new_positions[i] += delta * 0.1 * lamda
                else:
                    for d in range(problem.dim):
                        range_d = problem.upper[d] - problem.lower[d]
                        new_positions[i, d] += (
                            np.random.rand() - 0.5
                        ) * range_d
                new_positions[i] = self.problem.clamp(new_positions[i])

            new_fit = np.array([
                self.problem.evaluate(p) for p in new_positions
            ])
            old_fit = np.array([
                self.problem.evaluate(p) for p in population
            ])
            for i in range(self.m):
                if new_fit[i] < old_fit[i]:
                    population[i] = new_positions[i]
            return population

        def update_pheromone(self, population, fitness):
            self.Tau = (1 - self.rho) * self.Tau + self.Q / (
                fitness + 1e-10
            )

    aco = ContACO(problem, m=30, max_iter=100, verbose=False)
    result = aco.run()

    assert result.best_fitness < 10.0
