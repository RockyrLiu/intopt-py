import numpy as np

from intopt.algorithms.base import OptimizeResult


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def test_pso_continuous_returns_optimize_result():
    from intopt.algorithms import PSO
    from intopt.operators.initialize import initRandom
    from intopt.operators.velocity import velStd
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class _ContPSO(PSO):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def update_velocity(self):
            return velStd(
                self.X, self.V, self.pbest, self.gbest, self.w, self.c1, self.c2
            )

    pso = _ContPSO(problem, pop_size=20, maxiter=5, verbose=False)
    result = pso.run()

    assert isinstance(result, OptimizeResult)
    assert isinstance(result.best_solution, np.ndarray)
    assert result.best_solution.shape == (2,)
    assert result.best_fitness >= 0.0
    assert "best" in result.history
    assert "avg" in result.history
    assert len(result.history["best"]) > 0


def test_pso_sphere_2d_converges():
    from intopt.algorithms import PSO
    from intopt.operators.initialize import initRandom
    from intopt.operators.velocity import velStd
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class _ContPSO(PSO):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def update_velocity(self):
            return velStd(
                self.X, self.V, self.pbest, self.gbest, self.w, self.c1, self.c2
            )

    pso = _ContPSO(problem, pop_size=30, maxiter=100, verbose=False)
    result = pso.run()

    assert result.best_fitness < 0.01


def test_pso_rastrigin_10d_converges():
    from intopt.algorithms import PSO
    from intopt.operators.initialize import initRandom
    from intopt.operators.velocity import velStd
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_rastrigin, bounds=[(-5.12, 5.12)] * 10)

    class _ContPSO(PSO):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def update_velocity(self):
            return velStd(
                self.X, self.V, self.pbest, self.gbest, self.w, self.c1, self.c2
            )

    pso = _ContPSO(problem, pop_size=100, maxiter=500, verbose=False)
    result = pso.run()

    assert result.best_fitness <= 10.0


def test_pso_early_stopping_terminates_early():
    from intopt.algorithms import PSO
    from intopt.early_stopping import EarlyStopping
    from intopt.operators.initialize import initRandom
    from intopt.operators.velocity import velStd
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0)] * 2)

    class _ContPSO(PSO):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def update_velocity(self):
            return velStd(
                self.X, self.V, self.pbest, self.gbest, self.w, self.c1, self.c2
            )

    es = EarlyStopping(key="best", patience=20, min_delta=1e-6)
    pso = _ContPSO(problem, pop_size=30, maxiter=200, verbose=False)
    result = pso.run(early_stopping=es)

    assert result.best_fitness < 0.1
    max_possible = pso.maxiter + 1
    assert len(result.history["best"]) < max_possible
