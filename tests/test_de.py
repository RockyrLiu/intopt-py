import numpy as np

from intopt.algorithms.base import OptimizeResult


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def test_de_continuous_returns_optimize_result():
    from intopt.algorithms.de import DE
    from intopt.operators.crossover import cxBinomial
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutDERand1
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class _ContDE(DE):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return mutDERand1(solution, self.population, self.fitness, self.F)

        def crossover(self, target, donor):
            return cxBinomial(target, donor, self.CR)

    de = _ContDE(problem, pop_size=20, maxiter=5, verbose=False)
    result = de.run()

    assert isinstance(result, OptimizeResult)
    assert isinstance(result.best_solution, np.ndarray)
    assert result.best_solution.shape == (2,)
    assert "best" in result.history
    assert "avg" in result.history
    assert len(result.history["best"]) > 0


def test_de_rastrigin_10d_converges():
    from intopt.algorithms.de import DE
    from intopt.operators.crossover import cxBinomial
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutDECurrentToBest1
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(
        func=_rastrigin, bounds=[(-5.12, 5.12)] * 10
    )

    class _ContDE(DE):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return mutDECurrentToBest1(
                solution, self.population, self.fitness, self.F
            )

        def crossover(self, target, donor):
            return cxBinomial(target, donor, self.CR)

    de = _ContDE(problem, pop_size=100, maxiter=700, verbose=False)
    result = de.run()

    assert result.best_fitness <= 7.0


def test_de_sphere_2d_converges():
    from intopt.algorithms.de import DE
    from intopt.operators.crossover import cxBinomial
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutDERand1
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class _ContDE(DE):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return mutDERand1(solution, self.population, self.fitness, self.F)

        def crossover(self, target, donor):
            return cxBinomial(target, donor, self.CR)

    de = _ContDE(problem, pop_size=30, maxiter=100, verbose=False)
    result = de.run()

    assert result.best_fitness < 0.01


def test_de_early_stopping_terminates_early():
    from intopt.algorithms.de import DE
    from intopt.early_stopping import EarlyStopping
    from intopt.operators.crossover import cxBinomial
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutDERand1
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0)] * 2)

    class _ContDE(DE):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return mutDERand1(solution, self.population, self.fitness, self.F)

        def crossover(self, target, donor):
            return cxBinomial(target, donor, self.CR)

    es = EarlyStopping(key="best", patience=15, min_delta=1e-6)
    de = _ContDE(problem, pop_size=30, maxiter=200, verbose=False)
    result = de.run(early_stopping=es)

    assert result.best_fitness < 0.2
    max_possible = de.maxiter + 1
    assert len(result.history["best"]) < max_possible


def test_de_must_override_methods():
    from intopt.algorithms.de import DE
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-1, 1)] * 2)

    class _NoOverrideDE(DE):
        pass

    de = _NoOverrideDE(problem, verbose=False)
    try:
        de.run()
        assert False, "应抛出 NotImplementedError"
    except NotImplementedError:
        pass
