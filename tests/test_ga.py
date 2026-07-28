import numpy as np

from intopt.algorithms.base import OptimizeResult


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def test_ga_continuous_returns_optimize_result():
    from intopt.algorithms import GA
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    ga = GA(problem, pop_size=20, generations=5, verbose=False)
    result = ga.run()

    assert isinstance(result, OptimizeResult)
    assert isinstance(result.best_solution, np.ndarray)
    assert result.best_solution.shape == (2,)
    assert "best" in result.history
    assert "avg" in result.history
    assert len(result.history["best"]) > 0


def test_ga_rastrigin_10d_converges():
    from intopt.algorithms import GA
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_rastrigin, bounds=[(-5.12, 5.12)] * 10)
    ga = GA(problem, pop_size=100, generations=500, verbose=False)
    result = ga.run()

    assert result.best_fitness <= 1.0


def test_ga_with_init_solutions():
    from intopt.algorithms import GA
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_rastrigin, bounds=[(-5.12, 5.12)] * 10)
    rng = np.random.default_rng(0)
    init = rng.uniform(-0.5, 0.5, size=(50, 10))

    ga_init = GA(problem, pop_size=50, generations=20, init_solutions=init, verbose=False)
    ga_rand = GA(problem, pop_size=50, generations=20, verbose=False)

    result_init = ga_init.run()
    result_rand = ga_rand.run()

    assert result_init.best_fitness < result_rand.best_fitness


def test_ga_tsp_path_is_valid_permutation():
    from intopt.operators.crossover import cxOrdered

    from intopt.algorithms import GA
    from intopt.problems import TSPProblem

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = TSPProblem(coords)

    class _TspGA(GA):
        def crossover(self, p1, p2):
            return cxOrdered(p1, p2)

    ga = _TspGA(problem, pop_size=20, generations=30, verbose=False)
    result = ga.run()

    assert set(result.best_solution) == set(range(4))
    assert len(result.best_solution) == 4


def test_ga_tsp_data_improves():
    from intopt.operators.crossover import cxOrdered

    from intopt.algorithms import GA
    from intopt.problems import TSPProblem

    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)

    class _TspGA(GA):
        def crossover(self, p1, p2):
            return cxOrdered(p1, p2)

    ga = _TspGA(problem, pop_size=60, generations=50, verbose=False)
    result = ga.run()

    assert result.best_fitness <= result.history["best"][0]


def test_ga_early_stopping_terminates_early():
    from intopt.algorithms import GA
    from intopt.early_stopping import EarlyStopping
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0)] * 2)
    es = EarlyStopping(key="best", patience=15, min_delta=1e-6)
    ga = GA(problem, pop_size=30, generations=200, verbose=False)
    result = ga.run(early_stopping=es)

    assert result.best_fitness < 0.2
    max_possible = ga.generations + 1
    assert len(result.history["best"]) < max_possible
