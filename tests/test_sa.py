import numpy as np

from intopt.algorithms.base import OptimizeResult
from intopt.problems import ContinuousProblem, TSPProblem


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def test_sa_continuous_returns_optimize_result():
    from intopt.algorithms.sa import SA

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    sa = SA(problem, verbose=False)
    result = sa.run()

    assert isinstance(result, OptimizeResult)
    assert isinstance(result.best_solution, np.ndarray)
    assert result.best_solution.shape == (2,)
    assert result.best_fitness >= 0.0
    assert "best" in result.history
    assert len(result.history["best"]) > 0


def test_sa_sphere_2d_converges():
    from intopt.algorithms import SA

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    sa = SA(problem, initial_temp=100, final_temp=1e-3,
            cooling_rate=0.99, iter_per_temp=100, verbose=False)
    result = sa.run()

    assert result.best_fitness < 0.1


def test_sa_rastrigin_10d_converges():
    from intopt.algorithms import SA

    problem = ContinuousProblem(func=_rastrigin, bounds=[(-5.12, 5.12)] * 10)
    sa = SA(problem, initial_temp=100, final_temp=1e-3,
            cooling_rate=0.999, iter_per_temp=100, verbose=False)
    result = sa.run()

    assert result.best_fitness <= 1.0


def test_sa_tsp_path_is_valid_permutation():
    from intopt.algorithms import SA

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = TSPProblem(coords)
    sa = SA(problem, initial_temp=100, iter_per_temp=10, verbose=False)
    result = sa.run()

    assert set(result.best_solution) == set(range(4))
    assert len(result.best_solution) == 4


def test_sa_tsp_data_improves():
    from intopt.algorithms import SA

    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)
    sa = SA(problem, initial_temp=100, final_temp=1,
            cooling_rate=0.9, iter_per_temp=10, verbose=False)
    result = sa.run()

    assert result.best_fitness <= result.history["best"][0]
