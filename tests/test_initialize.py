import numpy as np

from intopt.problems import ContinuousProblem, TSPProblem


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def test_init_chaos_continuous():
    from intopt.operators.initialize import initChaosContinuous

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    pop = initChaosContinuous(problem, pop_size=20)
    assert pop.shape == (20, 2)
    assert np.all(pop >= -5.0) and np.all(pop <= 5.0)


def test_init_chaos_permutation():
    from intopt.operators.initialize import initChaosPermutation

    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
    problem = TSPProblem(coords)
    pop = initChaosPermutation(problem, pop_size=10)
    assert pop.shape == (10, 5)
    for ind in pop:
        assert set(ind) == set(range(5))
