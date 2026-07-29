import numpy as np

from intopt.algorithms.base import OptimizeResult


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def test_ia_continuous_returns_optimize_result():
    from intopt.algorithms.ia import IA
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutGaussian
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class _ContIA(IA):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return self.problem.clamp(mutGaussian(solution))

    ia = _ContIA(problem, pop_size=20, maxiter=5, verbose=False)
    result = ia.run()

    assert isinstance(result, OptimizeResult)
    assert isinstance(result.best_solution, np.ndarray)
    assert result.best_solution.shape == (2,)
    assert "best" in result.history
    assert "avg" in result.history
    assert len(result.history["best"]) > 0


def test_ia_rastrigin_10d_converges():
    from intopt.algorithms.ia import IA
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutGaussian
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_rastrigin, bounds=[(-5.12, 5.12)] * 10)

    sigma0 = 3.0

    class _ContIA(IA):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            sigma = sigma0 / (1 + self.current_gen * 0.01)
            return self.problem.clamp(mutGaussian(solution, sigma=sigma))

    ia = _ContIA(problem, pop_size=100, maxiter=500, verbose=False)
    result = ia.run()

    assert result.best_fitness <= 10.0


def test_ia_concentration_penalizes_clones():
    """浓度机制应抑制相似抗体，使独一无二的个体排名更靠前。"""
    from intopt.algorithms.ia import IA
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutGaussian
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(
        func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)]
    )

    class _ContIA(IA):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return self.problem.clamp(mutGaussian(solution))

    ia = _ContIA(problem, pop_size=10, maxiter=1, verbose=False)
    ia._validate_overrides()

    pop = ia.init_population()

    pop[1] = pop[0].copy()
    pop[2] = pop[0].copy()

    motivation, _ = ia._motivation(pop)
    sorted_indices = np.argsort(motivation)

    assert sorted_indices[0] not in [1, 2], (
        "克隆抗体应因浓度惩罚而排名靠后"
    )


def test_ia_tsp_path_is_valid_permutation():
    from intopt.algorithms.ia import IA
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutSwap
    from intopt.problems import TSPProblem

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = TSPProblem(coords)

    class _TspIA(IA):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return mutSwap(solution)

    ia = _TspIA(problem, pop_size=20, maxiter=30, verbose=False)
    result = ia.run()

    assert set(result.best_solution) == set(range(4))
    assert len(result.best_solution) == 4


def test_ia_tsp_data_improves():
    from intopt.algorithms.ia import IA
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutSwap
    from intopt.problems import TSPProblem

    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)

    class _TspIA(IA):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return mutSwap(solution)

    ia = _TspIA(problem, pop_size=60, maxiter=50, verbose=False)
    result = ia.run()

    assert result.best_fitness <= result.history["best"][0]


def test_ia_early_stopping_terminates_early():
    from intopt.algorithms.ia import IA
    from intopt.early_stopping import EarlyStopping
    from intopt.operators.initialize import initRandom
    from intopt.operators.mutate import mutGaussian
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0)] * 2)

    class _ContIA(IA):
        def init_population(self):
            return initRandom(self.problem, self.pop_size)

        def mutate(self, solution):
            return self.problem.clamp(mutGaussian(solution))

    es = EarlyStopping(key="best", patience=15, min_delta=1e-6)
    ia = _ContIA(problem, pop_size=30, maxiter=200, verbose=False)
    result = ia.run(early_stopping=es)

    assert result.best_fitness < 0.2
    max_possible = ia.maxiter + 1
    assert len(result.history["best"]) < max_possible


def test_ia_must_override_methods():
    from intopt.algorithms.ia import IA
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-1, 1)] * 2)

    class _NoOverrideIA(IA):
        pass

    ia = _NoOverrideIA(problem, verbose=False)
    try:
        ia.run()
        assert False, "应抛出 NotImplementedError"
    except NotImplementedError:
        pass
