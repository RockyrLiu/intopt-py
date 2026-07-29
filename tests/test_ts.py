import numpy as np

from intopt.algorithms.base import OptimizeResult


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def test_ts_continuous_returns_optimize_result():
    from intopt.algorithms.ts import TS
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class _ContTS(TS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._range = np.array([5.0 - (-5.0), 5.0 - (-5.0)])
            self._tabu_list = []

        def init_solution(self):
            return self.problem.random_solution()

        def generate_candidates(self, solution):
            candidates = []
            for _ in range(self.candidate_size):
                new_sol = solution + np.random.uniform(
                    -0.1 * self._range, 0.1 * self._range
                )
                new_sol = self.problem.clamp(new_sol)
                candidates.append({
                    "solution": new_sol,
                    "move": new_sol - solution,
                })
            return candidates

        def update_tabu(self, move):
            self._tabu_list.append(move)
            if len(self._tabu_list) > self.tabu_length:
                self._tabu_list.pop(0)

        def is_tabu(self, move):
            for t in self._tabu_list:
                if np.allclose(move, t, atol=1e-6):
                    return True
            return False

    ts = _ContTS(problem, tabu_length=10, candidate_size=50, max_iter=5, verbose=False)
    result = ts.run()

    assert isinstance(result, OptimizeResult)
    assert isinstance(result.best_solution, np.ndarray)
    assert result.best_solution.shape == (2,)
    assert result.best_fitness >= 0.0
    assert "best" in result.history
    assert "current" in result.history
    assert len(result.history["best"]) > 0


def test_ts_sphere_2d_converges():
    from intopt.algorithms.ts import TS
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class _ContTS(TS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._range = np.array([5.0 - (-5.0), 5.0 - (-5.0)])
            self._tabu_list = []

        def init_solution(self):
            return self.problem.random_solution()

        def generate_candidates(self, solution):
            candidates = []
            for _ in range(self.candidate_size):
                new_sol = solution + np.random.uniform(
                    -0.1 * self._range, 0.1 * self._range
                )
                new_sol = self.problem.clamp(new_sol)
                candidates.append({
                    "solution": new_sol,
                    "move": new_sol - solution,
                })
            return candidates

        def update_tabu(self, move):
            self._tabu_list.append(move)
            if len(self._tabu_list) > self.tabu_length:
                self._tabu_list.pop(0)

        def is_tabu(self, move):
            for t in self._tabu_list:
                if np.allclose(move, t, atol=1e-6):
                    return True
            return False

    ts = _ContTS(problem, tabu_length=10, candidate_size=50, max_iter=200, verbose=False)
    result = ts.run()

    assert result.best_fitness < 0.1


def test_ts_tsp_path_is_valid_permutation():
    from intopt.algorithms.ts import TS
    from intopt.problems import TSPProblem

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = TSPProblem(coords)

    class _TspTS(TS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.N = len(coords)
            self._tabu_table = np.zeros((self.N, self.N))

        def init_solution(self):
            return self.problem.random_solution()

        def generate_candidates(self, solution):
            candidates = []
            generated = set()
            while len(candidates) < min(self.candidate_size, self.N * (self.N - 1) // 2):
                i, j = np.random.choice(self.N, size=2, replace=False)
                if (i, j) in generated:
                    continue
                generated.add((i, j))
                new_sol = solution.copy()
                new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
                candidates.append({
                    "solution": new_sol,
                    "move": (i, j),
                })
            return candidates

        def update_tabu(self, move):
            self._tabu_table = np.maximum(self._tabu_table - 1, 0)
            i, j = move
            self._tabu_table[i, j] = self.tabu_length
            self._tabu_table[j, i] = self.tabu_length

        def is_tabu(self, move):
            i, j = move
            return self._tabu_table[i, j] > 0

    ts = _TspTS(problem, candidate_size=10, max_iter=30, verbose=False)
    result = ts.run()

    assert set(result.best_solution) == set(range(4))
    assert len(result.best_solution) == 4


def test_ts_tsp_data_improves():
    from intopt.algorithms.ts import TS
    from intopt.problems import TSPProblem

    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)
    N = len(coords)

    class _TspTS(TS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.N = N
            self._tabu_table = np.zeros((self.N, self.N))

        def init_solution(self):
            return self.problem.random_solution()

        def generate_candidates(self, solution):
            candidates = []
            generated = set()
            while len(candidates) < self.candidate_size:
                i, j = np.random.choice(self.N, size=2, replace=False)
                if (i, j) in generated:
                    continue
                generated.add((i, j))
                new_sol = solution.copy()
                new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
                candidates.append({
                    "solution": new_sol,
                    "move": (i, j),
                })
            return candidates

        def update_tabu(self, move):
            self._tabu_table = np.maximum(self._tabu_table - 1, 0)
            i, j = move
            self._tabu_table[i, j] = self.tabu_length
            self._tabu_table[j, i] = self.tabu_length

        def is_tabu(self, move):
            i, j = move
            return self._tabu_table[i, j] > 0

    ts = _TspTS(problem, candidate_size=50, max_iter=50, verbose=False)
    result = ts.run()

    assert result.best_fitness <= result.history["best"][0]


def test_ts_early_stopping_terminates_early():
    from intopt.algorithms.ts import TS
    from intopt.early_stopping import EarlyStopping
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0)] * 2)

    class _ContTS(TS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._range = np.array([10.0, 10.0])
            self._tabu_list = []

        def init_solution(self):
            return self.problem.random_solution()

        def generate_candidates(self, solution):
            candidates = []
            for _ in range(self.candidate_size):
                new_sol = solution + np.random.uniform(
                    -0.1 * self._range, 0.1 * self._range
                )
                new_sol = self.problem.clamp(new_sol)
                candidates.append({
                    "solution": new_sol,
                    "move": new_sol - solution,
                })
            return candidates

        def update_tabu(self, move):
            self._tabu_list.append(move)
            if len(self._tabu_list) > self.tabu_length:
                self._tabu_list.pop(0)

        def is_tabu(self, move):
            for t in self._tabu_list:
                if np.allclose(move, t, atol=1e-6):
                    return True
            return False

    es = EarlyStopping(key="best", patience=20, min_delta=1e-6)
    ts = _ContTS(problem, tabu_length=10, candidate_size=50, max_iter=200, verbose=False)
    result = ts.run(early_stopping=es)

    assert result.best_fitness < 0.2
    max_possible = ts.max_iter + 1
    assert len(result.history["best"]) < max_possible


def test_ts_must_override_methods():
    from intopt.algorithms.ts import TS
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-1, 1)] * 2)

    class _NoOverrideTS(TS):
        pass

    ts = _NoOverrideTS(problem, verbose=False)
    try:
        ts.run()
        assert False, "应抛出 NotImplementedError"
    except NotImplementedError:
        pass
