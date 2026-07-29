import numpy as np

from intopt.algorithms import TS
from intopt.problems import ContinuousProblem, TSPProblem
from intopt.visualize import plot_convergence, plot_tsp_path


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def demo_continuous():
    print("=" * 50)
    print("禁忌搜索 — 连续优化 (Sphere 2D)")
    print("=" * 50)

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)])

    class ContTS(TS):
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

    ts = ContTS(problem, tabu_length=10, candidate_size=50, max_iter=200)
    result = ts.run()

    print(f"最优适应度 : {result.best_fitness:.6f}")
    print(f"最优解      : {result.best_solution}")
    plot_convergence(result, title="TS — Sphere 2D 收敛曲线")


def demo_tsp():
    print("\n" + "=" * 50)
    print("禁忌搜索 — TSP (101 城市)")
    print("=" * 50)

    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)
    N = len(coords)

    class TspTS(TS):
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

    ts = TspTS(problem, candidate_size=50, max_iter=500)
    result = ts.run()

    print(f"最优路径长度 : {result.best_fitness:.2f}")
    print(f"最优路径     : {result.best_solution}")
    plot_tsp_path(result.best_solution, problem.coordinates, result.best_fitness)
    plot_convergence(result, title="TS — TSP 收敛曲线")


if __name__ == "__main__":
    demo_continuous()
    # demo_tsp()
