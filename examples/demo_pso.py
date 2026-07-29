import numpy as np

from intopt.algorithms import PSO
from intopt.operators.initialize import initRandom
from intopt.operators.velocity import velStd
from intopt.problems import ContinuousProblem, KnapsackProblem
from intopt.visualize import plot_convergence

# =========================================================================
# 连续 PSO
# =========================================================================


class ContinuousPSO(PSO):
    """标准连续 PSO。"""

    def init_population(self):
        return initRandom(self.problem, self.pop_size)

    def update_velocity(self):
        return velStd(
            self.X,
            self.V,
            self.pbest,
            self.gbest,
            self.w,
            self.c1,
            self.c2,
        )


# =========================================================================
# 离散 PSO
# =========================================================================


class BinaryPSO(PSO):
    """离散粒子群优化算法 (0-1 问题)。

    与标准 PSO 的区别：
    - 位置是二进制向量
    - 位置更新使用 sigmoid 概率映射
    """

    def init_population(self):
        n = self.problem.n_items
        return np.random.randint(0, 2, (self.pop_size, n)).astype(float)

    def update_velocity(self):
        popsize, dim = self.X.shape
        r1 = np.random.rand(popsize, dim)
        r2 = np.random.rand(popsize, dim)
        gbest_matrix = np.broadcast_to(self.gbest, (popsize, dim))
        return (
            self.w * self.V
            + self.c1 * r1 * (self.pbest - self.X)
            + self.c2 * r2 * (gbest_matrix - self.X)
        )

    def update_position(self, positions, velocities):
        sig_v = 1.0 / (1.0 + np.exp(-velocities))
        return (np.random.rand(*sig_v.shape) < sig_v).astype(float)

    def clamp_position(self, positions):
        return np.clip(positions, 0.0, 1.0)


def _make_minimize_problem(kp):
    """将 KnapsackProblem 包装为最小化问题。"""

    class _MinKnapsack:
        def __init__(self):
            self.n_items = kp.n_items

        def evaluate(self, solution):
            return -kp.evaluate(solution)

        def clamp(self, solution):
            return np.clip(solution, 0.0, 1.0)

    return _MinKnapsack()


# =========================================================================
# 演示
# =========================================================================


def demo_continuous():
    """连续 PSO — Rastrigin 10D"""
    print("=" * 50)
    print("连续 PSO — Rastrigin 10D")
    print("=" * 50)

    def rastrigin(x):
        A = 10
        return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

    problem = ContinuousProblem(func=rastrigin, bounds=[(-5.12, 5.12)] * 10)
    pso = ContinuousPSO(problem, pop_size=30, maxiter=500)
    result = pso.run()
    print(f"最优解 : {np.round(result.best_solution, 6)}")
    print(f"最优值 : {result.best_fitness:.6f}")
    plot_convergence(result, title="连续 PSO — Rastrigin 10D")


def demo_binary():
    """离散 PSO — 0-1 背包"""
    print("\n" + "=" * 50)
    print("离散 PSO — 0-1 背包问题")
    print("=" * 50)

    capacity = 300.0
    volumes = [95, 75, 23, 73, 50, 22, 6, 57, 89, 98]
    values = [89, 59, 19, 43, 100, 72, 44, 16, 7, 64]
    penalty = 2.0

    kp = KnapsackProblem(capacity, volumes, values, penalty)
    problem = _make_minimize_problem(kp)

    pso = BinaryPSO(problem, pop_size=300, maxiter=500)
    result = pso.run()

    best = result.best_solution
    selected = np.where(best > 0.5)[0]
    total_value = sum(v for i, v in enumerate(values) if i in selected)
    total_weight = sum(v for i, v in enumerate(volumes) if i in selected)

    print(f"\n选中的物品索引 : {selected.tolist()}")
    print(f"总价值 : {total_value}")
    print(f"总重量 : {total_weight} (容量: {int(capacity)})")
    print(f"最优适应度 : {result.best_fitness:.4f}")


if __name__ == "__main__":
    demo_continuous()
    demo_binary()
