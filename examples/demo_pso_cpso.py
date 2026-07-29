"""混沌粒子群优化算法 (Chaotic PSO)。

对应 archive/PSO.py 中的 ``CPSO`` 类和 ``main2`` 的 CPSO 调用。

在标准 PSO 基础上增加：
- Logistic 混沌映射初始化种群
- 每代对适应度最差的粒子进行混沌重置，维持多样性
"""

import numpy as np

from intopt.algorithms import PSO
from intopt.operators.initialize import initChaosContinuous
from intopt.operators.utils import chaos_sequence
from intopt.operators.velocity import velStd
from intopt.problems import ContinuousProblem
from intopt.visualize import plot_convergence


class CPSO(PSO):
    """混沌粒子群优化算法。

    Parameters
    ----------
    chaos_reset_ratio:
        每代中混沌重置的最差粒子比例，默认 0.2。
    """

    def __init__(self, problem, chaos_reset_ratio=0.2, **kwargs):
        super().__init__(problem, **kwargs)
        self.chaos_reset_ratio = float(chaos_reset_ratio)

    # ------------------------------------------------------------------
    # 算子
    # ------------------------------------------------------------------

    def init_population(self):
        return initChaosContinuous(self.problem, self.pop_size)

    def update_velocity(self):
        return velStd(
            self.X, self.V, self.pbest, self.gbest,
            self.w, self.c1, self.c2,
        )

    # ------------------------------------------------------------------
    # 带混沌重置的主循环
    # ------------------------------------------------------------------

    def run(self, early_stopping=None):
        from tqdm import tqdm

        from intopt.algorithms.base import OptimizeResult

        self._validate_overrides()

        self.X = self.init_population()
        dim = self.X.shape[1]
        self.V = np.random.uniform(
            -self.v_max, self.v_max, (self.pop_size, dim)
        )

        self.pbest = self.X.copy()
        self.p_fit = np.array(
            [self.problem.evaluate(x) for x in self.X]
        )

        best_idx = np.argmin(self.p_fit)
        self.fit = float(self.p_fit[best_idx])
        self.gbest = self.X[best_idx].reshape(1, -1).copy()

        history_best = [self.fit]
        history_avg = [float(np.mean(self.p_fit))]

        pbar = tqdm(
            total=self.maxiter,
            desc="CPSO 优化",
            disable=not self.verbose,
        )

        lower = self.problem.lower
        upper = self.problem.upper

        for t in range(self.maxiter):
            self.w = self.w_min + (self.w - self.w_min) * (
                self.maxiter - t
            ) / self.maxiter

            self.V = self.update_velocity()
            self.V = self.clamp_velocity(self.V)
            self.X = self.update_position(self.X, self.V)

            current_fit = np.array(
                [self.problem.evaluate(x) for x in self.X]
            )

            update_mask = current_fit < self.p_fit
            self.pbest[update_mask] = self.X[update_mask]
            self.p_fit[update_mask] = current_fit[update_mask]

            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = float(current_fit[min_idx])
                self.gbest = self.X[min_idx].reshape(1, -1).copy()

            # ---- 混沌重置 ----
            num_reset = max(1, int(self.pop_size * self.chaos_reset_ratio))
            worst_indices = np.argsort(current_fit)[-num_reset:]

            for i in worst_indices:
                ch = chaos_sequence(dim)
                self.X[i] = lower + ch * (upper - lower)
                self.V[i] = np.random.uniform(-self.v_max, self.v_max, dim)
                new_fit = self.problem.evaluate(self.X[i])
                current_fit[i] = new_fit
                if new_fit < self.p_fit[i]:
                    self.p_fit[i] = new_fit
                    self.pbest[i] = self.X[i].copy()

            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = float(current_fit[min_idx])
                self.gbest = self.X[min_idx].reshape(1, -1).copy()
            # ---- 混沌重置结束 ----

            history_best.append(self.fit)
            history_avg.append(float(np.mean(current_fit)))

            if self.verbose:
                pbar.set_postfix({
                    "最优值": f"{self.fit:.4f}",
                    "平均值": f"{np.mean(current_fit):.4f}",
                })
                pbar.update(1)

            history_dict = {"best": history_best, "avg": history_avg}
            if self._check_early_stop(early_stopping, history_dict):
                break

        pbar.close()

        return OptimizeResult(
            best_solution=self.gbest[0],
            best_fitness=float(self.fit),
            history={"best": history_best, "avg": history_avg},
        )


def demo():
    """与 archive main2 等价"""
    print("=" * 50)
    print("CPSO — Rastrigin 函数 10D")
    print("=" * 50)

    def rastrigin(x):
        A = 10
        return A * len(x) + sum(
            xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x
        )

    problem = ContinuousProblem(func=rastrigin, bounds=[(-5.12, 5.12)] * 10)
    pso = CPSO(problem, pop_size=300, maxiter=500, chaos_reset_ratio=0.3)
    result = pso.run()
    print(f"最优解 : {np.round(result.best_solution, 6)}")
    print(f"最优值 : {result.best_fitness:.6f}")
    plot_convergence(result, title="CPSO — Rastrigin 10D")


if __name__ == "__main__":
    demo()
