"""精英及多层精英粒子群优化算法。

核心思路：
- 运行 N 次独立 PSO，收集各次最优位置
- 以这些最优位置为初始种群，再执行一次最终 PSO
- 可多层嵌套（每层把下一层的最优位置作为自己的初始粒子）
"""

import numpy as np
from tqdm import tqdm

from intopt.algorithms import PSO
from intopt.operators.initialize import initCustom, initRandom
from intopt.operators.velocity import velStd
from intopt.problems import ContinuousProblem
from intopt.visualize import plot_convergence


class _ContinuousPSO(PSO):
    """内部使用的标准连续 PSO。"""

    def init_population(self):
        return initRandom(self.problem, self.pop_size)

    def update_velocity(self):
        return velStd(
            self.X, self.V, self.pbest, self.gbest,
            self.w, self.c1, self.c2,
        )


class ElitePSO:
    """精英粒子群优化。

    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        n_runs: 独立运行次数 (默认 10)
        pop_size: 每次运行的粒子数 (默认 30)
        maxiter: 每次运行的最大迭代次数 (默认 100)
        w: 初始惯性权重 (默认 0.8)
        w_min: 最小惯性权重 (默认 0.4)
        c1: 个体学习因子 (默认 2)
        c2: 社会学习因子 (默认 2)
        v_max: 最大速度 (默认 2)
        verbose: 是否显示进度 (默认 True)
    """

    def __init__(self, func, bounds, n_runs=10, pop_size=30, maxiter=100,
                 w=0.8, w_min=0.4, c1=2.0, c2=2.0, v_max=2.0,
                 verbose=True):
        self.func = func
        self.bounds = bounds
        self.n_runs = n_runs
        self.pop_size = pop_size
        self.maxiter = maxiter
        self.w = w
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.verbose = verbose

    def run(self):
        problem = ContinuousProblem(func=self.func, bounds=self.bounds)

        if self.verbose:
            print(f"\n{'=' * 50}")
            print(f"精英 PSO — 运行 {self.n_runs} 次独立 PSO")
            print("=" * 50)

        all_best_positions = []
        all_best_fitness = []

        for _ in tqdm(range(self.n_runs), desc="独立运行",
                       disable=not self.verbose):
            pso = _ContinuousPSO(
                problem,
                pop_size=self.pop_size,
                maxiter=self.maxiter,
                w=self.w,
                w_min=self.w_min,
                c1=self.c1,
                c2=self.c2,
                v_max=self.v_max,
                verbose=False,
            )
            result = pso.run()
            all_best_positions.append(result.best_solution)
            all_best_fitness.append(result.best_fitness)

        best_idx = np.argmin(all_best_fitness)
        if self.verbose:
            print(f"\n独立运行完成 (共 {self.n_runs} 次)")
            print(f"最佳适应度: {all_best_fitness[best_idx]:.6f}")
            print("-" * 50)

        best_positions = np.array(all_best_positions)

        if self.verbose:
            print(f"精英 PSO — 最终优化 (maxiter={self.maxiter})")
            print("-" * 50)

        class _InitPSO(_ContinuousPSO):
            def init_population(self):
                return initCustom(best_positions)

            def update_velocity(self):
                return _ContinuousPSO.update_velocity(self)

        elite = _InitPSO(
            problem,
            pop_size=self.n_runs,
            maxiter=self.maxiter,
            w=self.w,
            w_min=self.w_min,
            c1=self.c1,
            c2=self.c2,
            v_max=self.v_max,
            verbose=self.verbose,
        )
        result = elite.run()

        if self.verbose:
            print("\n优化完成")
            print(f"最终最优解 : {np.round(result.best_solution, 6)}")
            print(f"最终适应度 : {result.best_fitness:.6f}")
            print("=" * 50)

        return result


def demo():
    def rastrigin(x):
        A = 10
        return A * len(x) + sum(
            xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x
        )

    # 精英 PSO
    elite = ElitePSO(
        func=rastrigin,
        bounds=[(-5.12, 5.12)] * 10,
        n_runs=30,
        pop_size=50,
        maxiter=100,
    )
    result = elite.run()
    print(f"\n精英 PSO 最优解 : {np.round(result.best_solution, 6)}")
    print(f"精英 PSO 最优值 : {result.best_fitness:.6f}")
    plot_convergence(result, title="精英 PSO — Rastrigin 10D")


if __name__ == "__main__":
    demo()
