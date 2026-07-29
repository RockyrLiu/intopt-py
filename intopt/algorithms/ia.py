import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import Optimizer, OptimizeResult


class IA(Optimizer):
    """免疫算法 (Immune Algorithm)。

    免疫算法模拟生物免疫系统的克隆选择机制：
    对抗体进行克隆、高频变异，并通过浓度机制抑制相似抗体，
    维持种群多样性，同时随机刷新部分个体防止早熟。

    **必须重写的方法**：

    - ``init_population()`` — 初始化种群。连续型用 `initRandom`_，
      混沌连续型用 `initChaosContinuous`_，排列型用
      `initChaosPermutation`_。
    - ``mutate(solution)`` — 变异操作。连续型用 `mutGaussian`_，
      排列型用 `mutSwap`_，二值型用 `mutFlip`_。

      如需实现动态变异幅度，可通过 ``self.current_gen`` 获取
      当前代数（0-based），例如 ``sigma = sigma0 / (self.current_gen + 1)``。

    **可选重写的方法**：

    - ``distance(a, b)`` — 两个抗体间的距离，默认使用欧氏距离。
      浓度机制依赖此距离判断抗体相似度。

    Parameters
    ----------
    problem:
        待求解的优化问题实例。
    pop_size:
        种群大小，默认 100。
    maxiter:
        最大迭代次数，默认 100。
    mut_prob:
        变异概率，默认 0.7。
    sim_thresh:
        相似度阈值，距离小于此值视为相似抗体，默认 0.2。
    Ncl:
        克隆数量，每个抗体克隆 Ncl 份，默认 10。
    refresh_rate:
        种群刷新率，每代重新随机生成的比例，默认 0.5。
    alpha:
        亲和度权重，默认 1。
    beta:
        浓度惩罚权重，默认 1。
    verbose:
        是否显示进度条，默认 True。
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        maxiter: int = 100,
        mut_prob: float = 0.7,
        sim_thresh: float = 0.2,
        Ncl: int = 10,
        refresh_rate: float = 0.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        verbose: bool = True,
    ):
        super().__init__(problem)
        self.pop_size = int(pop_size)
        self.maxiter = int(maxiter)
        self.mut_prob = float(mut_prob)
        self.sim_thresh = float(sim_thresh)
        self.Ncl = int(Ncl)
        self.refresh_rate = float(refresh_rate)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.verbose = verbose
        self.population: np.ndarray | None = None
        self.current_gen: int = 0

    # ------------------------------------------------------------------
    # 必须重写的方法
    # ------------------------------------------------------------------

    def init_population(self) -> np.ndarray:
        """初始化种群。**必须重写**。

        Returns
        -------
        np.ndarray
            种群，shape ``(pop_size, dim)``。
        """
        raise NotImplementedError("Override init_population method")

    def mutate(self, solution: np.ndarray) -> np.ndarray:
        """变异操作。**必须重写**。

        如需动态变异幅度，可通过 ``self.current_gen`` 获取当前代数。

        Parameters
        ----------
        solution:
            待变异的抗体。

        Returns
        -------
        np.ndarray
            变异后的抗体。
        """
        raise NotImplementedError("Override mutate method")

    # ------------------------------------------------------------------
    # 可选重写的方法
    # ------------------------------------------------------------------

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        r"""两个抗体间的距离。默认使用欧氏距离。

        用于浓度计算，判断抗体相似度。排列型问题（如 TSP）
        可重写为汉明距离或其他度量。

        Parameters
        ----------
        a:
            抗体 a。
        b:
            抗体 b。

        Returns
        -------
        float
            两个抗体间的距离。
        """
        return float(np.sqrt(np.sum((a - b) ** 2)))

    # ------------------------------------------------------------------
    # 算子验证
    # ------------------------------------------------------------------

    def _validate_overrides(self):
        cls = type(self)
        for name in ("init_population", "mutate"):
            if name not in cls.__dict__:
                raise NotImplementedError(
                    f"Override {cls.__name__}.{name}() method"
                )

    # ------------------------------------------------------------------
    # 浓度计算
    # ------------------------------------------------------------------

    def _concentration(self, population: np.ndarray) -> np.ndarray:
        """计算种群中每个抗体的浓度。

        浓度定义为种群中与该抗体距离小于 ``sim_thresh`` 的
        抗体数量除以种群大小。浓度越高表示该抗体周围越拥挤。

        Returns
        -------
        np.ndarray
            浓度数组，shape ``(pop_size,)``，值域 [0, 1]。
        """
        n = len(population)
        concentration = np.zeros(n)
        for i in range(n):
            count = 0
            for j in range(n):
                if self.distance(population[i], population[j]) < self.sim_thresh:
                    count += 1
            concentration[i] = count / n
        return concentration

    # ------------------------------------------------------------------
    # 激励度（motivation）计算
    # ------------------------------------------------------------------

    def _motivation(
        self, population: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算激励度。

        激励度 = alpha * 亲和度 + beta * 浓度。

        亲和度即目标函数值（越低越好），浓度惩罚拥挤个体。
        激励度越低表示抗体质量越高（低目标值 + 低拥挤度）。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(motivation, affinity)``，shape 均为 ``(pop_size,)``。
        """
        affinity = np.array(
            [self.problem.evaluate(ind) for ind in population]
        )
        concentration = self._concentration(population)
        motivation = self.alpha * affinity + self.beta * concentration
        return motivation, affinity

    # ------------------------------------------------------------------
    # 种群刷新
    # ------------------------------------------------------------------

    def _refresh(self, num: int) -> tuple[np.ndarray, np.ndarray]:
        """生成 ``num`` 个随机新抗体。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(new_pop, new_fitness)``，new_pop shape ``(num, dim)``。
        """
        new_pop = np.array(
            [self.problem.random_solution() for _ in range(num)]
        )
        new_fitness = np.array(
            [self.problem.evaluate(ind) for ind in new_pop]
        )
        return new_pop, new_fitness

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, early_stopping=None) -> OptimizeResult:
        """执行优化，返回 ``OptimizeResult``。

        Parameters
        ----------
        early_stopping:
            ``EarlyStopping`` 实例，默认 None 表示不启用早停。

        Returns
        -------
        OptimizeResult
            history 包含以下键：
            - ``"best"``  每代全局最优适应度
            - ``"avg"``   每代平均适应度
        """
        self._validate_overrides()

        population = self.init_population()
        motivation, fitness = self._motivation(population)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        history_best = [float(best_fitness)]
        history_avg = [float(np.mean(fitness))]

        pbar = tqdm(
            total=self.maxiter,
            desc="IA 优化",
            disable=not self.verbose,
        )

        for gen in range(self.maxiter):
            self.current_gen = gen
            keep_num = max(1, int(self.pop_size * (1 - self.refresh_rate)))
            sorted_indices = np.argsort(motivation)
            sorted_pop = population[sorted_indices]

            elite_pop = np.zeros_like(sorted_pop[:keep_num])
            elite_fitness = np.zeros(keep_num)

            for i in range(keep_num):
                antibody = sorted_pop[i]
                clone_fits = np.zeros(self.Ncl)
                clone_fits[0] = self.problem.evaluate(antibody)

                best_clone = antibody.copy()
                best_clone_fit = clone_fits[0]

                for j in range(1, self.Ncl):
                    clone = antibody.copy()
                    if np.random.rand() < self.mut_prob:
                        clone = self.mutate(clone)
                    clone = self.problem.clamp(clone)
                    fit = self.problem.evaluate(clone)
                    clone_fits[j] = fit
                    if fit < best_clone_fit:
                        best_clone = clone
                        best_clone_fit = fit

                elite_pop[i] = best_clone
                elite_fitness[i] = best_clone_fit

            refresh_num = self.pop_size - keep_num
            new_pop, new_fitness = self._refresh(refresh_num)

            population = np.vstack([elite_pop, new_pop]) if refresh_num > 0 else elite_pop
            fitness = np.hstack([elite_fitness, new_fitness]) if refresh_num > 0 else elite_fitness

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

            motivation, _ = self._motivation(population)

            history_best.append(float(best_fitness))
            history_avg.append(float(np.mean(fitness)))

            if self.verbose:
                pbar.set_postfix(
                    {
                        "最优值": f"{best_fitness:.4f}",
                        "平均值": f"{np.mean(fitness):.4f}",
                    }
                )
                pbar.update(1)

            history_dict = {"best": history_best, "avg": history_avg}
            if self._check_early_stop(early_stopping, history_dict):
                break

        if self.verbose:
            pbar.close()

        self.population = population

        return OptimizeResult(
            best_solution=best_solution,
            best_fitness=float(best_fitness),
            history={"best": history_best, "avg": history_avg},
        )
