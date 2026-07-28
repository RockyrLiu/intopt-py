import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import Optimizer, OptimizeResult


class GA(Optimizer):
    """遗传算法。

    用户**必须**通过继承重写所有算子方法。

    **必须重写的方法**：

    - ``init_population()`` — 初始化种群。连续型用 `initRandom`_，
      混沌连续型用 `initChaosContinuous`_，排列型混沌用
      `initChaosPermutation`_，自定义用 `initCustom`_。
    - ``crossover(p1, p2)`` — 交叉操作。连续型用 `cxArithmetic`_ 或
      `cxSimulatedBinary`_，排列型用 `cxOrdered`_ 或
      `cxPartialyMatched`_，通用向量用 `cxOnePoint`_/`cxTwoPoint`_，
      离散/二值用 `cxUniform`_。
    - ``select(population, fitness, k)`` — 选择操作。可用
      `selTournament`_、`selRoulette`_、`selBest`_。
    - ``mutate(solution)`` — 变异操作。连续型用 `mutGaussian`_，
      排列型用 `mutSwap`_，二值型用 `mutFlip`_。

    Parameters
    ----------
    problem:
        待求解的优化问题实例。
    pop_size:
        种群大小，默认 50。
    generations:
        进化代数，默认 100。
    crossover_rate:
        交叉概率，默认 0.8。
    mutation_rate:
        变异概率，默认 0.2。
    elitism_ratio:
        精英保留比例，默认 0 表示不保留精英。
    verbose:
        是否显示进度条，默认 True。
    """

    def __init__(
        self,
        problem,
        pop_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism_ratio: float = 0.0,
        verbose: bool = True,
    ):
        super().__init__(problem)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elitism_ratio = float(elitism_ratio)
        self.elite_size = int(elitism_ratio * pop_size)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # 必须重写的方法
    # ------------------------------------------------------------------

    def init_population(self) -> np.ndarray:
        """初始化种群。**必须重写**。"""
        raise NotImplementedError("请重写 init_population 方法")

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray):
        """交叉操作。**必须重写**。"""
        raise NotImplementedError("请重写 crossover 方法")

    def mutate(self, solution: np.ndarray) -> np.ndarray:
        """变异操作。**必须重写**。"""
        raise NotImplementedError("请重写 mutate 方法")

    def select(
        self, population: np.ndarray, fitness: np.ndarray, k: int
    ) -> np.ndarray:
        """选择操作。**必须重写**。

        Parameters
        ----------
        k:
            需选择的个体数量。
        """
        raise NotImplementedError("请重写 select 方法")

    # ------------------------------------------------------------------
    # 算子验证
    # ------------------------------------------------------------------

    def _validate_overrides(self):
        cls = type(self)
        for name in ("init_population", "crossover", "select", "mutate"):
            if name not in cls.__dict__:
                raise NotImplementedError(
                    f"请重写 {cls.__name__}.{name}() 方法"
                )

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
        fitness = np.array([self.problem.evaluate(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        history_best = [float(best_fitness)]
        history_avg = [float(np.mean(fitness))]

        pbar = tqdm(
            total=self.generations, desc="GA 进化", disable=not self.verbose
        )

        for _gen in range(self.generations):
            old_elite_pop = []
            old_elite_fit = np.array([])
            if self.elite_size > 0:
                elite_indices = np.argsort(fitness)[: self.elite_size]
                old_elite_pop = population[elite_indices].copy()
                old_elite_fit = fitness[elite_indices].copy()

            selected = self.select(population, fitness, self.pop_size)
            np.random.shuffle(selected)

            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 >= len(selected):
                    offspring.append(self.problem.clamp(selected[i].copy()))
                    break
                p1, p2 = selected[i], selected[i + 1]
                if np.random.rand() < self.crossover_rate:
                    c1, c2 = self.crossover(p1, p2)
                    c1 = self.problem.clamp(c1)
                    c2 = self.problem.clamp(c2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                offspring.append(c1)
                offspring.append(c2)

            for i in range(len(offspring)):
                if np.random.rand() < self.mutation_rate:
                    offspring[i] = self.mutate(offspring[i])

            population = np.array(offspring[: self.pop_size])
            fitness = np.array(
                [self.problem.evaluate(ind) for ind in population]
            )

            if self.elite_size > 0:
                worst_indices = np.argsort(fitness)[-self.elite_size:]
                population[worst_indices] = old_elite_pop
                fitness[worst_indices] = old_elite_fit

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

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

        return OptimizeResult(
            best_solution=best_solution,
            best_fitness=float(best_fitness),
            history={"best": history_best, "avg": history_avg},
        )
