import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import Optimizer, OptimizeResult
from intopt.operators.crossover import cxArithmetic
from intopt.operators.selection import selTournament


class GA(Optimizer):
    """遗传算法。

    用户可通过继承重写 ``init_population``、``crossover``、``select``
    方法来自定义遗传操作，无需修改算法主循环。

    **默认算子**：

    - 交叉：`cxArithmetic`_（算术交叉），适用于连续型。排列型问题需重写为
      `cxOrdered`_ 或 `cxPartialyMatched`_。
    - 选择：`selTournament`_（锦标赛选择，tournsize=3）。
    - 变异：通过 ``problem.mutate()`` 委托给问题实例（如
      ContinuousProblem 默认用 `mutGaussian`_，TSPProblem 用 `mutSwap`_），
      GA 不直接选择变异算子。

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
    init_solutions:
        自定义初始种群 (pop_size, *ind_shape)，默认 None 表示随机生成。
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
        init_solutions: np.ndarray | None = None,
        verbose: bool = True,
    ):
        super().__init__(problem)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elitism_ratio = float(elitism_ratio)
        self.elite_size = int(elitism_ratio * pop_size)
        self.init_solutions = (
            np.array(init_solutions, copy=True)
            if init_solutions is not None
            else None
        )
        self.verbose = verbose

    # ------------------------------------------------------------------
    # 可被子类重写的方法
    # ------------------------------------------------------------------

    def init_population(self) -> np.ndarray:
        """初始化种群。"""
        if self.init_solutions is not None:
            if len(self.init_solutions) != self.pop_size:
                raise ValueError(
                    f"init_solutions 长度 ({len(self.init_solutions)}) "
                    f"与 pop_size ({self.pop_size}) 不匹配"
                )
            return self.init_solutions.copy()
        return np.array(
            [self.problem.random_solution() for _ in range(self.pop_size)]
        )

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray):
        """交叉操作。默认使用算术交叉。"""
        return cxArithmetic(parent1, parent2)

    def select(
        self, population: np.ndarray, fitness: np.ndarray, k: int | None = None
    ) -> np.ndarray:
        """选择操作。默认使用锦标赛选择。

        Parameters
        ----------
        k:
            选择的个体数量，默认等于 ``pop_size``。
        """
        if k is None:
            k = self.pop_size
        return selTournament(population, fitness, k)

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
                    offspring[i] = self.problem.clamp(
                        self.problem.mutate(offspring[i])
                    )

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
