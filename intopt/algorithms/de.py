import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import Optimizer, OptimizeResult


class DE(Optimizer):
    """差分进化算法 (Differential Evolution)。

    DE 的核心流程：差分变异 → 交叉 → 一对一选择。

    **必须重写的方法**：

    - ``init_population()`` — 初始化种群。连续型用 `initRandom`_，
      混沌连续型用 `initChaosContinuous`_。
    - ``mutate(solution)`` — 差分变异，返回单个供体向量。
      可通过 ``self.population`` 获取当前种群，通过
      ``self.fitness`` 获取适应度，结合 ``self.F`` 实现不同策略。
    - ``crossover(target, donor)`` — 二项式交叉，返回试验向量。
      通过 ``self.CR`` 控制交叉概率。

    Parameters
    ----------
    problem:
        待求解的优化问题实例。
    pop_size:
        种群大小，默认 50。
    maxiter:
        最大迭代次数，默认 100。
    F:
        缩放因子，默认 0.5。
    CR:
        交叉概率，默认 0.7。
    verbose:
        是否显示进度条，默认 True。
    """

    def __init__(
        self,
        problem,
        pop_size: int = 50,
        maxiter: int = 100,
        F: float = 0.5,
        CR: float = 0.7,
        verbose: bool = True,
    ):
        super().__init__(problem)
        self.pop_size = int(pop_size)
        self.maxiter = int(maxiter)
        self.F = float(F)
        self.CR = float(CR)
        self.verbose = verbose
        self.population: np.ndarray | None = None
        self.fitness: np.ndarray | None = None
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
        """差分变异。**必须重写**。

        对单个个体执行差分变异，返回供体向量。
        ``self.population`` 和 ``self.fitness`` 包含当前种群信息。

        Parameters
        ----------
        solution:
            当前目标向量。

        Returns
        -------
        np.ndarray
            供体向量。
        """
        raise NotImplementedError("Override mutate method")

    def crossover(
        self, target: np.ndarray, donor: np.ndarray
    ) -> np.ndarray:
        """二项式交叉。**必须重写**。

        Parameters
        ----------
        target:
            目标向量。
        donor:
            供体向量。

        Returns
        -------
        np.ndarray
            试验向量。
        """
        raise NotImplementedError("Override crossover method")

    # ------------------------------------------------------------------
    # 算子验证
    # ------------------------------------------------------------------

    def _validate_overrides(self):
        cls = type(self)
        for name in ("init_population", "mutate", "crossover"):
            if name not in cls.__dict__:
                raise NotImplementedError(
                    f"Override {cls.__name__}.{name}() method"
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

        self.population = self.init_population()
        self.fitness = np.array(
            [self.problem.evaluate(ind) for ind in self.population]
        )

        best_idx = np.argmin(self.fitness)
        best_solution = self.population[best_idx].copy()
        best_fitness = self.fitness[best_idx]

        history_best = [float(best_fitness)]
        history_avg = [float(np.mean(self.fitness))]

        pbar = tqdm(
            total=self.maxiter,
            desc="DE 优化",
            disable=not self.verbose,
        )

        for gen in range(self.maxiter):
            self.current_gen = gen

            new_pop = self.population.copy()
            new_fit = self.fitness.copy()

            for i in range(self.pop_size):
                donor = self.mutate(new_pop[i])
                trial = self.crossover(new_pop[i], donor)
                trial = self.problem.clamp(trial)
                trial_fit = self.problem.evaluate(trial)
                if trial_fit < new_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit

            self.population = new_pop
            self.fitness = new_fit

            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < best_fitness:
                best_solution = self.population[current_best_idx].copy()
                best_fitness = self.fitness[current_best_idx]

            history_best.append(float(best_fitness))
            history_avg.append(float(np.mean(self.fitness)))

            if self.verbose:
                pbar.set_postfix(
                    {
                        "最优值": f"{best_fitness:.4f}",
                        "平均值": f"{np.mean(self.fitness):.4f}",
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
