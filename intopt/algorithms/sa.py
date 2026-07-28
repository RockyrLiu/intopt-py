import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import Optimizer, OptimizeResult
from intopt.algorithms.utils import metropolis


class SA(Optimizer):
    """模拟退火算法。

    模拟退火 (Simulated Annealing) 模拟固体退火过程：高温时分子剧烈运动
    (接受差解的概率高)，随温度降低运动减缓 (接受差解的概率降低)，最终凝固
    在最低能量状态 (全局最优)。

    **必须重写的方法**：

    - ``mutate(solution)`` — 变异操作。连续型用 `mutGaussian`_，
      排列型用 `mutSwap`_，二值型用 `mutFlip`_。

    Parameters
    ----------
    problem:
        待求解的优化问题实例。
    init_solution:
        自定义初始解，默认 None 表示随机生成。
    initial_temp:
        初始温度，默认 100。
    final_temp:
        终止温度，默认 1e-3。
    cooling_rate:
        降温系数 (0, 1)，默认 0.99。
    iter_per_temp:
        每个温度下的迭代次数，默认 100。
    verbose:
        是否显示进度条，默认 True。
    """

    def __init__(
        self,
        problem,
        init_solution: np.ndarray | None = None,
        initial_temp: float = 100,
        final_temp: float = 1e-3,
        cooling_rate: float = 0.99,
        iter_per_temp: int = 100,
        verbose: bool = True,
    ):
        super().__init__(problem)
        self.init_solution = (
            np.array(init_solution, copy=True) if init_solution is not None else None
        )
        self.T0 = float(initial_temp)
        self.Tf = float(final_temp)
        self.alpha = float(cooling_rate)
        self.iter_per_temp = int(iter_per_temp)
        self.verbose = verbose

    def mutate(self, solution: np.ndarray) -> np.ndarray:
        """变异操作。**必须重写**。"""
        raise NotImplementedError("Override mutate method")

    # ------------------------------------------------------------------
    # 算子验证
    # ------------------------------------------------------------------

    def _validate_overrides(self):
        cls = type(self)
        if "mutate" not in cls.__dict__:
            raise NotImplementedError(
                f"Override {cls.__name__}.mutate()"
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
            - ``"best"``    每个温度结束时全局最优适应度
            - ``"current"`` 每个温度结束时当前解适应度
        """
        self._validate_overrides()

        if self.init_solution is not None:
            current = self.init_solution.copy()
        else:
            current = self.problem.random_solution()

        current_energy = self.problem.evaluate(current)
        best = current.copy()
        best_energy = current_energy

        history_best = [float(best_energy)]
        history_current = [float(current_energy)]

        T = self.T0
        total_steps = int(np.floor(np.log(self.Tf / self.T0) / np.log(self.alpha)))
        pbar = tqdm(total=total_steps, desc="退火进度", disable=not self.verbose)
        while T > self.Tf:
            for _ in range(self.iter_per_temp):
                candidate = self.mutate(current)
                candidate_energy = self.problem.evaluate(candidate)

                if metropolis(candidate_energy, current_energy, T):
                    current = candidate
                    current_energy = candidate_energy
                    if candidate_energy < best_energy:
                        best = candidate.copy()
                        best_energy = candidate_energy

            history_best.append(float(best_energy))
            history_current.append(float(current_energy))
            T *= self.alpha
            pbar.update(1)
            pbar.set_postfix({
                "温度": f"{T:.2e}",
                "当前值": f"{current_energy:.4f}",
                "最优值": f"{best_energy:.4f}",
            })
            if self._check_early_stop(
                early_stopping,
                {"best": history_best, "current": history_current},
            ):
                break
        pbar.close()

        return OptimizeResult(
            best_solution=best,
            best_fitness=float(best_energy),
            history={"best": history_best, "current": history_current},
        )
