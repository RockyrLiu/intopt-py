import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import Optimizer, OptimizeResult


class PSO(Optimizer):
    """粒子群优化算法。

    用户**必须**通过继承重写所有算子方法。

    **必须重写的方法**：

    - ``init_population()`` — 初始化粒子位置。连续型可用
      `initRandom`_ 或 `initChaosContinuous`_。
    - ``update_velocity()`` — 速度更新。连续型用 `velStd`_。

    **可选重写的方法**：

    - ``update_position(X, V)`` — 位置更新。默认
      ``self.clamp_position(X + V)``，离散型可重写。
    - ``clamp_position(X)`` — 位置钳制。默认委托
      ``problem.clamp(X)``。
    - ``clamp_velocity(V)`` — 速度钳制。默认截断到
      ``[-v_max, v_max]``。

    Parameters
    ----------
    problem:
        待求解的优化问题实例。
    pop_size:
        粒子数量，默认 30。
    maxiter:
        最大迭代次数，默认 100。
    w:
        初始惯性权重，默认 0.8。
    w_min:
        最小惯性权重（线性衰减终点），默认 0.4。
    c1:
        个体学习因子，默认 2。
    c2:
        社会学习因子，默认 2。
    v_max:
        最大速度限制，默认 2。
    verbose:
        是否显示进度条，默认 True。
    """

    def __init__(
        self,
        problem,
        pop_size: int = 30,
        maxiter: int = 100,
        w: float = 0.8,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        v_max: float = 2.0,
        verbose: bool = True,
    ):
        super().__init__(problem)
        self.pop_size = int(pop_size)
        self.maxiter = int(maxiter)
        self.w = float(w)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.v_max = float(v_max)
        self.verbose = verbose

        self.X: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.pbest: np.ndarray | None = None
        self.p_fit: np.ndarray | None = None
        self.gbest: np.ndarray | None = None
        self.fit: float | None = None

    # ------------------------------------------------------------------
    # 必须重写的方法
    # ------------------------------------------------------------------

    def init_population(self) -> np.ndarray:
        """初始化粒子位置。**必须重写**。

        Returns
        -------
        np.ndarray
            粒子位置，shape ``(pop_size, dim)``。
        """
        raise NotImplementedError("Override init_population method")

    def update_velocity(self) -> np.ndarray:
        """速度更新。**必须重写**。

        Returns
        -------
        np.ndarray
            更新后的速度，shape ``(pop_size, dim)``。
        """
        raise NotImplementedError("Override update_velocity method")

    # ------------------------------------------------------------------
    # 可选重写的方法
    # ------------------------------------------------------------------

    def update_position(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
        """位置更新。默认 ``clamp_position(X + V)``。

        Parameters
        ----------
        positions:
            当前位置，shape ``(pop_size, dim)``。
        velocities:
            当前速度，shape ``(pop_size, dim)``。

        Returns
        -------
        np.ndarray
            更新后的位置，shape ``(pop_size, dim)``。
        """
        return self.clamp_position(positions + velocities)

    def clamp_position(self, positions: np.ndarray) -> np.ndarray:
        """位置钳制。默认委托 ``problem.clamp()``。

        Parameters
        ----------
        positions:
            待钳制的位置，shape ``(pop_size, dim)``。

        Returns
        -------
        np.ndarray
            钳制后的位置，shape ``(pop_size, dim)``。
        """
        return self.problem.clamp(positions)

    def clamp_velocity(self, velocities: np.ndarray) -> np.ndarray:
        """速度钳制。默认截断到 ``[-v_max, v_max]``。

        Parameters
        ----------
        velocities:
            待钳制的速度，shape ``(pop_size, dim)``。

        Returns
        -------
        np.ndarray
            钳制后的速度，shape ``(pop_size, dim)``。
        """
        return np.clip(velocities, -self.v_max, self.v_max)

    # ------------------------------------------------------------------
    # 算子验证
    # ------------------------------------------------------------------

    def _validate_overrides(self):
        cls = type(self)
        for name in ("init_population", "update_velocity"):
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
            desc="PSO 优化",
            disable=not self.verbose,
        )

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

        best_solution = self.gbest[0]
        return OptimizeResult(
            best_solution=best_solution,
            best_fitness=float(self.fit),
            history={"best": history_best, "avg": history_avg},
        )
