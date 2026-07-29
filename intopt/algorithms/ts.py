import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import Optimizer, OptimizeResult


class TS(Optimizer):
    """禁忌搜索算法。

    禁忌搜索 (Tabu Search) 通过维护禁忌表记录近期搜索过的
    解空间区域，避免算法在局部最优附近循环。同时引入藐视准则：
    当候选解优于历史最优解时，即便该移动在禁忌表中也直接采纳。

    每次迭代从当前解的邻域生成 ``candidate_size`` 个候选解，
    选择其中最优且未被禁忌的解（或满足藐视准则的禁忌解）作为
    下一迭代的当前解。

    **必须重写的方法**：

    - ``init_solution()`` — 生成初始解。
    - ``generate_candidates(solution)`` — 从当前解生成候选解列表，
      返回 ``list[dict]``，每个字典含 ``"solution"`` 和 ``"move"``。
    - ``update_tabu(move)`` — 将移动加入禁忌表。
    - ``is_tabu(move)`` — 检查移动是否在禁忌表中。

    **可选重写的方法**：

    - ``aspiration(candidate_fitness, best_fitness)`` —
      藐视准则，默认候选解严格优于全局最优时返回 ``True``。

    Parameters
    ----------
    problem:
        待求解的优化问题实例。
    tabu_length:
        禁忌长度，默认 10。
    candidate_size:
        每轮生成的候选解数量，默认 50。
    max_iter:
        最大迭代次数，默认 500。
    verbose:
        是否显示进度条，默认 True。
    """

    def __init__(
        self,
        problem,
        tabu_length: int = 10,
        candidate_size: int = 50,
        max_iter: int = 500,
        verbose: bool = True,
    ):
        super().__init__(problem)
        self.tabu_length = int(tabu_length)
        self.candidate_size = int(candidate_size)
        self.max_iter = int(max_iter)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # 必须重写的方法
    # ------------------------------------------------------------------

    def init_solution(self) -> np.ndarray:
        """生成初始解。**必须重写**。

        Returns
        -------
        np.ndarray
            初始解向量。
        """
        raise NotImplementedError("Override init_solution method")

    def generate_candidates(self, solution: np.ndarray) -> list[dict]:
        """从当前解生成候选解。**必须重写**。

        返回的每个字典需包含 ``"solution"`` 和 ``"move"`` 两个键。

        Parameters
        ----------
        solution:
            当前解。

        Returns
        -------
        list[dict]
            候选解列表，每个候选为一个字典，
            ``"solution"`` 为候选解向量，``"move"`` 为移动标识。
        """
        raise NotImplementedError("Override generate_candidates method")

    def update_tabu(self, move):
        """将移动加入禁忌表。**必须重写**。

        Parameters
        ----------
        move:
            用于标识禁忌的移动信息。
        """
        raise NotImplementedError("Override update_tabu method")

    def is_tabu(self, move) -> bool:
        """检查移动是否在禁忌表中。**必须重写**。

        Parameters
        ----------
        move:
            需要检查的移动信息。

        Returns
        -------
        bool
            若在禁忌表中返回 ``True``。
        """
        raise NotImplementedError("Override is_tabu method")

    # ------------------------------------------------------------------
    # 可选重写的方法
    # ------------------------------------------------------------------

    def aspiration(self, candidate_fitness: float, best_fitness: float) -> bool:
        """藐视准则。默认：候选解严格优于全局最优时返回 ``True``。

        Parameters
        ----------
        candidate_fitness:
            候选解的适应度值。
        best_fitness:
            当前全局最优适应度值。

        Returns
        -------
        bool
            若应无视禁忌直接接受则返回 ``True``。
        """
        return candidate_fitness < best_fitness

    # ------------------------------------------------------------------
    # 算子验证
    # ------------------------------------------------------------------

    def _validate_overrides(self):
        cls = type(self)
        for name in ("init_solution", "generate_candidates", "update_tabu", "is_tabu"):
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
            - ``"best"``    每轮全局最优适应度
            - ``"current"`` 每轮当前解适应度
        """
        self._validate_overrides()

        current = self.init_solution()
        current_energy = self.problem.evaluate(current)
        best = current.copy()
        best_energy = current_energy

        history_best = [float(best_energy)]
        history_current = [float(current_energy)]

        pbar = tqdm(
            total=self.max_iter,
            desc="禁忌搜索进度",
            disable=not self.verbose,
        )

        for _ in range(self.max_iter):
            candidates = self.generate_candidates(current)

            best_candidate = None
            best_candidate_energy = float("inf")

            for candidate in candidates:
                candidate_solution = candidate["solution"]
                candidate_energy = self.problem.evaluate(candidate_solution)

                if self.aspiration(candidate_energy, best_energy):
                    best_candidate = candidate
                    best_candidate_energy = candidate_energy
                    break

                if not self.is_tabu(candidate["move"]) and candidate_energy < best_candidate_energy:
                    best_candidate = candidate
                    best_candidate_energy = candidate_energy

            if best_candidate is not None:
                current = best_candidate["solution"]
                current_energy = best_candidate_energy
                self.update_tabu(best_candidate["move"])

                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy

            history_best.append(float(best_energy))
            history_current.append(float(current_energy))

            pbar.update(1)
            pbar.set_postfix({
                "最优值": f"{best_energy:.4f}",
                "当前值": f"{current_energy:.4f}",
            })

            history_dict = {"best": history_best, "current": history_current}
            if self._check_early_stop(early_stopping, history_dict):
                break

        pbar.close()

        return OptimizeResult(
            best_solution=best,
            best_fitness=float(best_energy),
            history={"best": history_best, "current": history_current},
        )
