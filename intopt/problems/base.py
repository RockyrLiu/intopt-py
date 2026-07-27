from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    """问题抽象基类。

    所有优化问题都继承此类。算法只通过这个接口与问题交互，
    不知道也无需知道问题的内部细节。目标始终是最小化。
    """

    @abstractmethod
    def random_solution(self) -> np.ndarray:
        """生成一个随机可行解。"""
        ...

    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """评估单个解的适应度，返回标量。值越小越好。"""
        ...

    @abstractmethod
    def is_feasible(self, solution: np.ndarray) -> bool:
        """检查解是否满足所有约束。"""
        ...

    @abstractmethod
    def mutate(self, solution: np.ndarray) -> np.ndarray:
        """对 ``solution`` 施加变异，生成邻域解。"""
        ...
