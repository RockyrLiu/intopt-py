from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from intopt.problems.base import Problem


@dataclass
class OptimizeResult:
    """算法运行结果。

    Attributes
    ----------
    best_solution:
        最优解向量。
    best_fitness:
        最优适应度值（越小越好）。
    history:
        收敛历史字典。键由各算法自行定义，常用键包含 ``"best"``、``"avg"``。
    """

    best_solution: np.ndarray
    best_fitness: float
    history: dict = field(default_factory=lambda: {"best": [], "avg": []})


class Optimizer(ABC):
    """算法抽象基类。

    所有优化算法均继承此类。子类只需实现 ``run`` 方法即可
    获得统一的 ``OptimizeResult`` 输出格式。

    Parameters
    ----------
    problem:
        待求解的优化问题实例。
    """

    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def run(self) -> OptimizeResult:
        """执行优化，返回统一结果对象。"""
        ...
