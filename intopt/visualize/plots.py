import matplotlib.pyplot as plt
import numpy as np

from intopt.algorithms.base import OptimizeResult

plt.rcParams["font.sans-serif"] = ["SimHei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False


def plot_convergence(result: OptimizeResult, title: str = "收敛曲线"):
    """绘制收敛曲线。

    result.history 中 ``"best"`` 为必选；``"current"`` 或 ``"avg"`` 为可选。
    """
    best = result.history.get("best", [])
    current = result.history.get("current") or result.history.get("avg", [])

    plt.figure(figsize=(8, 4))
    plt.plot(best, "b-", label="最优适应度")
    if current:
        plt.plot(current, "r--", alpha=0.5, label="当前/平均适应度")
    plt.xlabel("迭代次数")
    plt.ylabel("适应度值")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_tsp_path(
    path: np.ndarray, coordinates: np.ndarray, length: float
):
    """绘制 TSP 路径图。

    Parameters
    ----------
    path:
        城市索引排列。
    coordinates:
        城市坐标，shape ``(n, 2)``。
    length:
        路径总长度。
    """
    ordered = coordinates[path]
    loop = np.vstack([ordered, ordered[0]])

    plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c="red", marker="o", zorder=2)
    plt.plot(loop[:, 0], loop[:, 1], "b-", linewidth=1, zorder=1)
    plt.scatter(*ordered[0], c="green", marker="*", s=200, label="起点", zorder=3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"TSP 最优路径 (总长度: {length:.2f})")
    plt.legend()
    plt.grid(True)
    plt.show()
