import matplotlib.pyplot as plt
import numpy as np

from intopt.algorithms.base import OptimizeResult

plt.rcParams['font.sans-serif'] = ['SimHei', 'Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False

def plot_convergence(result: OptimizeResult, title: str = "收敛曲线"):
    """绘制最优值/平均值的收敛曲线。

    要求 ``result.history`` 中包含 ``"best"`` 和 ``"avg"`` 键。
    """
    best = result.history.get("best", [])
    avg = result.history.get("avg", [])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(best, "b-", label="最优适应度")
    if avg:
        ax.plot(avg, "r--", alpha=0.5, label="平均适应度")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("适应度值")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig


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

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c="red", marker="o", zorder=2)
    plt.plot(loop[:, 0], loop[:, 1], "b-", linewidth=1, zorder=1)
    plt.scatter(*ordered[0], c="green", marker="*", s=200, label="起点", zorder=3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"TSP 最优路径 (总长度: {length:.2f})")
    plt.legend()
    plt.grid(True)
    return fig
