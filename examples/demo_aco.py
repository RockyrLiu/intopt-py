"""演示蚁群算法 (ACO) 在 TSP 上的用法（默认实现，无需覆写）。"""
import numpy as np

from intopt.algorithms.aco import ACO
from intopt.problems import TSPProblem


def demo():
    sj0 = np.loadtxt("tests/data/tsp2_data.txt")
    x = sj0[:, 0:8:2].flatten()
    y = sj0[:, 1:8:2].flatten()
    coords = np.column_stack((x, y))
    coords = np.vstack(([70, 40], coords))

    problem = TSPProblem(coords)
    aco = ACO(problem, m=30, max_iter=50, strategy="AS")
    result = aco.run()
    print(f"最短路径: {result.best_fitness:.2f}")


if __name__ == "__main__":
    demo()
