import numpy as np
from IA1 import IA, plot_history # 导入基础免疫算法类
from test_function import TSPProblem1


class TSP_IA(IA):
    """TSP免疫算法"""
    def __init__(self, Np=200, max_iter=1000, mut_prob=0.7, 
                 simThresh=0.2, Ncl=10, refresh_rate=0.5, verbose=True):
        self.tsp_problem = TSPProblem1()
        N = self.tsp_problem.N
        super().__init__(Np, N, max_iter, mut_prob, simThresh, Ncl, self._tsp_objective, 
                         0, N-1, refresh_rate=refresh_rate, verbose=verbose)

    def _tsp_objective(self, path):
        return self.tsp_problem.calculate_path_length(path.astype(int))

    def init_population(self):
        f = np.zeros((self.dim, self.Np), dtype=int)
        for i in range(self.Np):
            f[:, i] = np.random.permutation(self.dim)
        return f

    def mutate(self, a, gen):
        Na = np.tile(a, (self.Ncl, 1)).T
        for j in range(1, self.Ncl):
            p1, p2 = np.random.choice(self.dim, 2, replace=False)
            Na[p1, j], Na[p2, j] = Na[p2, j], Na[p1, j]
        return Na.astype(int)  # 强制返回整数

    def refreshPop(self, keep_num):
        refresh_num = self.Np - keep_num
        bf = np.zeros((self.dim, refresh_num), dtype=int)
        bFIT = np.zeros(refresh_num)
        for i in range(refresh_num):
            bf[:, i] = np.random.permutation(self.dim)
            bFIT[i] = self._tsp_objective(bf[:, i])
        return bf, bFIT

    def calculate_fitness(self, f, alpha=1, beta=1):
        affinity = self.get_affinity(f)
        fitness = affinity
        return fitness, affinity

    def run(self):
        best_solution, best_fitness = self.iterator()
        best_solution = best_solution.astype(int)  # 最终强制转换
        return best_solution, best_fitness

def main():
    """执行TSP优化的主函数""" 
    # 创建并运行TSP免疫算法，refresh_rate默认为0.5
    tsp_ia = TSP_IA(Np=200, max_iter=500)
    best_path, best_length = tsp_ia.run()
    
    # 输出结果
    print("\n最优路径:", best_path)
    print(f"最短距离:{best_length:.0f}")
    
    # 绘制结果
    tsp_ia.tsp_problem.plot_solution(best_path, best_length)
    # 绘制适应度
    plot_history(tsp_ia.history)
    
if __name__ == "__main__":
    main()