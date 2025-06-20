import numpy as np
import matplotlib.pyplot as plt
from ACO1 import ACO_TSP
from test_function import TSPProblem1, TSPProblem2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class EAS_TSP(ACO_TSP):
    """精英蚂蚁系统 (Elitist Ant System)"""
    def __init__(self, m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, 
                 elitist_weight=2.0, problem=TSPProblem1(), verbose=True):
        super().__init__(m, max_iter, alpha, beta, rho, Q, problem, verbose)
        self.elitist_weight = elitist_weight  # 精英蚂蚁的权重
        
    def update_pheromone(self):
        """更新信息素矩阵（精英蚂蚁系统）"""
        L = self.evaluate_solutions()
        Delta_Tau = np.zeros((self.n, self.n))
        
        # 普通蚂蚁的信息素更新
        for i in range(self.m):
            path = self.Tabu[i]
            for k in range(self.n - 1):
                Delta_Tau[path[k], path[k+1]] += self.Q / L[i]
            Delta_Tau[path[-1], path[0]] += self.Q / L[i]
        
        # 精英蚂蚁的信息素更新（额外增加）
        best_idx = np.argmin(L)
        best_path = self.Tabu[best_idx]
        best_length = L[best_idx]
        
        # 计算精英蚂蚁的信息素增量
        elite_delta = np.zeros((self.n, self.n))
        for k in range(self.n - 1):
            elite_delta[best_path[k], best_path[k+1]] += self.elitist_weight * self.Q / best_length
        elite_delta[best_path[-1], best_path[0]] += self.elitist_weight * self.Q / best_length
        
        # 合并更新
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau + elite_delta


class MMAS_TSP(ACO_TSP):
    """最大最小蚁群系统 (MAX-MIN Ant System)"""
    def __init__(self, m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, 
                 tau_max=2.0, tau_min=0.001, problem=TSPProblem1(), verbose=True):
        super().__init__(m, max_iter, alpha, beta, rho, Q, problem, verbose)
        self.tau_max = tau_max  # 信息素上限
        self.tau_min = tau_min  # 信息素下限
        self.global_best_solution = None  # 全局最优解
        self.global_best_fitness = np.inf  # 全局最优适应度
        
    def initialize(self):
        """初始化算法状态"""
        super().initialize()
        # 初始化信息素为上限值
        self.Tau = np.ones((self.n, self.n)) * self.tau_max
        self.global_best_solution = None
        self.global_best_fitness = np.inf
        
    def update_pheromone(self):
        """更新信息素矩阵（最大最小蚁群系统）"""
        L = self.evaluate_solutions()
        
        # 更新全局最优解
        current_best_idx = np.argmin(L)
        current_best_fitness = L[current_best_idx]
        
        if current_best_fitness < self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best_solution = self.Tabu[current_best_idx].copy()
        
        # 只使用全局最优解更新信息素
        Delta_Tau = np.zeros((self.n, self.n))
        best_path = self.global_best_solution
        
        # 更新最优路径上的信息素
        for k in range(self.n - 1):
            Delta_Tau[best_path[k], best_path[k+1]] += self.Q / self.global_best_fitness
        Delta_Tau[best_path[-1], best_path[0]] += self.Q / self.global_best_fitness
        
        # 信息素蒸发和更新
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau
        
        # 应用信息素上下限限制
        self.Tau = np.clip(self.Tau, self.tau_min, self.tau_max)
        
        # 更新当前最优解（用于历史记录）
        self.best_fitness = self.global_best_fitness
        self.best_solution = self.global_best_solution


class AAS_TSP(ACO_TSP):
    """自适应蚁群算法 (Adaptive Ant System)"""
    def __init__(self, m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, 
                 rho_min=0.01, rho_max=0.5, problem=TSPProblem1(), verbose=True):
        super().__init__(m, max_iter, alpha, beta, rho, Q, problem, verbose)
        self.rho_min = rho_min  # 最小挥发系数
        self.rho_max = rho_max  # 最大挥发系数
        self.initial_rho = rho  # 初始挥发系数
        
    def update_pheromone(self):
        """更新信息素矩阵（自适应蚁群算法）"""
        L = self.evaluate_solutions()
        Delta_Tau = np.zeros((self.n, self.n))
        
        # 计算当前迭代的平均路径长度和最优路径长度
        avg_length = np.mean(L)
        best_length = np.min(L)
        
        # 自适应调整rho：路径质量越好，挥发率越低
        # 当最优路径接近平均路径时，增加挥发率以增强探索
        # 当最优路径远好于平均路径时，降低挥发率以保留好路径
        ratio = best_length / avg_length
        self.rho = self.rho_min + (self.rho_max - self.rho_min) * ratio
        
        # 更新信息素
        for i in range(self.m):
            path = self.Tabu[i]
            for k in range(self.n - 1):
                Delta_Tau[path[k], path[k+1]] += self.Q / L[i]
            Delta_Tau[path[-1], path[0]] += self.Q / L[i]
        
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau


def test_aco_tsp():
    """测试ACO求解TSP问题"""
    PROBLEM = TSPProblem2()


    print("="*50)
    print("基础蚁群算法求解TSP问题")
    print("="*50)
    aco = ACO_TSP(m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, problem=PROBLEM)
    best_path, best_length = aco.run()
    print(f"最优路径长度: {best_length:.2f}")
    aco.plot_history()
    aco.problem.plot_solution(best_path, best_length)
    
    print("\n" + "="*50)
    print("精英蚂蚁系统求解TSP问题")
    print("="*50)
    eas = EAS_TSP(m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, elitist_weight=2.0, problem=PROBLEM)
    best_path, best_length = eas.run()
    print(f"最优路径长度: {best_length:.2f}")
    eas.plot_history()
    eas.problem.plot_solution(best_path, best_length)
    
    print("\n" + "="*50)
    print("最大最小蚁群系统求解TSP问题")
    print("="*50)
    mmas = MMAS_TSP(m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, 
                    tau_max=2.0, tau_min=0.001, problem=PROBLEM)
    best_path, best_length = mmas.run()
    print(f"最优路径长度: {best_length:.2f}")
    mmas.plot_history()
    mmas.problem.plot_solution(best_path, best_length)
    
    print("\n" + "="*50)
    print("自适应蚁群算法求解TSP问题")
    print("="*50)
    aas = AAS_TSP(m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, 
                 rho_min=0.01, rho_max=0.5, problem=PROBLEM)
    best_path, best_length = aas.run()
    print(f"最优路径长度: {best_length:.2f}")
    aas.plot_history()
    aas.problem.plot_solution(best_path, best_length)


if __name__ == "__main__":
    # 运行测试
    test_aco_tsp()