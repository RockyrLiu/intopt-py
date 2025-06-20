import numpy as np
import matplotlib.pyplot as plt
from test_function import TSPProblem1, TSPProblem2
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class ACO:
    """蚁群算法基类"""
    def __init__(self, m, max_iter, alpha, beta, rho, Q, verbose=True):
        self.m = m  # 蚂蚁数量
        self.max_iter = max_iter  # 最大迭代次数
        self.alpha = alpha  # 信息素重要程度参数
        self.beta = beta  # 启发式因子重要程度参数
        self.rho = rho  # 信息素蒸发系数
        self.Q = Q  # 信息素增加强度系数
        self.verbose = verbose  # 是否显示进度条
        self.history = {'best_fitness': [], 'avg_fitness': []}  # 记录历史最优值和平均值
        self.best_solution = None
        self.best_fitness = np.inf
        
    def initialize(self):
        """初始化算法状态"""
        raise NotImplementedError("Subclasses must implement initialize method")
        
    def build_solutions(self):
        """构建蚂蚁的解决方案"""
        raise NotImplementedError("Subclasses must implement build_solutions method")
        
    def evaluate_solutions(self):
        """评估解决方案的质量"""
        raise NotImplementedError("Subclasses must implement evaluate_solutions method")
        
    def update_pheromone(self):
        """更新信息素"""
        raise NotImplementedError("Subclasses must implement update_pheromone method")
        
    def update_history(self, fitness_values):
        """更新历史记录"""
        current_best = np.min(fitness_values)
        current_avg = np.mean(fitness_values)
        
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            self.best_solution = self.current_solutions[np.argmin(fitness_values)].copy()
            
        self.history['best_fitness'].append(self.best_fitness)
        self.history['avg_fitness'].append(current_avg)
        
    def iterator(self):
        """迭代优化"""
        self.initialize()
        
        iter_range = tqdm(range(self.max_iter), desc="ACO") if self.verbose else range(self.max_iter)
        
        for _ in iter_range:
            self.build_solutions()
            fitness_values = self.evaluate_solutions()
            self.update_history(fitness_values)
            self.update_pheromone()
            
            if self.verbose:
                iter_range.set_postfix({
                    '最优值': f"{self.best_fitness:.4f}",
                    '平均值': f"{self.history['avg_fitness'][-1]:.4f}"
                })
                
        return self.best_solution, self.best_fitness
    
    def run(self):
        """运行算法"""
        return self.iterator()
        
    def plot_history(self):
        """绘制适应度进化曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['best_fitness'], 'b-', linewidth=2, label='最佳适应度')
        plt.plot(self.history['avg_fitness'], 'r--', linewidth=2, label='平均适应度')
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title('适应度进化曲线')
        plt.legend()
        plt.grid(True)
        plt.show()


class ACO_TSP(ACO):
    """蚁群算法解决TSP问题"""
    def __init__(self, m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, 
                 problem=TSPProblem1(), verbose=True):
        super().__init__(m, max_iter, alpha, beta, rho, Q, verbose)
        
        self.problem = problem
        self.n = self.problem.N
        self.D = self.problem.D
        self.Eta = 1 / (self.D + np.eye(self.n) * 1e-10)
        self.Tau = np.ones((self.n, self.n))
        self.Tabu = None  # 蚂蚁的路径
        
    def initialize(self):
        """初始化算法状态"""
        self.Tabu = np.zeros((self.m, self.n), dtype=int)
        self.Tabu[:, 0] = self.initialize_ants()
        
    def initialize_ants(self):
        """初始化蚂蚁位置"""
        num_perms = int(np.ceil(self.m / self.n))
        all_perms = np.concatenate([np.random.permutation(self.n) for _ in range(num_perms)])
        return all_perms[:self.m]
    
    def build_solutions(self):
        """构建蚂蚁的路径解决方案"""
        for j in range(1, self.n):
            for i in range(self.m):
                visited = self.Tabu[i, :j]
                unvisited = np.setdiff1d(np.arange(self.n), visited)
                
                # 计算转移概率
                tau = self.Tau[visited[-1], unvisited]
                eta = self.Eta[visited[-1], unvisited]
                P = (tau ** self.alpha) * (eta ** self.beta)
                P /= P.sum()
                
                # 轮盘赌选择
                self.Tabu[i, j] = np.random.choice(unvisited, p=P)
                
    def evaluate_solutions(self):
        """评估路径长度"""
        self.current_solutions = self.Tabu
        return np.array([self.problem.calculate_path_length(path) for path in self.Tabu])
    
    def update_pheromone(self):
        """更新信息素矩阵"""
        L = self.evaluate_solutions()
        Delta_Tau = np.zeros((self.n, self.n))
        
        for i in range(self.m):
            path = self.Tabu[i]
            Delta_Tau[path[:-1], path[1:]] += self.Q / L[i]
            Delta_Tau[path[-1], path[0]] += self.Q / L[i]
        
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau


def test_aco_tsp():
    """测试ACO求解TSP问题"""
    print("="*50)
    print("蚁群算法求解TSP问题")
    print("="*50)
    
    # 创建并运行ACO_TSP
    aco = ACO_TSP(m=50, max_iter=200, alpha=1, beta=5, rho=0.1, Q=100, problem=TSPProblem2())
    best_path, best_length = aco.run()
    
    # 输出结果
    print(f"\n最优路径长度: {best_length:.2f}")
    print("最优路径:", best_path)
    
    # 绘制结果
    aco.plot_history()
    aco.problem.plot_solution(best_path, best_length)


if __name__ == "__main__":
    # 运行测试
    test_aco_tsp()
