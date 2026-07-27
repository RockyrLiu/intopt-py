import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from SA import TSP_SA  
from test_function import create_tsp_problem1, create_tsp_problem2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class ACO:
    """蚁群算法基类"""
    def __init__(self, tsp_problem, m=50, max_iter=200, alpha=1, beta=5, 
                 rho=0.1, Q=100, init_positions=None, verbose=True):
        """
        参数:
        tsp_problem: TSP问题实例
        m: 蚂蚁数量
        max_iter: 最大迭代次数
        alpha: 信息素重要程度参数
        beta: 启发式因子重要程度参数
        rho: 信息素蒸发系数
        Q: 信息素增加强度系数
        init_positions: 初始解列表(可选)，如果提供则使用这些解初始化部分蚂蚁
        verbose: 是否显示进度条
        """
        self.problem = tsp_problem
        self.m = m
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.init_positions = init_positions
        self.verbose = verbose
        
        # 问题相关参数
        self.n = self.problem.N  # 城市数量
        self.D = self.problem.D  # 距离矩阵
        self.Eta = 1 / (self.D + np.eye(self.n) * 1e-10)  # 启发式信息
        
        # 算法状态
        self.Tau = None  # 信息素矩阵
        self.Tabu = None  # 蚂蚁的路径
        self.current_solutions = None  # 当前解
        
        # 历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_solution': []
        }
        self.best_solution = None
        self.best_fitness = np.inf
    
    def initialize(self):
        """初始化算法状态"""
        self.Tau = np.ones((self.n, self.n))  # 初始化信息素
        self.Tabu = np.zeros((self.m, self.n), dtype=int)  # 初始化路径记录表
        self.Tabu[:, 0] = self._initialize_ants()  # 初始化蚂蚁位置
        
    def _initialize_ants(self):
        """初始化蚂蚁位置，支持使用初始解"""
        # 如果有初始解，优先使用
        if self.init_positions is not None and len(self.init_positions) > 0:
            # 确保初始解是有效的排列
            valid_positions = []
            for pos in self.init_positions:
                if len(pos) == self.n and len(set(pos)) == self.n:
                    valid_positions.append(pos)
            
            # 使用有效的初始解初始化部分蚂蚁
            num_init = min(len(valid_positions), self.m)
            init_ants = np.zeros(self.m, dtype=int)
            
            # 前num_init只蚂蚁使用初始解
            for i in range(num_init):
                init_ants[i] = valid_positions[i][0]  # 使用初始解的第一个城市
            
            # 剩余的蚂蚁随机初始化
            if num_init < self.m:
                remaining = self.m - num_init
                num_perms = int(np.ceil(remaining / self.n))
                all_perms = np.concatenate([np.random.permutation(self.n) for _ in range(num_perms)])
                init_ants[num_init:] = all_perms[:remaining]
            
            return init_ants
        else:
            # 没有初始解，完全随机初始化
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
        """更新信息素矩阵(由子类实现)"""
        raise NotImplementedError("子类必须实现update_pheromone方法")
    
    def update_history(self, fitness_values):
        """更新历史记录"""
        current_best = np.min(fitness_values)
        current_avg = np.mean(fitness_values)
        
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            self.best_solution = self.current_solutions[np.argmin(fitness_values)].copy()
            
        self.history['best_fitness'].append(self.best_fitness)
        self.history['avg_fitness'].append(current_avg)
        self.history['best_solution'].append(self.best_solution.copy())
    
    def run(self):
        """运行算法"""
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

class AS_TSP(ACO):
    """基本蚁群系统 (Ant System)"""
    def update_pheromone(self):
        """更新信息素矩阵"""
        L = self.evaluate_solutions()
        Delta_Tau = np.zeros((self.n, self.n))
        
        for i in range(self.m):
            path = self.Tabu[i]
            Delta_Tau[path[:-1], path[1:]] += self.Q / L[i]
            Delta_Tau[path[-1], path[0]] += self.Q / L[i]
        
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau

class EAS_TSP(ACO):
    """精英蚂蚁系统 (Elitist Ant System)"""
    def __init__(self, tsp_problem, m=50, max_iter=200, alpha=1, beta=5, 
                 rho=0.1, Q=100, elitist_weight=2.0, verbose=True):
        super().__init__(tsp_problem, m, max_iter, alpha, beta, rho, Q, verbose)
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

class MMAS_TSP(ACO):
    """最大最小蚁群系统 (MAX-MIN Ant System)"""
    def __init__(self, tsp_problem, m=50, max_iter=200, alpha=1, beta=5, 
                 rho=0.1, Q=100, tau_max=2.0, tau_min=0.001, verbose=True):
        super().__init__(tsp_problem, m, max_iter, alpha, beta, rho, Q, verbose)
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

class AAS_TSP(ACO):
    """自适应蚁群算法 (Adaptive Ant System)"""
    def __init__(self, tsp_problem, m=50, max_iter=200, alpha=1, beta=5, 
                 rho=0.1, Q=100, rho_min=0.01, rho_max=0.5, verbose=True):
        super().__init__(tsp_problem, m, max_iter, alpha, beta, rho, Q, verbose)
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
        ratio = best_length / avg_length
        self.rho = self.rho_min + (self.rho_max - self.rho_min) * ratio
        
        # 更新信息素
        for i in range(self.m):
            path = self.Tabu[i]
            for k in range(self.n - 1):
                Delta_Tau[path[k], path[k+1]] += self.Q / L[i]
            Delta_Tau[path[-1], path[0]] += self.Q / L[i]
        
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau

class Hybrid_ACO_SA:
    """蚁群-模拟退火混合算法"""
    def __init__(self, tsp_problem, aco_params=None, sa_params=None):
        """
        参数:
        tsp_problem: TSP问题实例
        aco_params: ACO参数字典
        sa_params: SA参数字典
        """
        self.tsp_problem = tsp_problem
        
        # 设置默认参数
        aco_params = aco_params or {
            'm': 50, 'max_iter': 100, 'alpha': 1, 'beta': 5, 
            'rho': 0.1, 'Q': 100, 'verbose': False
        }
        sa_params = sa_params or {
            'initial_temp': 200, 'final_temp': 1, 
            'cooling_rate': 0.999, 'iter_per_temp': 500,
            'verbose': False
        }
        
        # 运行ACO获得初始解
        self.aco = AS_TSP(tsp_problem, **aco_params)
        self.aco_solution, self.aco_length = self.aco.run()
        
        # 初始化SA
        self.sa = TSP_SA(tsp_problem, **sa_params)
        self.sa.init_solution = self._get_aco_solution  # 使用ACO的解作为初始解
    
    def _get_aco_solution(self):
        """获取ACO的解作为SA的初始解"""
        return self.aco_solution.tolist()
    
    def run(self):
        """运行混合算法"""
        # 运行SA优化
        best_path, best_length = self.sa.run()
        return best_path, best_length
    
    def plot_history(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 5))
        
        # ACO收敛曲线
        plt.subplot(121)
        plt.plot(self.aco.history['best_fitness'], 'b-', label='ACO最优值')
        plt.plot(self.aco.history['avg_fitness'], 'r--', label='ACO平均值')
        plt.title('蚁群算法收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('路径长度')
        plt.legend()
        plt.grid(True)
        
        # SA收敛曲线
        plt.subplot(122)
        plt.plot(self.sa.history['temperature'], self.sa.history['best_fitness'])
        plt.title('模拟退火收敛曲线')
        plt.xlabel('温度')
        plt.ylabel('路径长度')
        plt.gca().invert_xaxis()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def aco_tsp(tsp_problem, aco_type='AS', m=50, max_iter=200, alpha=1, beta=5, 
                     rho=0.1, Q=100, init_positions=None, verbose=True, plot=True, **kwargs):
    """
    ACO算法接口(TSP问题)
    参数:
    tsp_problem: TSP问题实例
    aco_type: ACO类型 ('AS', 'EAS', 'MMAS', 'AAS')
    m: 蚂蚁数量
    max_iter: 最大迭代次数
    alpha: 信息素重要程度
    beta: 启发式因子重要程度
    rho: 信息素蒸发系数
    Q: 信息素增量系数
    init_positions: 初始解列表(可选)
    verbose: 是否显示进度
    plot: 是否绘制结果
    kwargs: 各算法特有参数
    """
    # 创建ACO实例
    if aco_type == 'AS':
        aco = AS_TSP(tsp_problem, m=m, max_iter=max_iter, alpha=alpha, beta=beta, 
                     rho=rho, Q=Q, init_positions=init_positions, verbose=verbose)
    elif aco_type == 'EAS':
        aco = EAS_TSP(tsp_problem, m=m, max_iter=max_iter, alpha=alpha, beta=beta, 
                      rho=rho, Q=Q, elitist_weight=kwargs.get('elitist_weight', 2.0), 
                      init_positions=init_positions, verbose=verbose)
    elif aco_type == 'MMAS':
        aco = MMAS_TSP(tsp_problem, m=m, max_iter=max_iter, alpha=alpha, beta=beta, 
                       rho=rho, Q=Q, tau_max=kwargs.get('tau_max', 2.0), 
                       tau_min=kwargs.get('tau_min', 0.001), 
                       init_positions=init_positions, verbose=verbose)
    elif aco_type == 'AAS':
        aco = AAS_TSP(tsp_problem, m=m, max_iter=max_iter, alpha=alpha, beta=beta, 
                      rho=rho, Q=Q, rho_min=kwargs.get('rho_min', 0.01), 
                      rho_max=kwargs.get('rho_max', 0.5), 
                      init_positions=init_positions, verbose=verbose)
    else:
        raise ValueError(f"未知的ACO类型: {aco_type}")
    
    # 运行算法
    best_path, best_length = aco.run()
    
    if plot:
        # 绘制收敛曲线
        aco.plot_history()
        
        # 绘制最优路径
        tsp_problem.plot_solution(best_path, best_length)
    
    return best_path, best_length, aco.history

def acosa_tsp(tsp_problem, aco_params=None, sa_params=None, plot=True):
    """
    参数:
        tsp_problem: TSP问题实例
        aco_params: ACO参数字典 (可选)
        sa_params: SA参数字典 (可选)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_path: 最优路径
        best_length: 最优路径长度
    """
    # 创建混合算法实例
    hybrid = Hybrid_ACO_SA(tsp_problem, aco_params=aco_params, sa_params=sa_params)
    
    # 运行算法
    best_path, best_length = hybrid.run()
    
    if plot:
        # 绘制收敛曲线
        hybrid.plot_history()
        
        # 绘制最优路径
        tsp_problem.plot_solution(best_path, best_length)
    
    return best_path, best_length

problem1 = create_tsp_problem1()

def main():
    """测试函数"""
    # 生成一些初始解
    problem = problem1
    init_solutions = [
        list(np.random.permutation(problem.N)),
        list(np.random.permutation(problem.N)),
        list(np.random.permutation(problem.N))
    ]
    
    # 测试带初始解的ACO
    print("测试带初始解的蚁群算法:")
    best_path, best_length, history = aco_tsp(
        problem, 
        aco_type='AS',
        m=50,
        max_iter=200,
        init_positions=init_solutions,
        verbose=True
    )
    print(f"最优路径长度: {best_length:.2f}")
    
    # 测试混合算法
    print("\n测试混合算法:")
    best_path, best_length = acosa_tsp(
        problem,
        aco_params={'m': 120, 'max_iter': 100, 'init_positions': init_solutions},
        sa_params={'initial_temp': 200, 'cooling_rate': 0.999, 'iter_per_temp': 500}
    )
    print(f"最优路径长度: {best_length:.2f}")


if __name__ == "__main__":
    main()