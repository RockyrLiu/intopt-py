import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from test_function import TSPProblem2

class ACO_TSP:
    """蚁群算法解决TSP问题"""
    def __init__(self, tsp, num_ants=50, alpha=1, beta=5, rho=0.1, Q=100, 
                 max_iter=200, elite_weight=2.0, tau_min=0.1, tau_max=10.0,
                 verbose=True):
        """
        初始化蚁群算法参数
        
        参数:
        tsp: TSP问题实例
        num_ants: 蚂蚁数量
        alpha: 信息素重要程度
        beta: 启发式因子重要程度
        rho: 信息素挥发系数
        Q: 信息素强度
        max_iter: 最大迭代次数
        elite_weight: 精英蚂蚁信息素增强权重
        tau_min: 信息素最小值
        tau_max: 信息素最大值
        verbose: 是否显示进度条
        """
        self.tsp = tsp
        self.n = tsp.N  # 城市数量
        self.D = tsp.D  # 距离矩阵
        
        # 算法参数
        self.m = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_iter = max_iter
        self.elite_weight = elite_weight
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.verbose = verbose
        
        # 预计算启发因子(距离倒数)
        self.eta = 1 / (self.D + np.eye(self.n) * 1e10)  # 对角线设为接近0
        self.eta = np.where(np.isfinite(self.eta), self.eta, 0)  # 处理无穷大值
        
        # 初始化信息素矩阵
        self.tau = np.ones((self.n, self.n)) * tau_max
        
        # 结果记录
        self.best_path = None
        self.best_length = np.inf
        self.history = {'best_length': [], 'avg_length': []}
    
    def initialize_ants(self):
        """初始化蚂蚁位置"""
        tabu = np.zeros((self.m, self.n), dtype=int)
        # 使用向量化操作随机放置蚂蚁
        tabu[:, 0] = np.random.choice(self.n, self.m, replace=True)
        return tabu
    
    def construct_path_for_ant(self, ant_idx, tabu):
        """为单只蚂蚁构建路径"""
        visited = set([tabu[ant_idx, 0]])  # 使用集合存储已访问城市
        for step in range(1, self.n):
            current_city = tabu[ant_idx, step-1]
            
            # 获取未访问城市
            unvisited = np.array([city for city in range(self.n) if city not in visited])
            
            # 计算转移概率
            prob = (self.tau[current_city, unvisited] ** self.alpha) * \
                   (self.eta[current_city, unvisited] ** self.beta)
            
            # 避免除以零
            total_prob = np.sum(prob)
            if total_prob > 0:
                prob /= total_prob
            else:
                # 如果所有概率为零，则均匀分布
                prob = np.ones_like(prob) / len(prob)
            
            # 轮盘赌选择下一个城市
            next_city = np.random.choice(unvisited, p=prob)
            tabu[ant_idx, step] = next_city
            visited.add(next_city)
        
        return tabu[ant_idx]
    
    def construct_solutions(self, tabu):
        """构建所有蚂蚁的路径"""
        # 使用向量化操作构建路径
        for ant in range(self.m):
            tabu[ant] = self.construct_path_for_ant(ant, tabu)
        return tabu
    
    def calculate_path_lengths(self, tabu):
        """向量化计算所有路径长度"""
        # 创建路径对矩阵
        from_indices = tabu
        to_indices = np.roll(tabu, -1, axis=1)
        
        # 使用向量化索引获取所有距离
        distances = self.D[from_indices, to_indices]
        
        # 计算每条路径的总长度
        return np.sum(distances, axis=1)
    
    def update_pheromone(self, tabu, lengths):
        """更新信息素 - 使用向量化操作"""
        # 信息素挥发
        self.tau *= (1 - self.rho)
        
        # 初始化信息素增量矩阵
        delta_tau = np.zeros((self.n, self.n))
        
        # 计算所有蚂蚁的信息素贡献
        for ant in range(self.m):
            path = tabu[ant]
            # 创建路径边索引
            from_cities = path
            to_cities = np.roll(path, -1)
            
            # 为路径上的每条边添加信息素
            np.add.at(delta_tau, (from_cities, to_cities), self.Q / lengths[ant])
        
        # 精英蚂蚁额外增强
        if self.best_path is not None:
            from_cities = self.best_path
            to_cities = np.roll(self.best_path, -1)
            np.add.at(delta_tau, (from_cities, to_cities), self.elite_weight * self.Q / self.best_length)
        
        # 应用信息素增量
        self.tau += delta_tau
        
        # 信息素边界限制
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)
    
    def run(self):
        """运行蚁群算法"""
        # 初始化蚂蚁
        tabu = self.initialize_ants()
        
        # 设置迭代范围（带或不带进度条）
        iter_range = range(self.max_iter)
        if self.verbose:
            pbar = tqdm(iter_range, desc="蚁群算法优化")
        
        # 主迭代循环
        for iter in iter_range:
            # 构建路径
            tabu = self.construct_solutions(tabu)
            
            # 保留上代最优路径
            if self.best_path is not None:
                tabu[0] = self.best_path
            
            # 计算路径长度
            lengths = self.calculate_path_lengths(tabu)
            
            # 更新最优解
            min_idx = np.argmin(lengths)
            if lengths[min_idx] < self.best_length:
                self.best_length = lengths[min_idx]
                self.best_path = tabu[min_idx].copy()
            
            # 记录历史
            self.history['best_length'].append(self.best_length)
            self.history['avg_length'].append(np.mean(lengths))
            
            # 更新信息素
            self.update_pheromone(tabu, lengths)
            
            # 更新进度条
            if self.verbose:
                pbar.set_postfix({'最优长度': f"{self.best_length:.2f}"})
                pbar.update(1)
        
        # 关闭进度条
        if self.verbose:
            pbar.close()
        
        return self.best_path, self.best_length
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['best_length'], 'b-', linewidth=2, label='最优路径长度')
        plt.plot(self.history['avg_length'], 'g--', linewidth=1, label='平均路径长度')
        plt.xlabel('迭代次数')
        plt.ylabel('路径长度')
        plt.title('蚁群算法收敛曲线')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # 初始化TSP问题
    tsp = TSPProblem2()
    
    # 设置算法参数
    aco = ACO_TSP(
        tsp,
        num_ants=200,
        alpha=1,
        beta=5,
        rho=0.1,
        Q=100,
        max_iter=200,
        elite_weight=2.0,  # 精英蚂蚁权重
        tau_min=0.1,       # 信息素最小值
        tau_max=10.0,      # 信息素最大值
        verbose=True       # 显示进度条
    )
    
    # 运行算法
    best_path, best_length = aco.run()
    
    # 输出结果
    print(f"\n最优路径: {best_path}")
    print(f"最短距离: {best_length:.2f}")
    
    # 绘制收敛曲线
    aco.plot_convergence()
    
    # 绘制TSP解
    tsp.plot_solution(best_path, best_length)

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    main()