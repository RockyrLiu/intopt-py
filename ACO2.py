import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_function import TSPProblem2
import random

class ImprovedACO_TSP:
    """改进的蚁群算法解决TSP问题（增强探索能力）"""
    def __init__(self, tsp, num_ants=50, alpha=1, beta=5, rho=0.1, Q=100, 
                 max_iter=200, elite_weight=2.0, tau_min=0.1, tau_max=10.0,
                 verbose=True, stagnation_threshold=50, diversification_rate=0.05,
                 local_search_prob=0.1, restart_interval=100):
        """
        初始化改进的蚁群算法参数
        
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
        stagnation_threshold: 停滞阈值，超过此值未改进则触发探索机制
        diversification_rate: 路径多样化比例
        local_search_prob: 局部搜索概率
        restart_interval: 重启间隔（迭代次数）
        """
        self.tsp = tsp
        self.n = tsp.N  # 城市数量
        self.D = tsp.D  # 距离矩阵
        
        # 算法参数
        self.m = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.base_rho = rho  # 保存基础挥发率
        self.Q = Q
        self.max_iter = max_iter
        self.elite_weight = elite_weight
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.verbose = verbose
        self.stagnation_threshold = stagnation_threshold
        self.diversification_rate = diversification_rate
        self.local_search_prob = local_search_prob
        self.restart_interval = restart_interval
        
        # 预计算启发因子(距离倒数)
        self.eta = 1 / (self.D + np.eye(self.n) * 1e10)  # 对角线设为接近0
        self.eta = np.where(np.isfinite(self.eta), self.eta, 0)  # 处理无穷大值
        
        # 初始化信息素矩阵
        self.tau = np.ones((self.n, self.n)) * tau_max
        
        # 结果记录
        self.best_path = None
        self.best_length = np.inf
        self.history = {'best_length': [], 'avg_length': []}
        
        # 状态跟踪
        self.last_improvement = 0  # 记录上次改进的迭代次数
        self.stagnation_count = 0  # 停滞计数器
    
    def initialize_ants(self):
        """初始化蚂蚁位置"""
        tabu = np.zeros((self.m, self.n), dtype=int)
        # 使用向量化操作随机放置蚂蚁
        tabu[:, 0] = np.random.choice(self.n, self.m, replace=True)
        return tabu
    
    def construct_path_for_ant(self, ant_idx, tabu):
        """为单只蚂蚁构建路径"""
        visited = set([tabu[ant_idx, 0]])  # 使用集合存储已访问城市
        path = [tabu[ant_idx, 0]]  # 记录路径
        
        for step in range(1, self.n):
            current_city = path[-1]
            
            # 获取未访问城市
            unvisited = [city for city in range(self.n) if city not in visited]
            
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
            path.append(next_city)
            visited.add(next_city)
        
        return np.array(path)
    
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
        # 动态调整挥发率 - 随停滞次数增加挥发率
        adaptive_rho = min(self.rho * (1 + 0.1 * self.stagnation_count), 0.5)
        
        # 信息素挥发
        self.tau *= (1 - adaptive_rho)
        
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
    
    def apply_local_search(self, path):
        """应用2-opt局部搜索优化路径"""
        best_path = path.copy()
        best_length = self.tsp.calculate_path_length(path)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, self.n - 2):
                for j in range(i + 1, self.n):
                    if j - i == 1:
                        continue  # 跳过相邻边
                    
                    # 尝试2-opt交换
                    new_path = np.concatenate([
                        path[:i],
                        path[i:j][::-1],
                        path[j:]
                    ])
                    new_length = self.tsp.calculate_path_length(new_path)
                    
                    # 如果找到更好的解
                    if new_length < best_length:
                        best_path = new_path
                        best_length = new_length
                        improved = True
            
            path = best_path
        
        return best_path, best_length
    
    def diversify_population(self, tabu, lengths, diversification_rate=0.1):
        """多样化种群：替换部分蚂蚁为随机路径"""
        num_to_replace = int(self.m * diversification_rate)
        replace_indices = np.random.choice(self.m, num_to_replace, replace=False)
        
        for idx in replace_indices:
            new_path = np.random.permutation(self.n)
            new_length = self.tsp.calculate_path_length(new_path)
            tabu[idx] = new_path
            lengths[idx] = new_length
        
        return tabu, lengths
    
    def restart_information(self):
        """重启信息素系统（保留最优路径）"""
        # 保存当前最优路径的信息素
        best_path_edges = set()
        for i in range(self.n):
            j = (i + 1) % self.n
            best_path_edges.add((self.best_path[i], self.best_path[j]))
        
        # 重置信息素
        self.tau = np.ones((self.n, self.n)) * self.tau_max
        
        # 增强最优路径的信息素
        for i, j in best_path_edges:
            self.tau[i, j] = self.tau_max * 1.5
        
        # 边界限制
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)
        
        # 重置停滞计数器
        self.stagnation_count = 0
        
        if self.verbose:
            print(f"重启信息素系统（迭代 {self.last_improvement}）")
    
    def run(self):
        """运行改进的蚁群算法"""
        # 初始化蚂蚁
        tabu = self.initialize_ants()
        
        # 设置迭代范围（带或不带进度条）
        iter_range = range(self.max_iter)
        if self.verbose:
            pbar = tqdm(iter_range, desc="改进蚁群算法优化")
        
        # 主迭代循环
        for iter in iter_range:
            # 构建路径
            tabu = self.construct_solutions(tabu)
            
            # 保留上代最优路径
            if self.best_path is not None:
                tabu[0] = self.best_path
            
            # 计算路径长度
            lengths = self.calculate_path_lengths(tabu)
            
            # 应用局部搜索（以一定概率）
            if random.random() < self.local_search_prob:
                # 随机选择一只蚂蚁进行局部搜索
                ant_idx = random.randint(0, self.m-1)
                improved_path, improved_length = self.apply_local_search(tabu[ant_idx])
                if improved_length < lengths[ant_idx]:
                    tabu[ant_idx] = improved_path
                    lengths[ant_idx] = improved_length
            
            # 更新最优解
            min_idx = np.argmin(lengths)
            current_best = lengths[min_idx]
            
            improved = False
            if current_best < self.best_length:
                self.best_length = current_best
                self.best_path = tabu[min_idx].copy()
                self.last_improvement = iter
                improved = True
                self.stagnation_count = 0  # 重置停滞计数器
            else:
                # 更新停滞计数器
                self.stagnation_count += 1
            
            # 记录历史
            self.history['best_length'].append(self.best_length)
            self.history['avg_length'].append(np.mean(lengths))
            
            # 应用种群多样化
            if self.stagnation_count > self.stagnation_threshold:
                tabu, lengths = self.diversify_population(tabu, lengths, self.diversification_rate)
                # 增加多样化比例
                self.diversification_rate = min(self.diversification_rate * 1.2, 0.5)
            
            # 更新信息素
            self.update_pheromone(tabu, lengths)
            
            # 定期重启信息素系统
            if iter > 0 and iter % self.restart_interval == 0 and iter - self.last_improvement > self.restart_interval // 2:
                self.restart_information()
            
            # 更新进度条
            if self.verbose:
                stagnation_info = f"停滞:{self.stagnation_count}/{self.stagnation_threshold}"
                pbar.set_postfix({
                    '最优长度': f"{self.best_length:.2f}", 
                    '多样化率': f"{self.diversification_rate:.3f}",
                    stagnation_info: ''
                })
                pbar.update(1)
        
        # 关闭进度条
        if self.verbose:
            pbar.close()
        
        # 最终应用局部搜索优化最优解
        if self.best_path is not None:
            optimized_path, optimized_length = self.apply_local_search(self.best_path)
            if optimized_length < self.best_length:
                self.best_path = optimized_path
                self.best_length = optimized_length
                if self.verbose:
                    print(f"最终局部搜索改进: {optimized_length:.2f}")
        
        return self.best_path, self.best_length
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['best_length'], 'b-', linewidth=2, label='最优路径长度')
        plt.plot(self.history['avg_length'], 'g--', linewidth=1, label='平均路径长度')
        
        # 标记停滞和重启事件
        for i, length in enumerate(self.history['best_length']):
            if i > 0 and length == self.history['best_length'][i-1]:
                plt.plot(i, length, 'ro', markersize=3)  # 红色点标记停滞
        
        plt.xlabel('迭代次数')
        plt.ylabel('路径长度')
        plt.title('改进蚁群算法收敛曲线')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # 初始化TSP问题
    tsp = TSPProblem2()
    
    # 设置改进算法参数
    aco = ImprovedACO_TSP(
        tsp,
        num_ants=200,      # 增加蚂蚁数量
        alpha=1,
        beta=5,
        rho=0.1,
        Q=100,
        max_iter=500,       # 增加迭代次数
        elite_weight=2.5,   # 增强精英策略
        tau_min=0.01,       # 更严格的信息素边界
        tau_max=15.0,
        verbose=True,       # 显示进度条
        stagnation_threshold=30,  # 30代未改进触发多样化
        diversification_rate=0.1, # 初始多样化比例
        local_search_prob=0.3,    # 30%概率应用局部搜索
        restart_interval=100      # 每100代重启信息素
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