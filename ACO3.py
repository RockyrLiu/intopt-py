import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_function import TSPProblem2
import random
import numba
import time
import os

# 启用Numba加速
os.environ['NUMBA_DISABLE_JIT'] = '0'

class OptimizedACO_TSP:
    """高度优化的蚁群算法解决大规模TSP问题"""
    def __init__(self, tsp, num_ants=100, alpha=1, beta=5, rho=0.1, Q=100, 
                 max_iter=200, elite_weight=2.0, tau_min=0.1, tau_max=10.0,
                 verbose=True, stagnation_threshold=30, diversification_rate=0.1,
                 local_search_prob=0.2, restart_interval=100, 
                 max_local_search_iters=5, parallel_local_search=False):
        """
        初始化优化参数
        
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
        stagnation_threshold: 停滞阈值
        diversification_rate: 多样化比例
        local_search_prob: 局部搜索概率
        restart_interval: 重启间隔
        max_local_search_iters: 局部搜索最大迭代次数
        parallel_local_search: 是否并行执行局部搜索
        """
        self.tsp = tsp
        self.n = tsp.N  # 城市数量
        self.D = tsp.D  # 距离矩阵
        
        # 算法参数
        self.m = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.base_rho = rho
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
        self.max_local_search_iters = max_local_search_iters
        self.parallel_local_search = parallel_local_search
        
        # 预计算启发因子(距离倒数)
        self.eta = 1 / (self.D + np.eye(self.n) * 1e10)
        self.eta = np.where(np.isfinite(self.eta), self.eta, 0)
        
        # 初始化信息素矩阵
        self.tau = np.ones((self.n, self.n)) * tau_max
        
        # 结果记录
        self.best_path = None
        self.best_length = np.inf
        self.history = {'best_length': [], 'avg_length': [], 'time_per_iter': []}
        
        # 状态跟踪
        self.last_improvement = 0
        self.stagnation_count = 0
        self.total_time = 0
    
    def initialize_ants(self):
        """使用贪心策略初始化部分蚂蚁"""
        tabu = np.zeros((self.m, self.n), dtype=int)
        
        # 1/3的蚂蚁使用贪心策略初始化
        num_greedy = max(1, int(self.m * 0.3))
        for i in range(num_greedy):
            start_city = np.random.randint(0, self.n)
            path = self._greedy_path(start_city)
            tabu[i] = path
        
        # 其余蚂蚁随机初始化
        for i in range(num_greedy, self.m):
            tabu[i] = np.random.permutation(self.n)
        
        return tabu
    
    def _greedy_path(self, start_city):
        """贪心算法生成初始路径"""
        path = [start_city]
        unvisited = set(range(self.n))
        unvisited.remove(start_city)
        
        while unvisited:
            current_city = path[-1]
            # 找到最近的城市
            nearest_city = min(unvisited, key=lambda city: self.D[current_city, city])
            path.append(nearest_city)
            unvisited.remove(nearest_city)
        
        return np.array(path)
    
    @staticmethod
    @numba.njit
    def construct_path_numba(tau, eta, alpha, beta, n, m):
        """Numba加速的路径构建"""
        tabu = np.zeros((m, n), dtype=np.int32)
        
        for ant in range(m):
            path = np.zeros(n, dtype=np.int32)
            visited = np.zeros(n, dtype=np.bool_)
            
            # 随机选择起点
            start = np.random.randint(0, n)
            path[0] = start
            visited[start] = True
            
            for step in range(1, n):
                current_city = path[step-1]
                unvisited = np.where(~visited)[0]
                
                # 计算转移概率
                prob = np.zeros(len(unvisited))
                for idx, city in enumerate(unvisited):
                    prob[idx] = (tau[current_city, city] ** alpha) * (eta[current_city, city] ** beta)
                
                # 归一化概率
                total_prob = np.sum(prob)
                if total_prob > 0:
                    prob /= total_prob
                else:
                    prob = np.ones_like(prob) / len(prob)
                
                # 轮盘赌选择
                cum_prob = np.cumsum(prob)
                r = np.random.rand()
                next_idx = np.searchsorted(cum_prob, r)
                next_city = unvisited[next_idx]
                
                path[step] = next_city
                visited[next_city] = True
            
            tabu[ant] = path
        
        return tabu
    
    def construct_solutions(self):
        """构建所有蚂蚁的路径"""
        if self.n > 50:  # 只有在大规模问题上使用Numba加速
            return self.construct_path_numba(
                self.tau, self.eta, self.alpha, self.beta, self.n, self.m
            )
        else:
            return self._construct_solutions_python()
    
    def _construct_solutions_python(self):
        """Python实现的路径构建（用于小规模问题）"""
        tabu = np.zeros((self.m, self.n), dtype=int)
        for ant in range(self.m):
            tabu[ant] = self.construct_path_for_ant(ant, np.zeros((self.m, self.n), dtype=int))
        return tabu
    
    def construct_path_for_ant(self, ant_idx, tabu):
        """Python实现的单只蚂蚁路径构建"""
        visited = set([tabu[ant_idx, 0]])
        path = [tabu[ant_idx, 0]]
        
        for step in range(1, self.n):
            current_city = path[-1]
            unvisited = [city for city in range(self.n) if city not in visited]
            
            # 计算转移概率
            prob = (self.tau[current_city, unvisited] ** self.alpha) * \
                   (self.eta[current_city, unvisited] ** self.beta)
            
            # 归一化
            total_prob = np.sum(prob)
            if total_prob > 0:
                prob /= total_prob
            else:
                prob = np.ones_like(prob) / len(prob)
            
            # 轮盘赌选择
            next_city = np.random.choice(unvisited, p=prob)
            path.append(next_city)
            visited.add(next_city)
        
        return np.array(path)
    
    def calculate_path_lengths(self, tabu):
        """向量化计算路径长度（使用预缓存）"""
        # 使用预计算的距离矩阵
        total_length = np.zeros(self.m)
        
        for i in range(self.m):
            path = tabu[i]
            length = self.D[path[-1], path[0]]  # 回到起点的距离
            for j in range(self.n - 1):
                length += self.D[path[j], path[j+1]]
            total_length[i] = length
        
        return total_length
    
    def update_pheromone(self, tabu, lengths):
        """更新信息素 - 使用向量化操作"""
        # 动态调整挥发率
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
    
    @staticmethod
    @numba.njit
    def two_opt_swap_numba(path, dist_matrix, max_iter=5):
        """Numba加速的2-opt局部搜索"""
        n = len(path)
        new_path = path.copy()
        best_length = 0.0
        
        # 计算当前路径长度
        for i in range(n):
            j = (i + 1) % n
            best_length += dist_matrix[new_path[i], new_path[j]]
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 2, n):
                    if j == n - 1:
                        next_j = 0
                    else:
                        next_j = j + 1
                    
                    # 计算交换后的长度变化
                    a, b, c, d = new_path[i-1], new_path[i], new_path[j], new_path[next_j]
                    
                    # 原边长度
                    old_edge1 = dist_matrix[a, b]
                    old_edge2 = dist_matrix[c, d]
                    
                    # 新边长度
                    new_edge1 = dist_matrix[a, c]
                    new_edge2 = dist_matrix[b, d]
                    
                    # 计算差值
                    delta = (new_edge1 + new_edge2) - (old_edge1 + old_edge2)
                    
                    if delta < -1e-5:  # 有改进
                        # 反转路径段
                        new_path[i:j+1] = new_path[i:j+1][::-1]
                        best_length += delta
                        improved = True
                        break  # 跳出内层循环，重新开始扫描
                if improved:
                    break
            iteration += 1
        
        return new_path, best_length
    
    def apply_local_search(self, path):
        """应用优化的2-opt局部搜索"""
        if self.n > 50:
            return self.two_opt_swap_numba(path, self.D, self.max_local_search_iters)
        else:
            return self._two_opt_swap_python(path)
    
    def _two_opt_swap_python(self, path):
        """Python实现的2-opt优化"""
        n = len(path)
        best_path = path.copy()
        best_length = self.tsp.calculate_path_length(path)
        improved = True
        iteration = 0
        
        while improved and iteration < self.max_local_search_iters:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 2, n):
                    # 创建新路径 (反转i到j的片段)
                    new_path = np.concatenate([
                        best_path[:i],
                        best_path[i:j+1][::-1],
                        best_path[j+1:]
                    ])
                    new_length = self.tsp.calculate_path_length(new_path)
                    
                    if new_length < best_length - 1e-5:
                        best_path = new_path
                        best_length = new_length
                        improved = True
                        break
                if improved:
                    break
            iteration += 1
        
        return best_path, best_length
    
    def diversify_population(self, tabu, lengths):
        """多样化种群：替换部分蚂蚁为随机路径"""
        num_to_replace = max(1, int(self.m * self.diversification_rate))
        replace_indices = np.random.choice(self.m, num_to_replace, replace=False)
        
        for idx in replace_indices:
            # 50%概率使用随机路径，50%使用贪心路径
            if np.random.rand() < 0.5:
                new_path = np.random.permutation(self.n)
            else:
                start_city = np.random.randint(0, self.n)
                new_path = self._greedy_path(start_city)
            
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
        """运行优化的蚁群算法"""
        # 初始化蚂蚁
        tabu = self.initialize_ants()
        
        # 初始计算路径长度
        lengths = self.calculate_path_lengths(tabu)
        
        # 更新最优解
        min_idx = np.argmin(lengths)
        self.best_length = lengths[min_idx]
        self.best_path = tabu[min_idx].copy()
        self.history['best_length'].append(self.best_length)
        self.history['avg_length'].append(np.mean(lengths))
        
        # 设置迭代范围
        iter_range = range(self.max_iter)
        if self.verbose:
            pbar = tqdm(iter_range, desc="优化蚁群算法")
        
        # 主迭代循环
        total_start = time.time()
        for iter in iter_range:
            iter_start = time.time()
            
            # 构建路径
            tabu = self.construct_solutions()
            
            # 保留上代最优路径
            tabu[0] = self.best_path
            
            # 计算路径长度
            lengths = self.calculate_path_lengths(tabu)
            
            # 应用局部搜索（以一定概率）
            if random.random() < self.local_search_prob:
                # 只对表现最好的10%蚂蚁进行局部搜索
                num_to_search = max(1, int(self.m * 0.1))
                best_indices = np.argpartition(lengths, num_to_search)[:num_to_search]
                
                for idx in best_indices:
                    improved_path, improved_length = self.apply_local_search(tabu[idx])
                    if improved_length < lengths[idx]:
                        tabu[idx] = improved_path
                        lengths[idx] = improved_length
            
            # 更新最优解
            min_idx = np.argmin(lengths)
            current_best = lengths[min_idx]
            
            improved = False
            if current_best < self.best_length - 1e-5:
                self.best_length = current_best
                self.best_path = tabu[min_idx].copy()
                self.last_improvement = iter
                improved = True
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            
            # 记录历史
            self.history['best_length'].append(self.best_length)
            self.history['avg_length'].append(np.mean(lengths))
            
            # 应用种群多样化
            if self.stagnation_count > self.stagnation_threshold:
                tabu, lengths = self.diversify_population(tabu, lengths)
                # 增加多样化比例
                self.diversification_rate = min(self.diversification_rate * 1.2, 0.5)
            
            # 更新信息素
            self.update_pheromone(tabu, lengths)
            
            # 定期重启信息素系统
            if iter > 0 and iter % self.restart_interval == 0 and iter - self.last_improvement > self.restart_interval // 2:
                self.restart_information()
            
            # 计算迭代时间
            iter_time = time.time() - iter_start
            self.history['time_per_iter'].append(iter_time)
            
            # 更新进度条
            if self.verbose:
                avg_time = np.mean(self.history['time_per_iter'][-10:]) if self.history['time_per_iter'] else 0
                remaining = (self.max_iter - iter - 1) * avg_time
                pbar.set_postfix({
                    '最优长度': f"{self.best_length:.2f}", 
                    '停滞计数': self.stagnation_count,
                    '迭代时间': f"{iter_time:.2f}s",
                    '剩余时间': f"{remaining/60:.1f}min"
                })
                pbar.update(1)
        
        # 关闭进度条
        if self.verbose:
            pbar.close()
        
        # 最终应用局部搜索优化最优解
        final_start = time.time()
        optimized_path, optimized_length = self.apply_local_search(self.best_path)
        if optimized_length < self.best_length:
            self.best_path = optimized_path
            self.best_length = optimized_length
            if self.verbose:
                print(f"最终局部搜索改进: {optimized_length:.2f} (耗时: {time.time()-final_start:.2f}s)")
        
        self.total_time = time.time() - total_start
        if self.verbose:
            print(f"总运行时间: {self.total_time:.2f}秒")
            print(f"平均迭代时间: {np.mean(self.history['time_per_iter']):.2f}秒")
        
        return self.best_path, self.best_length
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 6))
        iterations = range(len(self.history['best_length']))
        
        plt.plot(iterations, self.history['best_length'], 'b-', linewidth=2, label='最优路径长度')
        plt.plot(iterations, self.history['avg_length'], 'g--', linewidth=1, label='平均路径长度')
        
        # 标记停滞事件
        stagnation_points = [i for i in range(1, len(self.history['best_length'])) 
                         if self.history['best_length'][i] >= self.history['best_length'][i-1]]
        plt.scatter(stagnation_points, [self.history['best_length'][i] for i in stagnation_points], 
                    c='red', s=10, label='停滞点')
        
        plt.xlabel('迭代次数')
        plt.ylabel('路径长度')
        plt.title(f'优化蚁群算法收敛曲线 (总时间: {self.total_time:.2f}秒)')
        plt.legend()
        plt.grid(True)
        
        # 添加时间分布图
        if len(self.history['time_per_iter']) > 10:
            plt.figure(figsize=(12, 4))
            plt.plot(self.history['time_per_iter'], 'r-')
            plt.xlabel('迭代次数')
            plt.ylabel('时间(秒)')
            plt.title('每次迭代运行时间')
            plt.grid(True)
        
        plt.show()

def main_tsp2():
    """解决TSP问题2的主函数"""
    # 初始化TSP问题2
    tsp = TSPProblem2()
    
    # 设置优化算法参数
    aco = OptimizedACO_TSP(
        tsp,
        num_ants=200,      # 蚂蚁数量
        alpha=1,
        beta=5,
        rho=0.1,
        Q=100,
        max_iter=500,      # 迭代次数
        elite_weight=2.5,  # 精英权重
        tau_min=0.01,      # 信息素边界
        tau_max=15.0,
        verbose=True,      # 显示进度
        stagnation_threshold=30, 
        diversification_rate=0.1,
        local_search_prob=0.2,  # 局部搜索概率
        restart_interval=100,
        max_local_search_iters=5  # 局部搜索最大迭代次数
    )
    
    # 运行算法
    print(f"开始解决TSP问题2 ({tsp.N}个地点)...")
    start_time = time.time()
    best_path, best_length = aco.run()
    total_time = time.time() - start_time
    
    # 输出结果
    print(f"\n最优路径长度: {best_length:.2f}km")
    print(f"总计算时间: {total_time/60:.2f}分钟")
    
    # 绘制收敛曲线
    aco.plot_convergence()
    
    # 绘制TSP解
    tsp.plot_solution(best_path, best_length)

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    main_tsp2()