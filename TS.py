import numpy as np
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
from tsp import problem1

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class TS:
    """禁忌搜索算法基类"""
    def __init__(self, func, tabu_length = 10, candidate_size = 50, 
                 max_iter = 500, init_positions = None,
                 verbose = True):
        """
        参数说明:
        func: 目标函数
        tabu_length: 禁忌长度
        candidate_size: 候选解数量
        max_iter: 最大迭代次数
        init_positions: 初始解列表(可选)
        verbose: 是否显示进度条
        """
        self.function = func
        self.tabu_length = tabu_length
        self.candidate_size = candidate_size
        self.max_iter = max_iter
        self.init_positions = init_positions
        self.verbose = verbose
        
        # 历史记录
        self.history = {
            'best_fitness': [],
            'current_fitness': [],
            'best_solution': []
        }
    
    def init_solution(self):
        """初始化解(由子类实现)"""
        raise NotImplementedError("子类必须实现init_solution方法")

    def generate_candidates(self, current_solution):
        """生成候选解(由子类实现)"""
        raise NotImplementedError("子类必须实现generate_candidates方法")
    
    def update_tabu(self, move):
        """更新禁忌表(由子类实现)"""
        raise NotImplementedError("子类必须实现update_tabu方法")
    
    def is_tabu(self, move):
        """检查是否在禁忌表中(由子类实现)"""
        raise NotImplementedError("子类必须实现is_tabu方法")
    
    def _record_history(self, current_energy, best_energy, best_solution):
        """记录历史数据"""
        self.history['best_fitness'].append(float(best_energy))
        self.history['current_fitness'].append(float(current_energy))
        self.history['best_solution'].append(best_solution.copy())
    
    def run(self):
        """优化迭代主函数"""
        # 初始化当前解和最优解
        current_solution = self.init_solution()
        current_energy = self.function(current_solution)
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        # 记录初始状态
        self._record_history(current_energy, best_energy, best_solution)
        
        # 使用进度条
        with tqdm(total=self.max_iter, desc="禁忌搜索进度", disable=not self.verbose) as pbar:
            for _ in range(self.max_iter):
                # 生成候选解
                candidates = self.generate_candidates(current_solution)
                
                # 评估候选解
                best_candidate = None
                best_candidate_energy = float('inf')
                
                for candidate in candidates:
                    candidate_energy = self.function(candidate['solution'])
                    move = candidate['move']
                    
                    # 藐视准则: 如果候选解优于当前最优解，直接接受
                    if candidate_energy < best_energy:
                        best_candidate = candidate
                        best_candidate_energy = candidate_energy
                        break
                    
                    # 如果不是禁忌解且优于当前候选解
                    if not self.is_tabu(move) and candidate_energy < best_candidate_energy:
                        best_candidate = candidate
                        best_candidate_energy = candidate_energy
                
                # 更新解和禁忌表
                if best_candidate is not None:
                    current_solution = best_candidate['solution']
                    current_energy = best_candidate_energy
                    self.update_tabu(best_candidate['move'])
                    
                    # 更新全局最优解
                    if current_energy < best_energy:
                        best_solution = current_solution.copy()
                        best_energy = current_energy
                
                # 记录当前状态
                self._record_history(current_energy, best_energy, best_solution)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "最优值": f"{float(best_energy):.4f}",
                    "当前值": f"{float(current_energy):.4f}"
                })
        
        return best_solution, best_energy

class TSP_TS(TS):
    """TSP问题的禁忌搜索算法"""
    def __init__(self, tsp_problem, tabu_length=None, candidate_size=50, 
                 max_iter=500, init_positions = None,
                 verbose=True):
        """
        参数:
        tsp_problem: TSP问题实例
        tabu_length: 禁忌长度(默认为sqrt(n*(n-1)/2))
        candidate_size: 候选解数量
        max_iter: 最大迭代次数
        init_positions: 初始路径列表(可选)
        verbose: 是否显示进度条
        """
        self.tsp_problem = tsp_problem
        self.N = tsp_problem.N  # 城市数量
        
        # 计算默认禁忌长度
        if tabu_length is None:
            tabu_length = round(np.sqrt(self.N*(self.N-1)/2))
        
        # 定义目标函数
        def obj_func(path):
            return tsp_problem.calculate_path_length(path)
        
        # TS基类初始化
        super().__init__(
            func=obj_func,
            tabu_length=tabu_length,
            candidate_size=candidate_size,
            max_iter=max_iter,
            init_positions=init_positions,
            verbose=verbose
        )
        
        # 初始化禁忌表
        self.tabu_table = np.zeros((self.N, self.N))
    
    def init_solution(self):
        """生成初始路径"""
        if self.init_positions is not None and len(self.init_positions) > 0:
            # 使用提供的初始解(随机选择一个)
            return random.choice(self.init_positions).copy()
        else:
            # 随机生成初始解
            return list(np.random.permutation(self.N))
    
    def generate_candidates(self, current_solution):
        """生成候选解(交换两个城市)"""
        candidates = []
        generated_pairs = set()
        
        while len(candidates) < self.candidate_size:
            # 随机选择两个不同的城市
            i, j = random.sample(range(self.N), 2)
            i, j = min(i, j), max(i, j)
            
            # 确保不重复生成相同的交换对
            if (i, j) in generated_pairs:
                continue
            generated_pairs.add((i, j))
            
            # 生成新解
            new_solution = current_solution.copy()
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
            candidates.append({
                'solution': new_solution,
                'move': (current_solution[i], current_solution[j])
            })
        
        return candidates
    
    def update_tabu(self, move):
        """更新禁忌表"""
        # 禁忌表中所有值减1
        self.tabu_table = np.maximum(self.tabu_table - 1, 0)
        # 添加新禁忌项
        city1, city2 = move
        self.tabu_table[city1, city2] = self.tabu_length
        self.tabu_table[city2, city1] = self.tabu_length
    
    def is_tabu(self, move):
        """检查是否在禁忌表中"""
        city1, city2 = move
        return self.tabu_table[city1, city2] > 0

class Continuous_TS(TS):
    """连续优化问题的TS实现"""
    def __init__(self, func, bounds, tabu_length=10, candidate_size=50, 
                 max_iter=500, neighbor_scale=0.1, 
                 init_positions = None,
                 verbose=True):
        """
        参数:
        func: 目标函数
        bounds: 变量边界列表
        tabu_length: 禁忌长度
        candidate_size: 候选解数量
        max_iter: 最大迭代次数
        neighbor_scale: 邻域搜索范围比例
        init_positions: 初始解列表(可选)
        verbose: 是否显示进度条
        """
        super().__init__(
            func=func,
            tabu_length=tabu_length,
            candidate_size=candidate_size,
            max_iter=max_iter,
            init_positions=init_positions,
            verbose=verbose
        )
        
        self.bounds = bounds
        self.dim = len(bounds)
        self.neighbor_scale = neighbor_scale
        self.lower_bound = np.array([b[0] for b in bounds])
        self.upper_bound = np.array([b[1] for b in bounds])
        
        # 禁忌表(存储最近访问的解)
        self.tabu_list = []
    
    def init_solution(self):
        """初始化解"""
        if self.init_positions is not None and len(self.init_positions) > 0:
            # 使用提供的初始解(随机选择一个)
            solution = np.array(random.choice(self.init_positions))
            # 确保解在边界内
            return np.clip(solution, self.lower_bound, self.upper_bound)
        else:
            # 随机生成初始解
            solution = np.zeros(self.dim)
            for d in range(self.dim):
                solution[d] = np.random.uniform(self.bounds[d][0], self.bounds[d][1])
            return solution
    
    def generate_candidates(self, current_solution):
        """生成候选解"""
        candidates = []
        for _ in range(self.candidate_size):
            # 在邻域内随机生成新解
            new_solution = current_solution.copy()
            for d in range(self.dim):
                delta = (self.bounds[d][1] - self.bounds[d][0]) * self.neighbor_scale
                new_solution[d] += np.random.uniform(-delta, delta)
                # 确保解在边界内
                new_solution[d] = np.clip(new_solution[d], self.bounds[d][0], self.bounds[d][1])
            
            # 计算移动量(用于禁忌检查)
            move = new_solution - current_solution
            
            candidates.append({
                'solution': new_solution,
                'move': move
            })
        
        return candidates
    
    def update_tabu(self, move):
        """更新禁忌表"""
        self.tabu_list.append(move)
        if len(self.tabu_list) > self.tabu_length:
            self.tabu_list.pop(0)
    
    def is_tabu(self, move):
        """检查是否在禁忌表中"""
        for tabu_move in self.tabu_list:
            if np.allclose(move, tabu_move, atol=1e-6):
                return True
        return False

def ts(func, bounds=None, tabu_length=10, candidate_size=50, max_iter=500, 
       neighbor_scale=0.1, init_positions=None, verbose=True, plot=True):
    """
    禁忌搜索算法(连续优化问题)

    参数:
        func: 目标函数
        bounds: 变量边界列表 [(min, max), ...]
        tabu_length: 禁忌长度 (默认10)
        candidate_size: 候选解数量 (默认50)
        max_iter: 最大迭代次数 (默认500)
        neighbor_scale: 邻域搜索范围比例 (默认0.1)
        init_positions: 初始解列表(可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 历史记录
    """
    optimizer = Continuous_TS(
        func=func,
        bounds=bounds,
        tabu_length=tabu_length,
        candidate_size=candidate_size,
        max_iter=max_iter,
        neighbor_scale=neighbor_scale,
        init_positions=init_positions,
        verbose=verbose
    )
    
    best_solution, best_energy = optimizer.run()
    
    if plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(optimizer.history['current_fitness'], 'r--', alpha=0.3, label='当前适应度')
        plt.title('禁忌搜索收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(optimizer.history['best_fitness'], 'b-', label='最优适应度')
        plt.title('适应度变化曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return best_solution, optimizer.history

def ts_tsp(tsp_problem, tabu_length=None, candidate_size=50, max_iter=500, 
           init_positions=None, verbose=True, plot=True):
    """
    禁忌搜索接口(TSP问题)
    
    参数:
        tsp_problem: TSP问题实例
        tabu_length: 禁忌长度 (默认sqrt(n*(n-1)/2))
        candidate_size: 候选解数量 (默认50)
        max_iter: 最大迭代次数 (默认500)
        init_positions: 初始路径列表(可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        最优路径, 路径长度, 历史记录
    """
    optimizer = TSP_TS(
        tsp_problem=tsp_problem,
        tabu_length=tabu_length,
        candidate_size=candidate_size,
        max_iter=max_iter,
        init_positions=init_positions,
        verbose=verbose
    )
    
    best_path, best_length = optimizer.run()
    
    if plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.history['best_fitness'], 'b-', label='最优路径长度')
        plt.plot(optimizer.history['current_fitness'], 'r--', alpha=0.3, label='当前路径长度')
        plt.title('禁忌搜索收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('路径长度')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(optimizer.history['best_fitness'], 'b-', label='最优路径长度')
        plt.title('路径长度变化曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('路径长度')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 绘制最优路径
        tsp_problem.plot_solution(best_path, best_length)
    
    return best_path, best_length, optimizer.history

def main1():
    # 示例1: 连续优化问题
    def test_func(x):
        x1, x2 = x
        numerator = math.cos(x1**2 + x2**2) - 0.1
        denominator = 1 + 0.3*(x1**2 + x2**2)**2
        return numerator / denominator + 3
    
    # 自定义初始解
    custom_init = [
        [1.0, 1.0],
        [-2.0, 2.0],
        [3.0, -3.0]
    ]
    
    best_solution, history = ts(
        func=test_func,
        bounds=[(-5, 5), (-5, 5)],
        tabu_length=10,
        candidate_size=50,
        max_iter=200,
        neighbor_scale=0.1,
        init_positions=custom_init,
        verbose=True
    )
    
    print(f"最优解: {best_solution}")
    print(f"最优值: {history['best_fitness'][-1]}")

def main2(): 
    # 示例2: TSP问题
    best_path, best_length, history = ts_tsp(
        tsp_problem=problem1,
        tabu_length=None,
        candidate_size=150,
        max_iter=3000,
        init_positions=None,
        verbose=True,
        plot=True
    )

if __name__ == "__main__":
    main1()