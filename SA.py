import math
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class SA:
    """模拟退火算法基类"""
    def __init__(self, func, bounds=None, initial_temp=100, final_temp=1e-3, 
                 cooling_rate=0.95, iter_per_temp=100, 
                 init_positions=None, verbose=True):
        """
        参数说明:
        func: 目标函数
        bounds: 边界列表，每个元素是元组 (min, max)（对于连续问题）或None（对于离散问题）
        initial_temp: 初始温度
        final_temp: 终止温度
        cooling_rate: 降温系数
        iter_per_temp: 每个温度下的迭代次数
        init_positions: 初始解列表(可选)，如果提供则使用第一个解作为初始解
        verbose: 是否显示进度条
        """
        self.function = func
        self.bounds = bounds
        self.dim = len(bounds) if bounds is not None else None
        self.T0 = initial_temp
        self.Tf = final_temp
        self.T = initial_temp
        self.alpha = cooling_rate
        self.iter_per_temp = iter_per_temp
        self.verbose = verbose
        self.init_positions = init_positions
        
        # 提取边界（仅当bounds不为None时）
        if bounds is not None:
            self.lower_bound = np.array([b[0] for b in bounds])
            self.upper_bound = np.array([b[1] for b in bounds])
        else:
            self.lower_bound = None
            self.upper_bound = None
        
        # 历史记录
        self.history = {
            'best_fitness': [],
            'current_fitness': [],
            'temperature': [],
            'best_solution': []
        }
    
    def init_solution(self):
        """初始化解(由子类实现)"""
        raise NotImplementedError("子类必须实现init_solution方法")

    def generate_neighbor(self, solution):
        """生成邻域解(由子类实现)"""
        raise NotImplementedError("子类必须实现generate_neighbor方法")
    
    def metropolis(self, current_energy, new_energy):
        """Metropolis接受准则"""
        if new_energy < current_energy:
            return True
        else:
            p = math.exp((current_energy - new_energy) / self.T)
            return random() < p
    
    def cooling(self):
        """温度下降"""
        self.T *= self.alpha
    
    def should_stop(self):
        """停止条件判断"""
        return self.T < self.Tf
    
    def run(self):
        """优化迭代主函数"""
        # 初始化当前解和最优解
        current_solution = self.init_solution()
        current_energy = self.function(current_solution)
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        # 记录初始状态
        self._record_history(current_energy, best_energy, best_solution)
        
        # 计算总迭代次数估计(用于进度条)
        total_iters = int(math.log(self.Tf/self.T0)/math.log(self.alpha)) + 1
        
        # 使用进度条
        with tqdm(total=total_iters, desc="退火进度", disable=not self.verbose) as pbar:
            while not self.should_stop():
                for _ in range(self.iter_per_temp):
                    # 生成新解
                    new_solution = self.generate_neighbor(current_solution)
                    new_energy = self.function(new_solution)
                    
                    # Metropolis准则判断是否接受新解
                    if self.metropolis(current_energy, new_energy):
                        current_solution, current_energy = new_solution, new_energy
                        
                        # 更新全局最优解
                        if new_energy < best_energy:
                            best_solution, best_energy = new_solution.copy(), new_energy
                
                # 记录当前状态
                self._record_history(current_energy, best_energy, best_solution)
                
                # 降温
                self.cooling()
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "最优值": f"{float(best_energy):.4f}",  # 确保转换为标量
                    "温度": f"{self.T:.2e}"
                })
        
        return best_solution, best_energy
    
    def _record_history(self, current_energy, best_energy, best_solution):
        """记录历史数据"""
        self.history['best_fitness'].append(float(best_energy))  # 确保存储标量
        self.history['current_fitness'].append(float(current_energy))
        self.history['temperature'].append(self.T)
        self.history['best_solution'].append(best_solution.copy())

class ContinuousSA(SA):
    """连续优化问题的SA实现"""
    def __init__(self, func, bounds, initial_temp=100, final_temp=1e-3, 
                 cooling_rate=0.95, iter_per_temp=100, neighbor_scale=0.1,
                 init_positions=None, verbose=True):
        """
        neighbor_scale: 邻域搜索范围比例(相对于变量范围)
        init_positions: 初始解列表(可选)，如果提供则使用第一个解作为初始解
        """
        super().__init__(func, bounds, initial_temp, final_temp, cooling_rate, 
                         iter_per_temp, init_positions, verbose)
        self.neighbor_scale = neighbor_scale
        self.func = wrapper_black_box(func)
    
    def init_solution(self):
        """初始化解"""
        if self.init_positions is not None and len(self.init_positions) > 0:
            # 使用提供的初始解
            solution = np.array(self.init_positions[0])
            # 确保解在边界内
            for d in range(self.dim):
                solution[d] = np.clip(solution[d], self.bounds[d][0], self.bounds[d][1])
            return solution
        else:
            # 随机生成初始解
            solution = np.zeros(self.dim)
            for d in range(self.dim):
                solution[d] = np.random.uniform(self.bounds[d][0], self.bounds[d][1])
            return solution
    
    def generate_neighbor(self, x):
        """生成邻域解"""
        new_x = x.copy()
        for d in range(self.dim):
            delta = (self.bounds[d][1] - self.bounds[d][0]) * self.neighbor_scale
            new_x[d] += np.random.uniform(-delta, delta)
            # 确保解在边界内
            new_x[d] = np.clip(new_x[d], self.bounds[d][0], self.bounds[d][1])
        return new_x

def wrapper_black_box(black_box_func):
    """封装黑箱函数，确保返回标量值"""
    def adapted_func(X):
        # 处理输入为1D数组的情况
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        # 调用原始函数并提取标量值
        result = black_box_func(X.tolist()[0])  # 直接处理单点输入
        return float(result)  # 确保返回标量
    return adapted_func

class TSP_SA(SA):
    """TSP问题的模拟退火算法"""
    def __init__(self, tsp_problem, initial_temp=1000, final_temp=1, 
                 cooling_rate=0.99, iter_per_temp=100, verbose=True):
        """
        参数:
        tsp_problem: TSP问题实例
        initial_temp: 初始温度
        final_temp: 终止温度
        cooling_rate: 降温系数
        iter_per_temp: 每个温度下的迭代次数
        verbose: 是否显示进度条
        """
        self.tsp_problem = tsp_problem
        self.N = tsp_problem.N  # 城市数量
        
        # 定义目标函数
        def obj_func(path):
            return tsp_problem.calculate_path_length(path)
        
        # SA基类初始化
        super().__init__(
            func=obj_func,
            bounds=None,  # TSP问题无边界约束
            initial_temp=initial_temp,
            final_temp=final_temp,
            cooling_rate=cooling_rate,
            iter_per_temp=iter_per_temp,
            verbose=verbose
        )
    
    def init_solution(self):
        """生成初始路径(随机排列)"""
        return list(np.random.permutation(self.N))
    
    def generate_neighbor(self, path):
        """生成邻域解(使用2-opt方法)"""
        # 确保选择两个不同的索引
        c1, c2 = 0, 0
        while c1 == c2:
            c1, c2 = sorted([randint(0, self.N-1), randint(0, self.N-1)])
        
        # 应用2-opt交换
        new_path = path[:c1] + path[c1:c2+1][::-1] + path[c2+1:]
        return new_path


def sa(func, bounds, initial_temp=100, final_temp=1e-3, 
       cooling_rate=0.95, iter_per_temp=100, neighbor_scale=0.1, 
       init_positions=None, verbose=True, plot=True):
    """
    模拟退火算法(连续优化问题)
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        initial_temp: 初始温度 (默认100)
        final_temp: 终止温度 (默认1e-3)
        cooling_rate: 降温系数 (默认0.95)
        iter_per_temp: 每个温度下的迭代次数 (默认100)
        neighbor_scale: 邻域搜索范围比例 (默认0.1)
        init_positions: 初始解列表(可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 历史记录
    """
    optimizer = ContinuousSA(
        func=func,
        bounds=bounds,
        initial_temp=initial_temp,
        final_temp=final_temp,
        cooling_rate=cooling_rate,
        iter_per_temp=iter_per_temp,
        neighbor_scale=neighbor_scale,
        init_positions=init_positions,
        verbose=verbose
    )
    best_solution, best_energy = optimizer.run()
    
    if plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.history['temperature'], optimizer.history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(optimizer.history['temperature'], optimizer.history['current_fitness'], 'r--', alpha=0.3, label='当前适应度')
        plt.title('模拟退火收敛曲线')
        plt.xlabel('温度')
        plt.ylabel('目标函数值')
        plt.gca().invert_xaxis()  # 温度从高到低显示
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

def sa_tsp(tsp_problem, initial_temp=1000, final_temp=1, 
              cooling_rate=0.99, iter_per_temp=100, verbose=True, plot=True):
    """
    模拟退火算法(TSP问题)
    参数:
        tsp_problem: TSP问题实例
        initial_temp: 初始温度 (默认1000)
        final_temp: 终止温度 (默认1)
        cooling_rate: 降温系数 (默认0.99)
        iter_per_temp: 每个温度下的迭代次数 (默认100)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        最优路径, 路径长度, 历史记录
    """
    # 创建SA实例
    sa = TSP_SA(
        tsp_problem=tsp_problem,
        initial_temp=initial_temp,
        final_temp=final_temp,
        cooling_rate=cooling_rate,
        iter_per_temp=iter_per_temp,
        verbose=verbose
    )
    
    # 运行算法
    best_path, best_length = sa.run()
    
    if plot:
        # 绘制收敛曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(sa.history['temperature'], sa.history['best_fitness'], 'b-', label='最优路径长度')
        plt.plot(sa.history['temperature'], sa.history['current_fitness'], 'r--', alpha=0.3, label='当前路径长度')
        plt.title('模拟退火收敛曲线')
        plt.xlabel('温度')
        plt.ylabel('路径长度')
        plt.gca().invert_xaxis()  # 温度从高到低显示
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(sa.history['best_fitness'], 'b-', label='最优路径长度')
        plt.title('路径长度变化曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('路径长度')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 绘制最优路径
        tsp_problem.plot_solution(best_path, best_length)
    
    return best_path, best_length, sa.history


def main1():
        # 定义原始黑箱函数（输入为列表）
    def rastrigin(X):
        A = 10
        return A * len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])
    
    # 自定义初始解
    custom_init = [np.random.uniform(-5.12, 5.12, 10) for _ in range(3)]  # 3个初始解
    
    # 调用SA优化
    best_solution, history = sa(
        func=rastrigin,
        bounds=[(-5.12, 5.12)] * 10,  # 10维Rastrigin函数
        initial_temp=100,
        final_temp=1e-4,
        cooling_rate=0.999,
        iter_per_temp=100,
        neighbor_scale=0.1,
        init_positions=custom_init,
        verbose=True)
    
    print("最优解:", best_solution)
    print("最优值:", history['best_fitness'][-1])

def main2():
    from tsp import problem1
    # 求解TSP问题
    best_path, best_length, history = sa_tsp(
        problem1,
        initial_temp=1000,
        final_temp=1,
        cooling_rate=0.999,
        iter_per_temp=1000,
        verbose=True
    )
    
    print(f"\n最优路径长度: {best_length:.2f}")
    print(f"最优路径: {best_path}")

if __name__ == "__main__":
    main1()