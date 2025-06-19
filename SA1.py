import math
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm
from test_function import func1, Rastrigin

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class SA:
    """模拟退火算法基类"""
    def __init__(self, obj_func, dim, initial_temp=100, final_temp=1e-3, 
                 cooling_rate=0.95, iter=1000, verbose=True):
        """
        参数:
        obj_func: 目标函数
        dim: 变量维度
        initial_temp: 初始温度
        final_temp: 终止温度
        cooling_rate: 降温系数
        iter: 每个温度下的迭代次数
        verbose: 是否显示进度条
        """
        self.obj_func = obj_func
        self.dim = dim
        self.T0 = initial_temp
        self.Tf = final_temp
        self.T = initial_temp
        self.alpha = cooling_rate
        self.iter = iter
        self.verbose = verbose
        self.history = {'best_fitness': [], 'temperature': []}
    
    def reset_temperature(self):
        """重置温度到初始温度"""
        self.T = self.T0

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
        current_energy = self.obj_func(current_solution)
        best_solution = current_solution
        best_energy = current_energy
        
        # 记录初始状态
        self.history['best_fitness'].append(best_energy)
        self.history['temperature'].append(self.T)
        
        # 计算总迭代次数估计(用于进度条)
        total_iters = int(math.log(self.Tf/self.T0)/math.log(self.alpha)) + 1
        
        # 使用进度条
        with tqdm(total=total_iters, desc="退火进度", disable=not self.verbose) as pbar:
            while not self.should_stop():
                for _ in range(self.iter):
                    # 生成新解
                    new_solution = self.generate_neighbor(current_solution)
                    new_energy = self.obj_func(new_solution)
                    
                    # Metropolis准则判断是否接受新解
                    if self.metropolis(current_energy, new_energy):
                        current_solution, current_energy = new_solution, new_energy
                        
                        # 更新全局最优解
                        if new_energy < best_energy:
                            best_solution, best_energy = new_solution, new_energy
                
                # 记录当前状态
                self.history['best_fitness'].append(best_energy)
                self.history['temperature'].append(self.T)
                
                # 降温
                self.cooling()
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "最优值": f"{best_energy:.4f}",
                    "温度": f"{self.T:.2e}"
                })
        
        return best_solution, best_energy

def plot_history(history):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['temperature'], history['best_fitness'])
    plt.title('模拟退火收敛曲线')
    plt.xlabel('温度')
    plt.ylabel('目标函数值')
    plt.gca().invert_xaxis()  # 温度从高到低显示
    plt.grid(True)
    plt.show()

class ContinuousSA(SA):
    """连续优化问题的SA实现"""
    def __init__(self, obj_func, dim, lower_bound, upper_bound, 
                    initial_temp=100, final_temp=1e-3, cooling_rate=0.95, 
                    iter=1000, verbose=True):
        super().__init__(obj_func, dim, initial_temp, final_temp, cooling_rate, iter, verbose)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def init_solution(self):
        """初始化解"""
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
    
    def generate_neighbor(self, x):
        """生成邻域解"""
        delta = (self.upper_bound - self.lower_bound) * 0.1
        new_x = x + np.random.uniform(-delta, delta, self.dim)
        return np.clip(new_x, self.lower_bound, self.upper_bound)

if __name__ == "__main__":
    # 定义目标函数
    def objective(x):
        return func1(x.reshape(1, -1))[0]
    
    # 创建并运行SA算法
    sa = ContinuousSA(obj_func=Rastrigin, dim=10, lower_bound=-5.12, upper_bound=5.12,
                      initial_temp=100, final_temp=1e-4, cooling_rate=0.999, 
                      iter=100, verbose=True)
    
    best_solution, best_value = sa.run()
    
    # 输出结果
    print(f"\n最优解: {np.round(best_solution, 4)}")
    print(f"最优值: {best_value:.4f}")
    
    # 绘制收敛曲线
    plot_history(sa.history)