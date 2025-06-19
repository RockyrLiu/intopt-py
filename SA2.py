from random import randint
import numpy as np
import matplotlib.pyplot as plt
from SA1 import SA 
from test_function import TSPProblem1, TSPProblem2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class TSP_SA(SA):
    """TSP问题的模拟退火算法"""
    def __init__(self, tsp_problem, initial_temp=1000, final_temp=1, 
                 cooling_rate=0.99, iter=100, verbose=True):
        self.tsp_problem = tsp_problem
        self.N = tsp_problem.N
        
        # 定义目标函数
        def obj_func(path):
            return tsp_problem.calculate_path_length(path)
        
        super().__init__(
            obj_func=obj_func,
            dim=self.N,
            initial_temp=initial_temp,
            final_temp=final_temp,
            cooling_rate=cooling_rate,
            iter=iter,
            verbose=verbose
        )
    
    def init_solution(self):
        """生成初始路径"""
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

def solve_TSP1():
    """解决TSP1问题"""
    # 初始化问题实例和算法
    tsp = TSPProblem1()
    sa = TSP_SA(tsp, initial_temp=1000, final_temp=1, cooling_rate=0.999, 
                iter=1000, verbose=True)
    
    # 求解问题
    best_path, best_length = sa.run()
    
    # 输出结果
    print(f"\n最优路径长度: {best_length}")
    print(f"最优路径: {best_path}")
    
    # 可视化
    tsp.plot_solution(best_path, best_length)
    plot_convergence(sa.history, '收敛曲线')

def solve_TSP2():
    """解决TSP2问题"""
    # 初始化问题实例和算法
    tsp = TSPProblem2()
    sa = TSP_SA(tsp, initial_temp=1000, final_temp=1, cooling_rate=0.999, 
                iter=1000, verbose=True)
    
    # 求解问题
    best_path, best_length = sa.run()
    
    # 输出结果
    print(f"最优路径长度: {best_length}")
    best_path = [int(x) for x in best_path]
    print(f"最优路径: {best_path}")
    
    # 可视化
    tsp.plot_solution(best_path, best_length)
    plot_convergence(sa.history, '收敛曲线')

def plot_convergence(history, title):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['temperature'], history['best_fitness'])
    plt.title(title)
    plt.xlabel('Temperature')
    plt.ylabel('Path Length')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    solve_TSP2()