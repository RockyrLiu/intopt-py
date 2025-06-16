# 本代码参考司守奎《数学建模算法与应用》，367-372
import math
from random import random, randint
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class SA:
    def __init__(self, initial_temp, final_temp, cooling_rate):
        """
        模拟退火算法基类
        
        参数:
        initial_temp: 初始温度
        final_temp: 终止温度
        cooling_rate: 降温速率
        """
        self.T = initial_temp
        self.T0 = initial_temp
        self.Tf = final_temp
        self.alpha = cooling_rate
        self.history = {'f': [], 'T': []}
    
    def metropolis(self, current_energy, new_energy):
        """
        Metropolis接受准则
        
        参数:
        current_energy: 当前解的能量值
        new_energy: 新解的能量值
        
        返回:
        bool: 是否接受新解
        """
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
    
    def record_history(self, energy):
        """记录历史数据"""
        self.history['f'].append(energy)
        self.history['T'].append(self.T)
    
    def reset_temperature(self):
        """重置温度"""
        self.T = self.T0
        self.history = {'f': [], 'T': []}

class TSP:
    def __init__(self, data_file):
        """
        旅行商问题类
        
        参数:
        data_file: 数据文件路径
        """
        self.load_data(data_file)
        self.calc_distance_matrix()
    
    def load_data(self, data_file):
        """加载城市坐标数据"""
        sj0 = np.loadtxt(data_file)
        x = sj0[:, 0:8:2].flatten()
        y = sj0[:, 1:8:2].flatten()
        sj = np.column_stack((x, y))
        d1 = np.array([70, 40])
        self.xy = np.vstack((d1, sj, d1))
        self.sj = self.xy * np.pi / 180  # 角度转弧度
    
    def calc_distance_matrix(self):
        """计算城市间距离矩阵"""
        n = len(self.sj)
        self.d = np.zeros((n, n))
        
        for i in range(n-1):
            for j in range(i+1, n):
                self.d[i,j] = 6370 * math.acos(
                    math.cos(self.sj[i,0]-self.sj[j,0]) * 
                    math.cos(self.sj[i,1]) * math.cos(self.sj[j,1]) + 
                    math.sin(self.sj[i,1]) * math.sin(self.sj[j,1]))
        
        self.d = self.d + self.d.T
    
    def generate_initial_solution(self):
        """生成初始路径"""
        path = [0] + list(np.random.permutation(100) + 1) + [101]
        return path, self.calculate_path_length(path)
    
    def generate_neighbor(self, path):
        """生成邻域解(使用2-opt方法)"""
        c = sorted([randint(1, 100), randint(1, 100)])  # 随机选择两个不同的城市
        c1, c2 = c[0], c[1]
        new_path = path[:c1] + path[c2:c1-1:-1] + path[c2+1:]  # 反转选中的子路径
        return new_path
    
    def calculate_path_length(self, path):
        """计算路径长度"""
        length = 0
        for i in range(len(path)-1):
            length += self.d[path[i], path[i+1]]
        return length
    
    def solve_with_sa(self, sa_algorithm, max_iter=1000):
        """
        使用模拟退火算法求解
        
        参数:
        sa_algorithm: 模拟退火算法实例
        max_iter: 每个温度下的最大迭代次数
        
        返回:
        tuple: (最优路径, 最优路径长度)
        """
        # 生成初始解
        current_path, current_length = self.generate_initial_solution()
        best_path, best_length = current_path.copy(), current_length
        
        sa_algorithm.reset_temperature()
        
        # 计算总迭代次数估计(用于进度条)
        total_iters = int(math.log(sa_algorithm.Tf/sa_algorithm.T0)/math.log(sa_algorithm.alpha)) + 1
        
        # 使用单个进度条
        with tqdm(total=total_iters, desc="退火进度", unit="iter") as pbar:
            while not sa_algorithm.should_stop():
                for _ in range(max_iter):
                    # 生成新解
                    new_path = self.generate_neighbor(current_path)
                    new_length = self.calculate_path_length(new_path)
                    
                    # Metropolis准则判断是否接受新解
                    if sa_algorithm.metropolis(current_length, new_length):
                        current_path, current_length = new_path, new_length
                        
                        # 更新全局最优解
                        if new_length < best_length:
                            best_path, best_length = new_path.copy(), new_length
                
                # 记录当前状态
                sa_algorithm.record_history(best_length)
                # 降温
                sa_algorithm.cooling()
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "最优解": f"{best_length:.2f}",
                    "温度": f"{sa_algorithm.T:.2e}",
                    "当前解": f"{current_length:.2f}"
                })
        
        return best_path, best_length

def plot_path(xy, path, title):
    """绘制路径图"""
    xx = xy[path, 0]
    yy = xy[path, 1]
    plt.figure(figsize=(10, 6))
    plt.plot(xx, yy, '-*')
    plt.title(title)
    plt.show()

def plot_convergence(history, title):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['T'], history['f'])
    plt.title(title)
    plt.xlabel('Temperature')
    plt.ylabel('Path Length')
    plt.gca().invert_xaxis()
    plt.show()

if __name__ == "__main__":
    # 初始化问题实例和算法
    tsp = TSP(r'data\obj_longitude_latitude.txt')
    sa = SA(initial_temp=1, final_temp=0.1, cooling_rate=0.999)
    
    # 求解问题
    best_path, best_length = tsp.solve_with_sa(sa, max_iter=1000)
    
    # 输出结果
    print(f"最优路径长度: {best_length}")
    best_path = [int(x) for x in best_path]  # 将所有元素转为普通int
    print(f"最优路径: {best_path}")
    
    # 可视化
    plot_path(tsp.xy, best_path, '旅行商问题最优路径')
    plot_convergence(sa.history, '收敛曲线')