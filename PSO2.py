import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_function import KnapsackProblem

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class BinaryPSO:
    """离散粒子群优化算法 (0-1问题)"""
    def __init__(self, pN, dim, max_iter, objective_func=None, verbose=True):
        """
        初始化离散粒子群算法
        :param pN: 粒子数量
        :param dim: 问题维度
        :param max_iter: 最大迭代次数
        :param objective_func: 目标函数
        :param verbose: 是否显示进度条
        """
        # 算法参数
        self.w = 0.8  # 初始惯性权重
        self.w_min = 0.4  # 最小惯性权重
        self.c1 = 1.5  # 个体学习因子
        self.c2 = 1.5  # 社会学习因子
        self.pN = pN  # 粒子数量
        self.dim = dim  # 问题维度
        self.max_iter = max_iter  # 最大迭代次数
        self.verbose = verbose
        
        if objective_func is None:
            raise ValueError("必须提供目标函数 (objective_func 不能为 None)")
        self.function = objective_func
        
        # 初始化粒子位置和速度
        self.X = np.random.randint(0, 2, (pN, dim))  # 二进制位置 (0或1)
        self.V = np.random.uniform(-4, 4, (pN, dim))  # 速度
        
        # 初始化最优位置和适应度
        self.pbest = self.X.copy()  # 个体最优位置
        self.p_fit = self.function(self.X)  # 个体最优适应度
        self.gbest = self.X[np.argmax(self.p_fit)].copy()  # 全局最优位置
        self.fit = np.max(self.p_fit)  # 全局最优适应度
        
    def clamp_velocity(self, velocity):
        """速度限制"""
        return np.clip(velocity, -4, 4)
    
    def update_position(self, velocity):
        """根据速度更新位置 (二进制)"""
        # 使用sigmoid函数将速度转换为概率
        sig_v = 1 / (1 + np.exp(-velocity))
        # 根据概率随机生成二进制位置
        return (np.random.rand(*sig_v.shape) < sig_v).astype(int)
    
    def iterator(self):
        """执行优化迭代"""
        fitness_history = []
        
        # 根据verbose参数选择是否显示进度条
        iter_range = range(self.max_iter)
        if self.verbose:
            iter_range = tqdm(iter_range, desc="PSO优化进度", unit="iter")
        
        for t in iter_range:
            # 动态衰减惯性权重（线性递减）
            w = self.w_min + (self.w - self.w_min) * (self.max_iter - t) / self.max_iter
            
            # 生成随机因子矩阵
            r1 = np.random.rand(self.pN, self.dim)
            r2 = np.random.rand(self.pN, self.dim)
            
            # 计算全局最优矩阵
            gbest_matrix = np.tile(self.gbest, (self.pN, 1))
            
            # 更新速度
            self.V = (w * self.V + 
                     self.c1 * r1 * (self.pbest - self.X) + 
                     self.c2 * r2 * (gbest_matrix - self.X))
            self.V = self.clamp_velocity(self.V)
            
            # 更新位置
            self.X = self.update_position(self.V)
            
            # 计算新位置的适应度
            current_fit = self.function(self.X)
            
            # 找出需要更新的个体
            update_mask = current_fit > self.p_fit
            self.pbest[update_mask] = self.X[update_mask]
            self.p_fit[update_mask] = current_fit[update_mask]
            
            # 更新全局最优
            max_idx = np.argmax(current_fit)
            if current_fit[max_idx] > self.fit:
                self.fit = current_fit[max_idx]
                self.gbest = self.X[max_idx].copy()
            
            # 记录每代最优适应度
            fitness_history.append(self.fit)

        return fitness_history

    def run(self):
        """运行优化算法"""
        return self.iterator()


def plot_pso(fitness_history, title="PSO优化过程 - 适应度变化"):
    """可视化优化过程"""
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14)
    plt.xlabel("迭代次数", size=12)
    plt.ylabel("适应度值", size=12)
    t = np.arange(len(fitness_history))  
    plt.plot(t, fitness_history, color='b', linewidth=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":   
    # 背包问题参数
    capacity = 300  # 背包容量
    volumes = [95, 75, 23, 73, 50, 22, 6, 57, 89, 98]  # 物品体积
    values = [89, 59, 19, 43, 100, 72, 44, 16, 7, 64]   # 物品价值
    penalty = 2  # 惩罚系数
    
    # 创建背包问题实例
    knapsack = KnapsackProblem(capacity, volumes, values, penalty)
    
    # 创建离散PSO实例
    bpso = BinaryPSO(
        pN=100, 
        dim=len(volumes), 
        max_iter=200,
        objective_func=knapsack.fitness
    )
    
    # 运行优化
    fitness_history = bpso.run()
    
    # 输出结果
    print("\n优化结果:")
    print(f"全局最优解: {bpso.gbest}")
    print(f"选中的物品索引: {np.where(bpso.gbest == 1)[0]}")
    print(f"总价值: {bpso.fit:.2f}")
    print(f"总体积: {np.sum(bpso.gbest * volumes):.2f} (容量: {capacity})")
    
    # 绘制优化过程
    plot_pso(fitness_history, title="离散PSO优化背包问题")