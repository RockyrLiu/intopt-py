import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class PSO:
    """粒子群优化算法"""
    def __init__(self, pN, dim, max_iter, lower_bound=-5.12, upper_bound=5.12, v_max=2, 
                 objective_func=None, initial_positions=None, verbose=True):
        # 算法参数
        self.w = 0.8  # 初始惯性系数
        self.w_min = 0.4  # 最小惯性系数
        self.c1 = 2   # 自吸引系数
        self.c2 = 2   # 群体吸引系数
        self.pN = pN  # 粒子数
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.lower_bound = lower_bound  # 位置下界
        self.upper_bound = upper_bound  # 位置上界
        self.v_max = v_max  # 最大速度限制
        if objective_func is None:
            raise ValueError("必须提供目标函数 (objective_func 不能为 None)")
        self.function = objective_func 
        self.verbose = verbose
        
        # 初始化粒子位置和速度
        if initial_positions is not None:
            assert initial_positions.shape == (pN, dim), "初始位置形状错误"
            self.X = initial_positions.copy()
        else:
            self.X = np.random.uniform(self.lower_bound, self.upper_bound, (self.pN, self.dim))
        
        self.V = np.random.uniform(-self.v_max, self.v_max, (self.pN, self.dim))
        
        # 初始化最优位置和适应度
        self.pbest = self.X.copy()  # 个体最佳位置
        self.gbest = self.X[np.argmin(self.function(self.X))].reshape(1, -1).copy()  # 全局最优位置
        self.p_fit = self.function(self.X)  # 个体历史最佳适应度值
        self.fit = np.min(self.p_fit)  # 全局最佳适应度值
        
    def clamp_position(self, position):
        """位置钳制，确保粒子在搜索空间内"""
        return np.clip(position, self.lower_bound, self.upper_bound)
    
    def clamp_velocity(self, velocity):
        """速度钳制，防止粒子移动过快"""
        return np.clip(velocity, -self.v_max, self.v_max)
    
    def iterator(self):
        """执行优化迭代"""
        fitness_history = []
        
        for t in range(self.max_iter):
            # 动态衰减惯性权重（线性递减）
            self.w = self.w_min + (self.w - self.w_min) * (self.max_iter - t) / self.max_iter
            # # 随机惯性权重
            # self.w = np.random.uniform(self.w_min, self.w)
            
            # 生成随机因子矩阵
            r1 = np.random.rand(self.pN, self.dim)
            r2 = np.random.rand(self.pN, self.dim)
            
            # 计算全局最优矩阵
            gbest_matrix = np.repeat(self.gbest, self.pN, axis=0)
            
            # 更新速度
            self.V = (self.w * self.V + 
                     self.c1 * r1 * (self.pbest - self.X) + 
                     self.c2 * r2 * (gbest_matrix - self.X))
            self.V = self.clamp_velocity(self.V)
            
            # 更新位置
            self.X = self.X + self.V
            self.X = self.clamp_position(self.X)
            
            # 计算新位置的适应度
            current_fit = self.function(self.X)
            
            # 找出需要更新的个体
            update_mask = current_fit < self.p_fit
            self.pbest[update_mask] = self.X[update_mask]
            self.p_fit[update_mask] = current_fit[update_mask]
            
            # 更新全局最优
            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = current_fit[min_idx]
                self.gbest = self.X[min_idx].reshape(1, -1)
            
            # 记录每代最优适应度
            fitness_history.append(self.fit)
            
            # 打印进度
            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.fit:.6f}")

        return fitness_history

    def run(self):
        """运行优化算法"""
        return self.iterator()

def fitness1(X):
    """目标函数"""
    x1 = X[:, 0]  # 第一个维度
    x2 = X[:, 1]  # 第二个维度
    return 3 * x1**2 - 2.1 * x1**4 + (x1**6) / 3 + x1 * x2 - 3 * x2**2 + 3 * x2**4

def fitness2(X):
    """Rastrigin测试函数"""
    A = 10
    return A *  X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)

def plot_pso(fitness_history):
    """可视化优化过程"""
    plt.figure(figsize=(10, 6))
    plt.title("PSO优化过程 - 适应度变化", fontsize=14)
    plt.xlabel("迭代次数", size=12)
    plt.ylabel("适应度值", size=12)
    t = np.arange(len(fitness_history))  
    plt.plot(t, fitness_history, color='b', linewidth=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 创建并运行PSO优化器
    dim = 3  # 测试函数维度

    pso = PSO(pN=30, dim=dim, max_iter=50, lower_bound=-5, upper_bound=5, v_max=2, objective_func=fitness2)
    fitness_history = pso.run()
    
    # 输出最终结果
    print("\n优化结果:")
    print(f"全局最优解: {pso.gbest[0]}")
    print(f"全局最优适应度: {pso.fit:.6f}")
    
    # 绘制优化过程
    plot_pso(fitness_history)