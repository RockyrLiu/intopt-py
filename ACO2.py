import numpy as np
import matplotlib.pyplot as plt
from test_function import Rastrigin, Square, func2
from ACO1 import ACO

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


class ACO_Continuous(ACO):
    """蚁群算法解决连续函数优化问题"""
    def __init__(self, obj_func, m=20, max_iter=200, alpha=1, beta=5, rho=0.9, Q=1, 
                 lower_bound=-5, upper_bound=5, dim=None, P0=0.2, step=0.1, verbose=True):
        """
        参数:
        obj_func: 目标函数，接受numpy数组输入
        dim: 明确指定的问题维度（可选）。若为None则根据边界参数自动推断
        lower_bound: 下界，可以是标量（所有维度相同）或列表/数组（每个维度不同）
        upper_bound: 上界，可以是标量（所有维度相同）或列表/数组（每个维度不同）
        """
        super().__init__(m, max_iter, alpha, beta, rho, Q, verbose)
        self.obj_func = obj_func
        self.P0 = P0
        self.step = step
        
        # 处理边界参数（支持标量或列表/数组）
        self.lower_bound = np.atleast_1d(lower_bound)
        self.upper_bound = np.atleast_1d(upper_bound)
        
        # 维度确定逻辑
        if dim is not None:
            self.dim = int(dim)
            # 扩展边界参数以匹配维度
            if len(self.lower_bound) == 1:
                self.lower_bound = np.repeat(self.lower_bound, self.dim)
            if len(self.upper_bound) == 1:
                self.upper_bound = np.repeat(self.upper_bound, self.dim)
        else:
            # 自动推断维度
            if len(self.lower_bound) != len(self.upper_bound):
                raise ValueError("当未指定dim时，lower_bound和upper_bound的维度必须一致")
            self.dim = len(self.lower_bound)
        
        # 最终检查边界维度
        if len(self.lower_bound) != self.dim or len(self.upper_bound) != self.dim:
            raise ValueError(f"边界参数维度({len(self.lower_bound)}和{len(self.upper_bound)})与指定维度({self.dim})不匹配")
            
        self.positions = None  # 蚂蚁位置矩阵 (dim, m)
        self.Tau = None       # 信息素
        
    def initialize(self):
        """初始化蚂蚁位置和信息素"""
        # 为每个维度生成随机位置
        self.positions = np.zeros((self.dim, self.m))
        for d in range(self.dim):
            lb = self.lower_bound[d]
            ub = self.upper_bound[d]
            self.positions[d, :] = np.random.uniform(lb, ub, self.m)
        
        # 计算初始信息素（目标函数值）
        self.Tau = np.array([self.obj_func(self.positions[:, i]) 
                           for i in range(self.m)])
    
    def clip_position(self, position):
        """确保位置在边界范围内"""
        return np.clip(position, self.lower_bound, self.upper_bound)
    
    def build_solutions(self):
        """构建蚂蚁的位置解决方案"""
        self.gen = getattr(self, 'gen', 0)  # 当前迭代次数
        lamda = 1 / (self.gen + 1)  # 动态调整参数
        
        # 找到当前最优蚂蚁
        best_idx = np.argmin(self.Tau)
        Tau_best = self.Tau[best_idx]
        
        new_positions = np.copy(self.positions)
        
        for i in range(self.m):
            P = (Tau_best - self.Tau[i]) / (Tau_best + 1e-10)  # 避免除零
            
            if P < self.P0:
                # 局部搜索
                delta = (2 * np.random.rand(self.dim) - 1)  # [-1, 1]区间随机值
                new_positions[:, i] += delta * self.step * lamda
            else:
                # 全局搜索
                for d in range(self.dim):
                    range_d = self.upper_bound[d] - self.lower_bound[d]
                    new_positions[d, i] += (np.random.rand() - 0.5) * range_d
            
            # 边界处理
            new_positions[:, i] = self.clip_position(new_positions[:, i])
        
        # 计算新位置的目标函数值
        new_values = np.array([self.obj_func(new_positions[:, i]) 
                             for i in range(self.m)])
        old_values = np.array([self.obj_func(self.positions[:, i]) 
                             for i in range(self.m)])
        
        # 更新位置（只保留更好的解）
        for i in range(self.m):
            if new_values[i] < old_values[i]:
                self.positions[:, i] = new_positions[:, i]
        
        self.gen += 1
        
    def evaluate_solutions(self):
        """评估目标函数值"""
        self.current_solutions = self.positions.T  # 转置为(m, dim)格式
        return np.array([self.obj_func(self.positions[:, i]) 
                       for i in range(self.m)])
    
    def update_pheromone(self):
        """更新信息素"""
        self.Tau = (1 - self.rho) * self.Tau + self.evaluate_solutions()
    
def test_aco_continuous():
    """测试ACO求解连续函数优化问题"""
    print("\n" + "="*50)
    print("蚁群算法求解连续函数优化问题")
    print("="*50)
    
    # 定义目标函数（使用Rastrigin函数作为测试）
    def objective_func(x):
        return Rastrigin(np.array(x))  # 将输入转换为numpy数组
    
    # 参数设置
    dim = 2  # 问题维度
    lower_bound = -5.12  # Rastrigin函数的典型搜索范围
    upper_bound = 5.12
    
    # 创建并运行ACO_Continuous
    aco = ACO_Continuous(obj_func=objective_func, 
                        m=50,               # 增加蚂蚁数量
                        max_iter=200,        # 增加迭代次数
                        alpha=1, 
                        beta=2,              # 调整beta值
                        rho=0.5,            # 调整蒸发系数
                        Q=1,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        dim=dim,
                        P0=0.2, 
                        step=0.2,            # 增大步长
                        verbose=True)
    
    best_solution, best_value = aco.run()
    
    # 输出结果
    print(f"\n最优解: {np.round(best_solution, 4)}")
    print(f"最优值: {best_value:.4f}")
    print(f"理论最优值: 0.0 (在[0,0,...,0]处取得)")
    
    # 绘制适应度进化曲线
    aco.plot_history()
    
    # 绘制函数等高线和最优解（仅适用于2维问题）
    if dim == 2:
        x = np.linspace(lower_bound, upper_bound, 100)
        y = np.linspace(lower_bound, upper_bound, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # 向量化计算提高效率
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = objective_func(np.array([X[i, j], Y[i, j]]))  # 确保传入numpy数组
        
        plt.figure(figsize=(12, 6))
        
        # 等高线图
        plt.subplot(1, 2, 1)
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar()
        plt.scatter(best_solution[0], best_solution[1], 
                   color='red', s=100, label='最优解')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('目标函数等高线图')
        plt.legend()
        
        # 3D曲面图
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.scatter(best_solution[0], best_solution[1], best_value, 
                  color='red', s=100, label='最优解')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x)')
        ax.set_title('目标函数3D视图')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 运行测试
    test_aco_continuous()