import numpy as np
import matplotlib.pyplot as plt
from PSO1 import PSO 
from test_function import Rastrigin

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class CPSO(PSO):
    """混沌粒子群优化算法"""
    def __init__(self, pN, dim, max_iter, lower_bound=-5.12, upper_bound=5.12, v_max=2, 
                 objective_func=None, initial_positions=None, verbose=True, chaos_reset_ratio=0.2):
        """
        参数:
        chaos_reset_ratio: 每代中重置的最差粒子比例
        """
        self.chaos_reset_ratio = chaos_reset_ratio
        
        # 混沌初始化粒子位置
        if initial_positions is None:
            initial_positions = self.chaos_initialization(pN, dim, lower_bound, upper_bound)
        
        # 调用父类初始化
        super().__init__(pN, dim, max_iter, lower_bound, upper_bound, v_max, 
                         objective_func, initial_positions, verbose)
    
    def chaos_initialization(self, pN, dim, lb, ub):
        """使用Logistic混沌映射生成初始种群"""
        # 处理边界参数 - 允许标量或数组形式的输入
        if np.isscalar(lb):
            lb = np.full(dim, lb)
        else:
            assert len(lb) == dim, "下界数组长度必须与维度匹配"
            lb = np.array(lb)
            
        if np.isscalar(ub):
            ub = np.full(dim, ub)
        else:
            assert len(ub) == dim, "上界数组长度必须与维度匹配"
            ub = np.array(ub)
        
        X = np.zeros((pN, dim))
        for i in range(pN):
            # 生成混沌序列
            chaos_seq = self.generate_chaos_sequence(dim)
            # 映射到搜索空间 - 每个维度使用自己的范围
            X[i] = lb + chaos_seq * (ub - lb)
        return X
    
    def generate_chaos_sequence(self, dim):
        """生成混沌序列(Logistic映射)"""
        # 随机初始化混沌种子(避免0.5)
        seed = np.random.rand()
        while seed == 0.5:  # 避免不动点
            seed = np.random.rand()
        
        sequence = []
        x = seed
        for _ in range(dim):
            x = 4 * x * (1 - x)  # Logistic映射(μ=4)
            sequence.append(x)
        return np.array(sequence)
    
    def iterator(self):
        """执行优化迭代(添加混沌重置)"""
        fitness_history = []
        
        for t in range(self.max_iter):
            # 动态衰减惯性权重
            self.w = self.w_min + (self.w - self.w_min) * (self.max_iter - t) / self.max_iter
            
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
            
            # ======== 混沌重置步骤 ========
            # 计算需要重置的粒子数量
            num_reset = max(1, int(self.pN * self.chaos_reset_ratio))
            
            # 选择适应度最差的粒子(最大适应度值)
            worst_indices = np.argsort(current_fit)[-num_reset:]
            
            for i in worst_indices:
                # 混沌重置粒子位置
                self.X[i] = self.lower_bound + self.generate_chaos_sequence(self.dim) * (
                    self.upper_bound - self.lower_bound)
                
                # 重置粒子速度
                self.V[i] = np.random.uniform(-self.v_max, self.v_max, self.dim)
                
                # 计算新位置的适应度
                new_fit = self.function(self.X[i].reshape(1, -1))[0]
                current_fit[i] = new_fit
                
                # 更新个体最优
                if new_fit < self.p_fit[i]:
                    self.p_fit[i] = new_fit
                    self.pbest[i] = self.X[i].copy()
            
            # 更新全局最优(重置后可能变化)
            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = current_fit[min_idx]
                self.gbest = self.X[min_idx].reshape(1, -1)
            # ======== 混沌重置结束 ========
            
            # 记录每代最优适应度
            fitness_history.append(self.fit)
            
            # 打印进度
            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.fit:.6f}")

        return fitness_history

def plot_pso(fitness_history):
    plt.figure(figsize=(10, 6))
    plt.title("CPSO优化过程 - 适应度变化", fontsize=14)
    plt.xlabel("迭代次数", size=12)
    plt.ylabel("适应度值", size=12)
    t = np.arange(len(fitness_history))  
    plt.plot(t, fitness_history, color='b', linewidth=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试CPSO
    dim = 3
    # 定义每个维度的不同边界
    lower_bounds = [-5, -5.12, -10]  # 每个维度的下界
    upper_bounds = [5, 5.12, 10]     # 每个维度的上界
    cpso = CPSO(pN=30, dim=dim, max_iter=50, lower_bound=lower_bounds, upper_bound=upper_bounds, 
                v_max=2, objective_func=Rastrigin, chaos_reset_ratio=0.3)
    fitness_history = cpso.run()
    
    print("\n优化结果:")
    print(f"全局最优解: {cpso.gbest[0]}")
    print(f"全局最优适应度: {cpso.fit:.6f}")
    
    plot_pso(fitness_history)