import numpy as np
import matplotlib.pyplot as plt
from test_function import Rastrigin, Square, func2, plot_func2, func3, plot_func3

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class DE:
    """基本差分进化算法"""
    def __init__(self, Np, dim, max_iter, F0, crossover_rate, obj_function,
                 lower_bound, upper_bound):
        self.Np = Np
        self.dim = dim
        self.max_iter = max_iter
        self.F0 = F0
        self.crossover_rate = crossover_rate
        self.obj_function = obj_function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.history = {'best_fitness': [], 'avg_fitness': []}

    def init_population(self):
        """初始化种群"""
        x = np.random.rand(self.dim, self.Np) * (self.upper_bound - self.lower_bound) + self.lower_bound
        fitness = np.array([self.obj_function(x[:, m]) for m in range(self.Np)])  # 计算适应度
        self.history['best_fitness'].append(fitness.min())
        self.history['avg_fitness'].append(fitness.mean())
        return x, fitness
    
    def check_bounds(self, u):
        """边界处理"""
        u = np.clip(u, self.lower_bound, self.upper_bound)
        # 对于越界的值重新随机生成
        mask = (u < self.lower_bound) | (u > self.upper_bound)
        u[mask] = np.random.rand(np.sum(mask)) * (self.upper_bound - self.lower_bound) + self.lower_bound
        return u

    def F(self, gen):
        """基本变异"""
        return self.F0

    def adaptive_F(self, gen):
        """自适应变异"""
        lamda = np.exp(1 - self.max_iter/(self.max_iter + 1 - gen))
        return self.F0 * 2**lamda

    def mutate(self, x, F):
        """变异操作"""
        v = np.zeros((self.dim, self.Np))
        for m in range(self.Np):
            # 选择三个不同的个体
            r1, r2, r3 = np.random.choice([i for i in range(self.Np) if i != m], 3, replace=False)
            v[:, m] = x[:, r1] + F * (x[:, r2] - x[:, r3])
        return v

    def crossover(self, x, v):
        """交叉操作"""
        u = np.zeros((self.dim, self.Np))
        r = np.random.randint(0, self.dim)
        for n in range(self.dim):
            cr = np.random.rand()
            if cr <= self.crossover_rate or n == r:
                u[n, :] = v[n, :]
            else:
                u[n, :] = x[n, :]
        return u

    def iterator(self):
        """优化迭代"""
        # 差分进化循环
        x, fitness = self.init_population()
        for gen in range(self.max_iter):
            F = self.F(gen)
            # 变异操作
            v = self.mutate(x, F)
            # 交叉操作
            u = self.crossover(x, v)
            # 边界处理
            u = self.check_bounds(u)
            # 选择操作
            fitness1 = np.array([self.obj_function(u[:, m]) for m in range(self.Np)])
            for m in range(self.Np):
                if fitness1[m] < fitness[m]:
                    x[:, m] = u[:, m]
                    fitness[m] = fitness1[m]
            
            self.history['best_fitness'].append(fitness.min())
            self.history['avg_fitness'].append(fitness.mean())
        
        # 排序结果
        sorted_indices = np.argsort(fitness)
        x = x[:, sorted_indices]
        best_solution = x[:, 0]    # 最优变量
        best_fitness = np.min(fitness)  # 最优值

        return best_solution, best_fitness
    
    def run(self):
        return self.iterator()

class adaptiveDE(DE):
    """自适应差分进化算法"""
    def __init__(self, Np, dim, max_iter, F0, crossover_rate, obj_function, lower_bound, upper_bound):
        super().__init__(Np, dim, max_iter, F0, crossover_rate, obj_function, lower_bound, upper_bound)
    def F(self, gen):
        return self.adaptive_F(gen)

class intDE(DE):
    """离散差分进化算法"""
    def __init__(self, Np, dim, max_iter, F0, crossover_rate, obj_function, lower_bound, upper_bound):
        super().__init__(Np, dim, max_iter, F0, crossover_rate, obj_function, lower_bound, upper_bound)

    def init_population(self):
        """初始化种群（使用整数初始化）"""
        x = np.random.randint(self.lower_bound, self.upper_bound+1, size=(self.dim, self.Np))
        fitness = np.array([self.obj_function(x[:, m]) for m in range(self.Np)])  # 计算适应度
        self.history['best_fitness'].append(fitness.min())
        self.history['avg_fitness'].append(fitness.mean())
        return x, fitness
    
    def mutate(self, x, F):
        """变异操作"""
        v = np.zeros((self.dim, self.Np), dtype=int)
        for m in range(self.Np):
            # 选择三个不同的个体
            r1, r2, r3 = np.random.choice([i for i in range(self.Np) if i != m], 3, replace=False)
            v[:, m] = np.floor(x[:, r1] + F * (x[:, r2] - x[:, r3])).astype(int)
        return v
    
    def crossover(self, x, v):
        """交叉操作"""
        u = np.zeros((self.dim, self.Np), dtype=int)
        r = np.random.randint(0, self.dim)
        for n in range(self.dim):
            cr = np.random.rand()
            if cr <= self.crossover_rate or n == r:
                u[n, :] = v[n, :]
            else:
                u[n, :] = x[n, :]
        return u
    
    def check_bounds(self, u):
        return np.clip(u, self.lower_bound, self.upper_bound)

def plot_history(history): 
    """绘制适应度进化曲线"""
    plt.figure()
    plt.plot(history['best_fitness'], label='最佳适应度')
    plt.plot(history['avg_fitness'], label='平均适应度')
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title('适应度进化曲线')
    plt.legend()
    plt.show()

def main1():
    Np = 50        # 种群个体数量
    dim = 10       # 变量维度
    max_iter = 200  # 最大进化代数
    F0 = 0.4       # 初始变异算子
    crossover_rate = 0.1  # 交叉概率
        
    de1 = adaptiveDE(Np, dim, max_iter, F0, crossover_rate, Square, lower_bound=-20, upper_bound=20)
    best_solution, best_value = de1.run()
    print(f"最优解: {np.round(best_solution, 4)}")
    print(f"最优值: {best_value:.4f}")
    plot_history(de1.history)

def main2():
    # 使用直接截断
    class DE2(DE):
        def __init__(self, Np, dim, max_iter, F0, crossover_rate, obj_function, lower_bound, upper_bound):
            super().__init__(Np, dim, max_iter, F0, crossover_rate, obj_function, lower_bound, upper_bound)

        def check_bounds(self, u):
            return np.clip(u, self.lower_bound, self.upper_bound)
        
    de2 = DE2(20, 2, 100, 0.5, 0.1, func2, -4, 4)
    best_solution, best_value = de2.run()
    print(f"最优解:  {np.round(best_solution, 4)}")
    print(f"最优值: {best_value:.4f}")
    plot_history(de2.history)
    plot_func2()

def main3():
    # 整数规划
    de3 = intDE(20, 2, 100, 0.5, 0.1, func3, -100, 100)
    best_solution, best_value = de3.run()
    print(f"最优解:  {np.round(best_solution, 4)}")
    print(f"最优值: {best_value:.4f}")
    plot_history(de3.history)
    plot_func3()

def main4():
    Np = 50        # 种群个体数量
    dim = 10       # 变量维度
    max_iter = 300  # 最大进化代数
    F0 = 0.4       # 初始变异算子
    crossover_rate = 0.1  # 交叉概率
        
    de4 = DE(Np, dim, max_iter, F0, crossover_rate, Rastrigin, lower_bound=-5.12, upper_bound=5.12)
    best_solution, best_value = de4.run()
    print(f"最优解: {np.round(best_solution, 4)}")
    print(f"最优值: {best_value:.4f}")
    plot_history(de4.history)

if __name__ == "__main__":
    main4()