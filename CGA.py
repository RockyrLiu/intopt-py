import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from GA1 import BaseGA, plot_convergence
from test_function import Rastrigin, TSPProblem1, TSPProblem2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class ChaosContinuousGA(BaseGA):
    """混沌优化的遗传算法(连续问题)"""
    def __init__(self, obj_func, dim, lower_bound, upper_bound, **kwargs):
        """
        参数:
        obj_func: 目标函数
        dim: 变量维度
        lower_bound: 下界
        upper_bound: 上界
        """
        super().__init__(obj_func, dim, **kwargs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def _generate_chaos_sequence(self, length):
        """生成混沌序列"""
        sequence = np.zeros(length)
        sequence[0] = np.random.rand()
        for i in range(1, length):
            sequence[i] = 4 * sequence[i-1] * (1 - sequence[i-1])
        return sequence
    
    def init_population(self):
        """使用混沌序列初始化种群"""
        ch = self._generate_chaos_sequence(self.pop_size * self.dim)
        ch = ch.reshape(self.pop_size, self.dim)
        return self.lower_bound + ch * (self.upper_bound - self.lower_bound)
    
    def crossover(self, parent1, parent2):
        """混沌交叉操作"""
        alpha = self._generate_chaos_sequence(self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    
    def mutate(self, individual):
        """混沌变异操作"""
        ch = self._generate_chaos_sequence(self.dim)
        mutation_strength = 0.1 * (self.upper_bound - self.lower_bound)
        mutated = individual + mutation_strength * (ch - 0.5) * 2
        return mutated
    
    def check_bounds(self, individual):
        """边界检查"""
        return np.clip(individual, self.lower_bound, self.upper_bound)

class ChaosTSPGA(BaseGA):
    """混沌优化的遗传算法(TSP问题)"""
    def __init__(self, tsp_problem, **kwargs):
        """
        参数:
        tsp_problem: TSP问题实例
        """
        self.tsp_problem = tsp_problem
        self.num_cities = tsp_problem.N
        super().__init__(self._tsp_objective, self.num_cities, **kwargs)
    
    def _tsp_objective(self, path):
        path = self.check_bounds(path)
        return self.tsp_problem.calculate_path_length(path)
    
    def _generate_chaos_sequence(self, length):
        """生成混沌序列"""
        sequence = np.zeros(length)
        sequence[0] = np.random.rand()
        for i in range(1, length):
            sequence[i] = 4 * sequence[i-1] * (1 - sequence[i-1])
        return sequence
    
    def init_population(self):
        """使用混沌序列初始化种群"""
        pop = []
        for _ in range(self.pop_size):
            ch = self._generate_chaos_sequence(self.num_cities)
            individual = np.argsort(ch)
            pop.append(individual)
        return np.array(pop)
    
    def crossover(self, parent1, parent2):
        """混沌顺序交叉(OX)"""
        parent1 = parent1.astype(int)
        parent2 = parent2.astype(int)
        size = len(parent1)
        
        # 使用混沌序列确定交叉点
        ch = self._generate_chaos_sequence(2)
        cx1, cx2 = sorted(np.floor(ch * size).astype(int))
        cx1 = np.clip(cx1, 0, size-2)
        cx2 = np.clip(cx2, cx1+1, size-1)
        
        child1 = np.full(size, -1, dtype=int)
        child2 = np.full(size, -1, dtype=int)
        
        child1[cx1:cx2+1] = parent1[cx1:cx2+1]
        child2[cx1:cx2+1] = parent2[cx1:cx2+1]
        
        current_index = (cx2 + 1) % size
        for p in [parent2, parent1]:
            for i in range(size):
                index = (i + cx2 + 1) % size
                city = p[index]
                if city not in child1:
                    child1[current_index] = city
                    current_index = (current_index + 1) % size
        
        current_index = (cx2 + 1) % size
        for p in [parent1, parent2]:
            for i in range(size):
                index = (i + cx2 + 1) % size
                city = p[index]
                if city not in child2:
                    child2[current_index] = city
                    current_index = (current_index + 1) % size
        
        return child1, child2
    
    def mutate(self, individual):
        """混沌交换变异"""
        individual = individual.astype(int)
        ch = self._generate_chaos_sequence(2)
        idx1, idx2 = np.floor(ch * len(individual)).astype(int)
        idx1 = np.clip(idx1, 0, len(individual)-1)
        idx2 = np.clip(idx2, 0, len(individual)-1)
        mutated = individual.copy()
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    def check_bounds(self, path):
        """确保路径是有效的排列"""
        try:
            path = path.astype(int)
        except:
            return np.random.permutation(self.num_cities)
        
        if len(path) != self.num_cities:
            return np.random.permutation(self.num_cities)
        
        if len(np.unique(path)) != self.num_cities:
            return np.random.permutation(self.num_cities)
        
        if not np.array_equal(np.sort(path), np.arange(self.num_cities)):
            return np.random.permutation(self.num_cities)
        
        return path

def solve_continuous_problem():
    """解决连续优化问题"""
    print("="*50)
    print("连续优化问题测试(混沌GA)")
    print("="*50)
    
    # 混沌GA
    ga_chaos = ChaosContinuousGA(
        obj_func=Rastrigin, 
        dim=10, 
        lower_bound=-5.12, 
        upper_bound=5.12,
        pop_size=50,
        generations=1000,
        verbose=True
    )
    best_solution, best_value = ga_chaos.run(elitism=True)
    print(f"混沌GA - 最优值: {best_value:.4f}")
    plot_convergence(ga_chaos.history, '混沌GA收敛曲线(连续问题)')

def solve_tsp_problem1():
    """解决TSP问题1"""
    print("\n" + "="*50)
    print("TSP问题1测试(混沌GA)")
    print("="*50)
    
    tsp_problem = TSPProblem1()
    
    # 混沌TSP GA
    ga_tsp_chaos = ChaosTSPGA(
        tsp_problem=tsp_problem,
        pop_size=100,
        generations=200,
        verbose=True
    )
    best_path, best_length = ga_tsp_chaos.run(elitism=True)
    print(f"混沌TSP GA - 最短路径长度: {best_length:.2f}")
    tsp_problem.plot_solution(best_path, best_length)
    plot_convergence(ga_tsp_chaos.history, '混沌TSP GA收敛曲线(问题1)')

def solve_tsp_problem2():
    """解决TSP问题2"""
    print("\n" + "="*50)
    print("TSP问题2测试(混沌GA)")
    print("="*50)
    
    tsp_problem = TSPProblem2()
    
    # 混沌TSP GA
    ga_tsp_chaos = ChaosTSPGA(
        tsp_problem=tsp_problem,
        pop_size=100,
        generations=200,
        verbose=True
    )
    best_path, best_length = ga_tsp_chaos.run(elitism=True)
    print(f"混沌TSP GA - 最短路径长度: {best_length:.2f}")
    tsp_problem.plot_solution(best_path, best_length)
    plot_convergence(ga_tsp_chaos.history, '混沌TSP GA收敛曲线(问题2)')

if __name__ == "__main__":
    solve_continuous_problem()
    solve_tsp_problem1()
    solve_tsp_problem2()