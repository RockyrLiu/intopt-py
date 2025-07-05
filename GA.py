import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tsp import problem1

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GA:
    """遗传算法基类"""
    def __init__(self, func, bounds=None, pop_size=50, generations=100, 
                 crossover_rate=0.8, mutation_rate=0.2, elitism_ratio=0.1,
                 verbose=True):
        """
        func: 目标函数
        bounds: 变量边界列表，每个元素是元组(min,max)(连续问题)或None(离散问题)
        pop_size: 种群大小
        generations: 进化代数
        crossover_rate: 交叉概率
        mutation_rate: 变异概率
        elitism_ratio: 精英保留比例
        verbose: 是否显示进度条
        """
        self.func = func
        self.bounds = bounds
        self.dim = len(bounds) if bounds is not None else None
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.elite_size = max(1, int(elitism_ratio * pop_size))
        self.verbose = verbose
        
        # 提取边界(连续问题)
        if bounds is not None:
            self.lower_bound = np.array([b[0] for b in bounds])
            self.upper_bound = np.array([b[1] for b in bounds])
        else:
            self.lower_bound = None
            self.upper_bound = None
            
        # 历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_solution': []
        }
    
    def init_population(self):
        """初始化种群"""
        raise NotImplementedError("子类必须实现init_population方法")
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        raise NotImplementedError("子类必须实现crossover方法")
    
    def mutate(self, individual):
        """变异操作"""
        raise NotImplementedError("子类必须实现mutate方法")
    
    def check_bounds(self, individual):
        """边界检查"""
        return individual  # 默认实现
    
    def select(self, population, fitness):
        """锦标赛选择"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.pop_size):
            candidates = np.random.choice(len(population), tournament_size, replace=False)
            best_index = candidates[np.argmin(fitness[candidates])]
            selected.append(population[best_index])
        
        return np.array(selected)
    
    def _evolve(self, elite_pop, selected):
        """进化操作(精英保留)"""
        new_pop = list(elite_pop)
        np.random.shuffle(selected)
        
        for i in range(0, len(selected), 2):
            if i+1 >= len(selected):
                new_pop.append(self.check_bounds(selected[i]))
                break
            
            parent1, parent2 = selected[i], selected[i+1]
            if np.random.rand() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.check_bounds(child1)
                child2 = self.check_bounds(child2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            new_pop.append(child1)
            new_pop.append(child2)
        
        # 变异(不变异精英个体)
        for i in range(self.elite_size, len(new_pop)):
            if np.random.rand() < self.mutation_rate:
                mutated = self.mutate(new_pop[i])
                new_pop[i] = self.check_bounds(mutated)
        
        return new_pop
    
    def _record_history(self, best_fitness, avg_fitness, best_solution):
        """记录历史数据"""
        self.history['best_fitness'].append(float(best_fitness))
        self.history['avg_fitness'].append(float(avg_fitness))
        self.history['best_solution'].append(best_solution.copy())

    def run(self):
        """运行遗传算法"""
        population = self.init_population()
        fitness = np.array([self.func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        self._record_history(best_fitness, np.mean(fitness), best_solution)
        
        if self.verbose:
            pbar = tqdm(range(self.generations), desc="GA进化进度")
        
        for gen in range(self.generations):
            # 精英保留
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elite_pop = population[elite_indices]
            
            # 选择
            selected = self.select(population, fitness)
            
            # 交叉和变异
            new_pop = self._evolve(elite_pop, selected)
            
            population = np.array(new_pop)[:self.pop_size]
            fitness = np.array([self.func(ind) for ind in population])
            
            # 更新最优解
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx]
                best_fitness = fitness[current_best_idx]
            
            self._record_history(best_fitness, np.mean(fitness), best_solution)
            
            if self.verbose:
                pbar.set_postfix({'最优值': f"{best_fitness:.4f}"})
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return best_solution, best_fitness

class ContinuousGA(GA):
    """连续优化问题的遗传算法"""
    def init_population(self):
        """初始化种群"""
        return np.random.uniform(
            self.lower_bound, self.upper_bound, 
            (self.pop_size, self.dim)
        )
    
    def crossover(self, parent1, parent2):
        """算术交叉"""
        alpha = np.random.rand(self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    
    def mutate(self, individual):
        """高斯变异"""
        mutation_strength = 0.1 * (self.upper_bound - self.lower_bound)
        mutated = individual + mutation_strength * np.random.randn(self.dim)
        return mutated
    
    def check_bounds(self, individual):
        """边界检查"""
        return np.clip(individual, self.lower_bound, self.upper_bound)

class ChaosContinuousGA(ContinuousGA):
    """混沌优化的连续遗传算法"""
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
        """混沌交叉"""
        alpha = self._generate_chaos_sequence(self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    
    def mutate(self, individual):
        """混沌变异"""
        ch = self._generate_chaos_sequence(self.dim)
        mutation_strength = 0.1 * (self.upper_bound - self.lower_bound)
        mutated = individual + mutation_strength * (ch - 0.5) * 2
        return mutated

class TSP_GA(GA):
    """TSP问题的遗传算法"""
    def __init__(self, tsp_problem, **kwargs):
        self.tsp_problem = tsp_problem
        self.num_cities = tsp_problem.N
        super().__init__(self._tsp_objective, bounds=None, **kwargs)
    
    def _tsp_objective(self, path):
        path = self.check_bounds(path)
        return self.tsp_problem.calculate_path_length(path)
    
    def init_population(self):
        """初始化种群"""
        pop = []
        for _ in range(self.pop_size):
            individual = np.random.permutation(self.num_cities)
            pop.append(individual)
        return np.array(pop)
    
    def crossover(self, parent1, parent2):
        """顺序交叉(OX)"""
        parent1 = parent1.astype(int)
        parent2 = parent2.astype(int)
        size = len(parent1)
        cx1, cx2 = sorted(np.random.choice(size, 2, replace=False))
        
        child1 = np.full(size, -1, dtype=int)
        child2 = np.full(size, -1, dtype=int)
        
        child1[cx1:cx2+1] = parent1[cx1:cx2+1]
        child2[cx1:cx2+1] = parent2[cx1:cx2+1]
        
        self._fill_child(child1, parent2, cx1, cx2)
        self._fill_child(child2, parent1, cx1, cx2)
        
        return child1, child2
    
    def _fill_child(self, child, parent, cx1, cx2):
        """填充子代剩余城市"""
        size = len(child)
        current_pos = (cx2 + 1) % size
        for i in range(size):
            city = parent[(cx2 + 1 + i) % size]
            if city not in child:
                child[current_pos] = city
                current_pos = (current_pos + 1) % size
    
    def mutate(self, individual):
        """交换变异"""
        individual = individual.astype(int)
        idx1, idx2 = np.random.choice(self.num_cities, 2, replace=False)
        mutated = individual.copy()
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    def check_bounds(self, path):
        """确保路径是有效的排列"""
        try:
            path = path.astype(int)
            if (len(path) != self.num_cities or 
                len(np.unique(path)) != self.num_cities or
                not np.array_equal(np.sort(path), np.arange(self.num_cities))):
                return np.random.permutation(self.num_cities)
            return path
        except:
            return np.random.permutation(self.num_cities)

class ChaosTSP_GA(TSP_GA):
    """混沌优化的TSP遗传算法"""
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
        """混沌顺序交叉"""
        parent1 = parent1.astype(int)
        parent2 = parent2.astype(int)
        size = len(parent1)
        
        ch = self._generate_chaos_sequence(2)
        cx1, cx2 = sorted(np.floor(ch * size).astype(int))
        cx1 = np.clip(cx1, 0, size-2)
        cx2 = np.clip(cx2, cx1+1, size-1)
        
        child1 = np.full(size, -1, dtype=int)
        child2 = np.full(size, -1, dtype=int)
        
        child1[cx1:cx2+1] = parent1[cx1:cx2+1]
        child2[cx1:cx2+1] = parent2[cx1:cx2+1]
        
        self._fill_child(child1, parent2, cx1, cx2)
        self._fill_child(child2, parent1, cx1, cx2)
        
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

def plot_convergence(history, title='遗传算法收敛曲线'):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['best_fitness'], label='最优适应度')
    plt.plot(history['avg_fitness'], label='平均适应度')
    plt.title(title)
    plt.xlabel('进化代数')
    plt.ylabel('适应度值')
    plt.legend()
    plt.grid(True)
    plt.show()

def ga(func, bounds, pop_size=50, generations=100, crossover_rate=0.8, 
       mutation_rate=0.2, elitism_ratio=0.1, use_chaos=False, 
       verbose=True, plot=True):
    """
    遗传算法接口(连续优化问题)
    
    参数:
    func: 目标函数
    bounds: 变量边界列表，每个元素是元组(min,max)
    pop_size: 种群大小
    generations: 进化代数
    crossover_rate: 交叉概率
    mutation_rate: 变异概率
    elitism_ratio: 精英保留比例
    use_chaos: 是否使用混沌优化
    verbose: 是否显示进度条
    plot: 是否绘制收敛曲线
    
    返回:
    best_solution: 最优解
    best_fitness: 最优适应度
    history: 历史记录
    """
    GA = ChaosContinuousGA if use_chaos else ContinuousGA
    optimizer = GA(
        func=func,
        bounds=bounds,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_ratio=elitism_ratio,
        verbose=verbose
    )
    
    best_solution, best_fitness = optimizer.run()
    
    if plot:
        title = '混沌GA收敛曲线' if use_chaos else '标准GA收敛曲线'
        plot_convergence(optimizer.history, title)
    
    return best_solution, best_fitness, optimizer.history

def ga_tsp(tsp_problem, pop_size=100, generations=200, crossover_rate=0.8,
           mutation_rate=0.2, elitism_ratio=0.1, use_chaos=False,
           verbose=True, plot=True):
    """
    遗传算法接口(TSP问题)
    
    参数:
    tsp_problem: TSP问题实例
    pop_size: 种群大小
    generations: 进化代数
    crossover_rate: 交叉概率
    mutation_rate: 变异概率
    elitism_ratio: 精英保留比例
    use_chaos: 是否使用混沌优化
    verbose: 是否显示进度条
    plot: 是否绘制收敛曲线和路径
    
    返回:
    best_path: 最优路径
    best_length: 最短路径长度
    history: 历史记录
    """
    GA = ChaosTSP_GA if use_chaos else TSP_GA
    optimizer = GA(
        tsp_problem=tsp_problem,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_ratio=elitism_ratio,
        verbose=verbose
    )
    
    best_path, best_length = optimizer.run()
    
    if plot:
        title = '混沌TSP GA收敛曲线' if use_chaos else '标准TSP GA收敛曲线'
        plot_convergence(optimizer.history, title)
        tsp_problem.plot_solution(best_path, best_length)
    
    return best_path, best_length, optimizer.history

def main1():
    """连续优化问题测试"""
    def sphere(X):
        return np.sum(X ** 2)  # 更简洁的球函数实现
    
    dim = 10
    bounds = [(-5.12, 5.12)] * dim
    
    print("标准GA测试:")
    solution, fitness, _ = ga(
        func=sphere,
        bounds=bounds,
        pop_size=50,
        generations=1000,
        use_chaos=False,
        verbose=True
    )
    print(f"最优值: {fitness:.4f}")
    print(f"最优解: {solution}")
    
    print("\n混沌GA测试:")
    solution, fitness, _ = ga(
        func=sphere,
        bounds=bounds,
        pop_size=50,
        generations=1000,
        use_chaos=True,
        verbose=True
    )
    print(f"最优值: {fitness:.4f}")
    print(f"最优解: {solution}")

def main2():
    """TSP问题测试"""
    tsp_problem = problem1
    
    print("\n标准TSP_GA测试:")
    path, length, _ = ga_tsp(
        tsp_problem=tsp_problem,
        pop_size=100,
        generations=200,
        use_chaos=False,
        verbose=True
    )
    print(f"最短路径长度: {length:.2f}")
    print(f"最优路径: {path}")
    
    print("\n混沌TSP_GA测试:")
    path, length, _ = ga_tsp(
        tsp_problem=tsp_problem,
        pop_size=100,
        generations=200,
        use_chaos=True,
        verbose=True
    )
    print(f"最短路径长度: {length:.2f}")
    print(f"最优路径: {path}")

if __name__ == "__main__":
    main1()
    main2()