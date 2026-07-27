import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class DE:
    """差分进化算法
    
    支持以下变异策略:
    1. 'rand/1/bin' - 经典DE/rand/1/bin策略
    2. 'best/1/bin' - DE/best/1/bin策略
    3. 'rand/2/bin' - DE/rand/2/bin策略
    4. 'best/2/bin' - DE/best/2/bin策略
    5. 'current-to-rand/1' - DE/current-to-rand/1策略
    6. 'current-to-best/1' - DE/current-to-best/1策略
    """
    def __init__(self, func, bounds, popsize=50, maxiter=100, 
                 F=0.5, CR=0.7, strategy='rand/1/bin',
                 init_positions=None, verbose=True, plot=True):
        """
        参数说明:
        func: 目标函数
        bounds: 边界列表，每个元素是元组 (min, max)
        popsize: 种群大小
        maxiter: 最大迭代次数
        F: 缩放因子
        CR: 交叉概率
        strategy: 变异策略(见类文档)
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条
        plot: 是否绘制结果
        """
        self.function = self._wrapper_function(func)
        self.bounds = bounds
        self.dim = len(bounds)
        self.lower_bound = np.array([b[0] for b in bounds])
        self.upper_bound = np.array([b[1] for b in bounds])
        self.popsize = popsize
        self.maxiter = maxiter
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.verbose = verbose
        self.plot = plot
        
        # 初始化种群
        self.population = self._init_population(init_positions)
        self.fitness = self.function(self.population)
        
        # 历史记录
        self.history = {
            'best_fitness': [np.min(self.fitness)],
            'avg_fitness': [np.mean(self.fitness)]
        }
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _wrapper_function(self, func):
        """封装目标函数"""
        def adapted_func(X):
            return np.array([func(x) for x in X])
        return adapted_func
    
    def _init_population(self, init_positions=None):
        """初始化种群"""
        if init_positions is not None:
            if init_positions.shape != (self.popsize, self.dim):
                raise ValueError(f"初始位置形状应为({self.popsize}, {self.dim})")
            return init_positions.copy()
        
        population = np.zeros((self.popsize, self.dim))
        for d in range(self.dim):
            population[:, d] = np.random.uniform(
                self.lower_bound[d], 
                self.upper_bound[d], 
                self.popsize
            )
        return population
    
    def _mutate(self, population):
        """变异操作(支持多种策略)"""
        mutants = np.zeros_like(population)
        current_best_idx = np.argmin(self.fitness)
        current_best = population[current_best_idx]
        
        if self.strategy == 'rand/1/bin':
            # DE/rand/1/bin: v_i = x_r1 + F*(x_r2 - x_r3)
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutants[i] = population[r1] + self.F * (population[r2] - population[r3])
        
        elif self.strategy == 'best/1/bin':
            # DE/best/1/bin: v_i = x_best + F*(x_r1 - x_r2)
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutants[i] = current_best + self.F * (population[r1] - population[r2])
        
        elif self.strategy == 'rand/2/bin':
            # DE/rand/2/bin: v_i = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
                mutants[i] = population[r1] + self.F * (population[r2] - population[r3]) + \
                             self.F * (population[r4] - population[r5])
        
        elif self.strategy == 'best/2/bin':
            # DE/best/2/bin: v_i = x_best + F*(x_r1 - x_r2) + F*(x_r3 - x_r4)
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
                mutants[i] = current_best + self.F * (population[r1] - population[r2]) + \
                             self.F * (population[r3] - population[r4])
        
        elif self.strategy == 'current-to-rand/1':
            # DE/current-to-rand/1: v_i = x_i + K*(x_r1 - x_i) + F*(x_r2 - x_r3)
            K = 0.5  # 通常设置为0.5
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutants[i] = population[i] + K * (population[r1] - population[i]) + \
                             self.F * (population[r2] - population[r3])
        
        elif self.strategy == 'current-to-best/1':
            # DE/current-to-best/1: v_i = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutants[i] = population[i] + self.F * (current_best - population[i]) + \
                             self.F * (population[r1] - population[r2])
        
        else:
            raise ValueError(f"未知变异策略: {self.strategy}")
        
        return mutants
    
    def _crossover(self, population, mutants):
        """交叉操作"""
        trials = np.zeros_like(population)
        for i in range(self.popsize):
            # 确保至少有一个维度来自变异向量
            j_rand = np.random.randint(0, self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == j_rand:
                    trials[i, j] = mutants[i, j]
                else:
                    trials[i, j] = population[i, j]
        return trials
    
    def _select(self, population, trials):
        """选择操作"""
        trial_fitness = self.function(trials)
        new_population = population.copy()
        new_fitness = self.fitness.copy()
        
        for i in range(self.popsize):
            if trial_fitness[i] < self.fitness[i]:
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
        
        return new_population, new_fitness
    
    def _check_bounds(self, vectors):
        """边界处理"""
        clipped = vectors.copy()
        for d in range(self.dim):
            # 越界处理: 随机重置
            mask = (vectors[:, d] < self.lower_bound[d]) | (vectors[:, d] > self.upper_bound[d])
            clipped[mask, d] = np.random.uniform(
                self.lower_bound[d], 
                self.upper_bound[d], 
                np.sum(mask)
            )
        return clipped
    
    def iterator(self):
        """执行优化迭代"""
        if self.verbose:
            pbar = tqdm(total=self.maxiter, desc="DE优化进度")
        
        for _ in range(self.maxiter):
            # 变异
            mutants = self._mutate(self.population)
            
            # 交叉
            trials = self._crossover(self.population, mutants)
            
            # 边界处理
            trials = self._check_bounds(trials)
            
            # 选择
            new_pop, new_fitness = self._select(self.population, trials)
            
            # 更新种群和适应度
            self.population = new_pop
            self.fitness = new_fitness
            
            # 更新最优解
            current_best_idx = np.argmin(self.fitness)
            current_best = self.fitness[current_best_idx]
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = self.population[current_best_idx]
            
            # 记录历史
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(np.mean(self.fitness))
            
            # 更新进度条
            if self.verbose:
                pbar.set_postfix({
                    '最优值': f"{self.best_fitness:.6f}",
                    '平均值': f"{np.mean(self.fitness):.6f}",
                    '策略': self.strategy
                })
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return self.history
    
    def run(self):
        """运行优化"""
        history = self.iterator()
        
        if self.plot:
            self._plot_results(history)
        
        return self.best_solution, history
    
    def _plot_results(self, history):
        """绘制结果"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(history['avg_fitness'], 'r--', label='平均适应度')
        plt.xlabel('迭代次数', size=12)
        plt.ylabel('适应度值', size=12)
        plt.title(f'DE优化过程 ({self.strategy}) - 适应度变化', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class AdaptiveDE(DE):
    """自适应差分进化算法"""
    def __init__(self, func, bounds, popsize=50, maxiter=100, 
                 F=0.5, CR=0.7, strategy='rand/1/bin',
                 init_positions=None, verbose=True, plot=True):
        super().__init__(func, bounds, popsize, maxiter, F, CR, strategy, init_positions, verbose, plot)
    
    def _adaptive_F(self, gen):
        """自适应缩放因子"""
        # 线性递减策略
        return self.F * (1 - gen / self.maxiter)
    
    def iterator(self):
        """执行优化迭代(添加自适应F)"""
        if self.verbose:
            pbar = tqdm(total=self.maxiter, desc="自适应DE优化进度")
        
        for gen in range(self.maxiter):
            # 自适应F
            current_F = self._adaptive_F(gen)
            
            # 变异
            mutants = self._mutate(self.population)
            
            # 交叉
            trials = self._crossover(self.population, mutants)
            
            # 边界处理
            trials = self._check_bounds(trials)
            
            # 选择
            new_pop, new_fitness = self._select(self.population, trials)
            
            # 更新种群和适应度
            self.population = new_pop
            self.fitness = new_fitness
            
            # 更新最优解
            current_best_idx = np.argmin(self.fitness)
            current_best = self.fitness[current_best_idx]
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = self.population[current_best_idx]
            
            # 记录历史
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(np.mean(self.fitness))
            
            # 更新进度条
            if self.verbose:
                pbar.set_postfix({
                    '最优值': f"{self.best_fitness:.6f}",
                    '平均值': f"{np.mean(self.fitness):.6f}",
                    'F值': f"{current_F:.3f}",
                    '策略': self.strategy
                })
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return self.history

class IntDE(DE):
    """整数差分进化算法"""
    def __init__(self, func, bounds, popsize=50, maxiter=100, 
                 F=0.5, CR=0.7, strategy='rand/1/bin',
                 init_positions=None, verbose=True, plot=True):
        super().__init__(func, bounds, popsize, maxiter, F, CR, strategy, init_positions, verbose, plot)
    
    def _init_population(self, init_positions=None):
        """整数初始化"""
        if init_positions is not None:
            if init_positions.shape != (self.popsize, self.dim):
                raise ValueError(f"初始位置形状应为({self.popsize}, {self.dim})")
            return init_positions.astype(int)
        
        population = np.zeros((self.popsize, self.dim), dtype=int)
        for d in range(self.dim):
            population[:, d] = np.random.randint(
                self.lower_bound[d], 
                self.upper_bound[d] + 1, 
                self.popsize
            )
        return population
    
    def _mutate(self, population):
        """整数变异"""
        mutants = np.zeros_like(population)
        current_best_idx = np.argmin(self.fitness)
        current_best = population[current_best_idx]
        
        if self.strategy == 'rand/1/bin':
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutants[i] = population[r1] + np.floor(self.F * (population[r2] - population[r3])).astype(int)
        
        elif self.strategy == 'best/1/bin':
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutants[i] = current_best + np.floor(self.F * (population[r1] - population[r2])).astype(int)
        
        elif self.strategy == 'rand/2/bin':
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
                mutants[i] = population[r1] + np.floor(self.F * (population[r2] - population[r3])).astype(int) + \
                             np.floor(self.F * (population[r4] - population[r5])).astype(int)
        
        elif self.strategy == 'best/2/bin':
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
                mutants[i] = current_best + np.floor(self.F * (population[r1] - population[r2])).astype(int) + \
                             np.floor(self.F * (population[r3] - population[r4])).astype(int)
        
        elif self.strategy == 'current-to-rand/1':
            K = 0.5
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutants[i] = population[i] + np.floor(K * (population[r1] - population[i])).astype(int) + \
                             np.floor(self.F * (population[r2] - population[r3])).astype(int)
        
        elif self.strategy == 'current-to-best/1':
            for i in range(self.popsize):
                candidates = [idx for idx in range(self.popsize) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutants[i] = population[i] + np.floor(self.F * (current_best - population[i])).astype(int) + \
                             np.floor(self.F * (population[r1] - population[r2])).astype(int)
        
        else:
            raise ValueError(f"未知变异策略: {self.strategy}")
        
        return mutants
    
    def _check_bounds(self, vectors):
        """整数边界处理"""
        return np.clip(vectors, self.lower_bound, self.upper_bound)


def de(func, bounds, popsize=50, maxiter=100, 
       F=0.5, CR=0.7, strategy='rand/1/bin',
       init_positions=None, verbose=True, plot=True):
    """
    差分进化算法(连续优化问题)
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        popsize: 种群大小 (默认50)
        maxiter: 最大迭代次数 (默认100)
        F: 缩放因子 (默认0.5)
        CR: 交叉概率 (默认0.7)
        strategy: 变异策略 (默认'rand/1/bin')
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 包含'best_fitness'和'avg_fitness'的历史记录
    """

    optimizer = DE(
        func=func,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        F=F,
        CR=CR,
        strategy=strategy,
        init_positions=init_positions,
        verbose=verbose,
        plot=plot
    )
    return optimizer.run()

def adaptive_de(func, bounds, popsize=50, maxiter=100, 
                F=0.5, CR=0.7, strategy='rand/1/bin',
                init_positions=None, verbose=True, plot=True):
    """
    自适应差分进化算法(连续优化问题)
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        popsize: 种群大小 (默认50)
        maxiter: 最大迭代次数 (默认100)
        F: 缩放因子 (默认0.5)
        CR: 交叉概率 (默认0.7)
        strategy: 变异策略 (默认'rand/1/bin')
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 包含'best_fitness'和'avg_fitness'的历史记录
    """

    optimizer = AdaptiveDE(
        func=func,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        F=F,
        CR=CR,
        strategy=strategy,
        init_positions=init_positions,
        verbose=verbose,
        plot=plot
    )
    return optimizer.run()

def int_de(func, bounds, popsize=50, maxiter=100, 
           F=0.5, CR=0.7, strategy='rand/1/bin',
           init_positions=None, verbose=True, plot=True):
    """
    差分进化算法(整数优化问题)
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        popsize: 种群大小 (默认50)
        maxiter: 最大迭代次数 (默认100)
        F: 缩放因子 (默认0.5)
        CR: 交叉概率 (默认0.7)
        strategy: 变异策略 (默认'rand/1/bin')
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 包含'best_fitness'和'avg_fitness'的历史记录
    """

    optimizer = IntDE(
        func=func,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        F=F,
        CR=CR,
        strategy=strategy,
        init_positions=init_positions,
        verbose=verbose,
        plot=plot
    )
    return optimizer.run()


def main():
    # 测试函数
    def sphere(X):
        return X[0] ** 2 + X[1] ** 2 + X[2] ** 2
    
    # 测试所有变异策略
    strategies = [
        'rand/1/bin',
        'best/1/bin',
        'rand/2/bin',
        'best/2/bin',
        'current-to-rand/1',
        'current-to-best/1'
    ]
    
    # 标准DE测试
    print("标准DE测试:")
    for strategy in strategies:
        print(f"\n策略: {strategy}")
        best_solution, history = de(
            func=sphere,
            bounds=[(-5, 5), (-5, 5), (-5, 5)],
            popsize=30,
            maxiter=100,
            strategy=strategy,
            verbose=True
        )
        print("最优解:", best_solution)
        print("最优值:", history['best_fitness'][-1])
    
    # 自适应DE测试
    print("\n自适应DE测试:")
    best_solution, history = adaptive_de(
        func=sphere,
        bounds=[(-5, 5), (-5, 5), (-5, 5)],
        popsize=30,
        maxiter=100,
        strategy='best/1/bin',
        verbose=True
    )
    print("最优解:", best_solution)
    print("最优值:", history['best_fitness'][-1])
    
    # 整数DE测试
    print("\n整数DE测试:")
    # 生成初始位置
    init_pos = np.random.randint(-5, 6, size=(30, 3))
    best_solution, history = int_de(
        func=sphere,
        bounds=[(-5, 5), (-5, 5), (-5, 5)],
        popsize=30,
        maxiter=100,
        strategy='rand/1/bin',
        init_positions=init_pos,
        verbose=True
    )
    print("最优解:", best_solution)
    print("最优值:", history['best_fitness'][-1])

if __name__ == "__main__":
    main()