import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_function import Rastrigin

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class BaseGA:
    """遗传算法基类"""
    def __init__(self, obj_func, dim, pop_size=50, generations=100, 
                 crossover_rate=0.8, mutation_rate=0.1, elitism_ratio=0.1, 
                 verbose=True):
        """
        参数:
        obj_func: 目标函数
        dim: 变量维度
        pop_size: 种群大小
        generations: 进化代数
        crossover_rate: 交叉概率
        mutation_rate: 变异概率
        elitism_ratio: 精英保留比例
        verbose: 是否显示进度条
        """
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.elite_size = max(1, int(elitism_ratio * pop_size))
        self.verbose = verbose
        self.history = {'best_fitness': [], 'avg_fitness': []}
    
    def init_population(self):
        """初始化种群"""
        raise NotImplementedError("子类必须实现initialize_population方法")
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        raise NotImplementedError("子类必须实现crossover方法")
    
    def mutate(self, individual):
        """变异操作"""
        raise NotImplementedError("子类必须实现mutate方法")
    
    def decode_individual(self, individual):
        """解码个体"""
        return individual  # 默认不解码
    
    def check_bounds(self, individual):
        """
        检查个体是否在可行域内
        对于连续问题，确保变量在上下界内
        对于TSP问题，确保路径是有效排列
        """
        return individual  # 默认不处理
    
    def select(self, population, fitness):
        """锦标赛选择"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.pop_size):
            candidates = np.random.choice(len(population), tournament_size, replace=False)
            best_index = candidates[np.argmin(fitness[candidates])]
            selected.append(population[best_index])
        
        return np.array(selected)
    
    def standard_iterator(self):
        """标准遗传算法"""
        population = self.init_population()
        fitness = np.array([self.obj_func(ind) for ind in population])
        
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(np.mean(fitness))
        
        if self.verbose:
            pbar = tqdm(range(self.generations), desc="标准GA进化进度")
        
        for gen in range(self.generations):
            # 选择
            population = self.select(population, fitness)
            
            # 交叉
            new_population = []
            np.random.shuffle(population)
            for i in range(0, self.pop_size, 2):
                if i+1 >= self.pop_size:
                    new_population.append(self.check_bounds(population[i]))
                    break
                
                parent1, parent2 = population[i], population[i+1]
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                    # 边界检查
                    child1 = self.check_bounds(child1)
                    child2 = self.check_bounds(child2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                new_population.append(child1)
                new_population.append(child2)
            
            # 变异
            for i in range(len(new_population)):
                if np.random.rand() < self.mutation_rate:
                    mutated = self.mutate(new_population[i])
                    # 边界检查
                    new_population[i] = self.check_bounds(mutated)
            
            population = np.array(new_population)[:self.pop_size]
            
            # 计算适应度
            fitness = np.array([self.obj_func(self.check_bounds(ind)) for ind in population])
            
            # 更新历史记录
            current_best = np.min(fitness)
            current_avg = np.mean(fitness)
            self.history['best_fitness'].append(current_best)
            self.history['avg_fitness'].append(current_avg)
            
            # 更新最优解
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmin(fitness)]
            
            if self.verbose:
                pbar.set_postfix({'最优值': f"{best_fitness:.4f}", '当前最优': f"{current_best:.4f}"})
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return best_individual, best_fitness
    
    def elitism_iterator(self):
        """精英主义遗传算法"""
        population = self.init_population()
        fitness = np.array([self.obj_func(ind) for ind in population])
        
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(np.mean(fitness))
        
        if self.verbose:
            pbar = tqdm(range(self.generations), desc="精英GA进化进度")
        
        for gen in range(self.generations):
            # 选择精英
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elite_population = population[elite_indices]
            elite_fitness = fitness[elite_indices]
            
            # 选择非精英
            non_elite_population = np.delete(population, elite_indices, axis=0)
            non_elite_fitness = np.delete(fitness, elite_indices)
            
            # 对非精英进行选择
            selected_non_elite = self.select(non_elite_population, non_elite_fitness)
            
            # 交叉
            new_population = list(elite_population)
            np.random.shuffle(selected_non_elite)
            for i in range(0, len(selected_non_elite), 2):
                if i+1 >= len(selected_non_elite):
                    new_population.append(self.check_bounds(selected_non_elite[i]))
                    break
                
                parent1, parent2 = selected_non_elite[i], selected_non_elite[i+1]
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                    # 边界检查
                    child1 = self.check_bounds(child1)
                    child2 = self.check_bounds(child2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                new_population.append(child1)
                new_population.append(child2)
            
            # 变异
            for i in range(self.elite_size, len(new_population)):
                if np.random.rand() < self.mutation_rate:
                    mutated = self.mutate(new_population[i])
                    # 边界检查
                    new_population[i] = self.check_bounds(mutated)
            
            population = np.array(new_population)[:self.pop_size]
            
            # 计算适应度
            fitness = np.array([self.obj_func(self.check_bounds(ind)) for ind in population])
            
            # 更新历史记录
            current_best = np.min(fitness)
            current_avg = np.mean(fitness)
            self.history['best_fitness'].append(current_best)
            self.history['avg_fitness'].append(current_avg)
            
            # 更新最优解
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmin(fitness)]
            
            if self.verbose:
                pbar.set_postfix({'最优值': f"{best_fitness:.4f}", '当前最优': f"{current_best:.4f}"})
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return best_individual, best_fitness
    
    def run(self, elitism=False):
        """运行遗传算法"""
        if elitism:
            return self.elitism_iterator()
        else:
            return self.standard_iterator()


class ContinuousGA(BaseGA):
    """连续优化问题的遗传算法"""
    def __init__(self, obj_func, dim, lower_bound, upper_bound, **kwargs):
        super().__init__(obj_func, dim, **kwargs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def init_population(self):
        """初始化种群"""
        return np.random.uniform(
            self.lower_bound, self.upper_bound, 
            (self.pop_size, self.dim)
        )
    
    def check_bounds(self, individual):
        """确保个体在边界范围内"""
        return np.clip(individual, self.lower_bound, self.upper_bound)
    
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


def plot_convergence(history, title='遗传算法收敛曲线'):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['best_fitness'], label='最优适应度')
    plt.plot(history['avg_fitness'], label='平均适应度')
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()


def solve_continuous_problem():
    """解决连续优化问题"""
    print("="*50)
    print("连续优化问题测试")
    print("="*50)
    
    # 基本GA
    ga_cont = ContinuousGA(
        obj_func=Rastrigin, 
        dim=10, 
        lower_bound=-5.12, 
        upper_bound=5.12,
        pop_size=100,
        generations=100,
        verbose=True
    )
    best_solution, best_value = ga_cont.run(elitism=False)
    print(f"基本GA - 最优值: {best_value:.4f}")
    plot_convergence(ga_cont.history, '基本GA收敛曲线')
    
    # 精英GA
    ga_elite = ContinuousGA(
        obj_func=Rastrigin, 
        dim=10, 
        lower_bound=-5.12, 
        upper_bound=5.12,
        pop_size=100,
        generations=100,
        verbose=True
    )
    best_solution, best_value = ga_elite.run(elitism=True)
    print(f"精英GA - 最优值: {best_value:.4f}")
    plot_convergence(ga_elite.history, '精英GA收敛曲线')

if __name__ == "__main__":
    solve_continuous_problem()
