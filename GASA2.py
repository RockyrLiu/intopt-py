import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from test_function import TSPProblem1, TSPProblem2  

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# ================== 初始化DEAP类 ==================
# 检查并删除已存在的类
if "FitnessMin" in creator.__dict__:
    del creator.FitnessMin
if "Individual" in creator.__dict__:
    del creator.Individual

# 创建新的DEAP类
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# ================== 遗传模拟退火算法核心 ==================
def simulated_annealing_acceptance(new_fitness, old_fitness, temperature):
    """模拟退火接受准则"""
    if new_fitness < old_fitness:
        return True
    return math.exp((old_fitness - new_fitness) / temperature) > random.random()

def gasa(problem, population_size=100, generations=500, 
         cx_prob=0.8, mut_prob=0.2, initial_temp=1000, cooling_rate=0.999):
    """遗传模拟退火算法主函数"""
    
    # 1. 问题相关设置
    if isinstance(problem, TSPProblem1):
        num_cities = problem.N
        is_problem1 = True
    else:
        num_cities = problem.N  # TSPProblem2中的N已包含起点
        is_problem1 = False
    
    # 2. 初始化工具箱
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_cities), num_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 3. 适应度函数定义
    def evaluate(individual):
        if is_problem1:
            return problem.calculate_path_length(individual),
        else:
            # TSPProblem2需要固定起点在位置0
            full_path = [0] + individual
            return problem.calculate_path_length(full_path),
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOrdered)      # 有序交叉
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # 随机交换变异
    toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择
    
    # 4. 初始化种群和变量
    pop = toolbox.population(n=population_size)
    temperature = initial_temp
    best_fitness = float('inf')
    best_individual = None
    
    # 5. 评估初始种群
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        if fit[0] < best_fitness:
            best_fitness = fit[0]
            best_individual = toolbox.clone(ind)
    
    # 6. 进化主循环
    for gen in range(generations):
        # 选择下一代
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # 交叉操作
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # 重新评估所有适应度值无效的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            # 同时更新全局最优
            if fit[0] < best_fitness:
                best_fitness = fit[0]
                best_individual = toolbox.clone(ind)
        
        # 变异操作 + 模拟退火
        for i in range(len(offspring)):
            if random.random() < mut_prob:
                mutant = offspring[i]
                # 保存原始个体和适应度
                old_mutant = toolbox.clone(mutant)
                old_fitness = mutant.fitness.values[0]
                
                # 执行变异
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
                # 评估新个体
                new_fitness = toolbox.evaluate(mutant)[0]
                
                # 模拟退火接受准则
                if simulated_annealing_acceptance(new_fitness, old_fitness, temperature):
                    mutant.fitness.values = (new_fitness,)
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_individual = toolbox.clone(mutant)
                else:
                    # 拒绝变异，恢复为原来的个体
                    offspring[i] = old_mutant
        
        # 更新种群
        pop[:] = offspring
        
        # 更新温度
        temperature *= cooling_rate
        
        # 打印当前最优解
        if gen % 50 == 0:
            print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}, Temp = {temperature:.2f}")
    
    # 7. 提取最优解
    if is_problem1:
        best_path = best_individual
    else:
        best_path = [0] + best_individual  # TSPProblem2添加起点
    
    return best_path, best_fitness

# ================== 求解TSP问题 ==================
if __name__ == "__main__":
    # 求解TSPProblem1（中国31个城市）
    print("Solving TSPProblem1 (31 cities)...")
    problem1 = TSPProblem1()
    best_path1, best_length1 = gasa(
        problem1,
        population_size=100,
        generations=1000,
        initial_temp=1000,
        cooling_rate=0.99
    )
    print(f"Best path length: {best_length1:.2f}")
    problem1.plot_solution(best_path1, best_length1)
    
    # 求解TSPProblem2（含固定起点）
    print("\nSolving TSPProblem2 (with fixed starting point)...")
    problem2 = TSPProblem2()
    best_path2, best_length2 = gasa(
        problem2,
        population_size=150,
        generations=6000,
        initial_temp=3000,
        cooling_rate=0.999
    )
    print(f"Best path length: {best_length2:.2f} km")
    problem2.plot_solution(best_path2, best_length2)