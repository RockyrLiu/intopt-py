import random
import numpy as np
from deap import base, creator, tools, algorithms
import math

# 问题定义：求解Rastrigin函数最小值
def rastrigin(individual):
    """Rastrigin函数，多峰优化测试函数"""
    n = len(individual)
    return 10 * n + sum([(x**2 - 10 * np.cos(2 * math.pi * x)) for x in individual]),

# 1. 创建类型
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 2. 初始化工具箱
toolbox = base.Toolbox()

# 定义变量范围
NDIM = 10  # 变量维度
LOW = -5.12
HIGH = 5.12

# 注册函数
toolbox.register("attr_float", random.uniform, LOW, HIGH)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_float, n=NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评价函数
toolbox.register("evaluate", rastrigin)

# 遗传操作
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 3. 遗传模拟退火算法实现
def gasa(population, toolbox, cxpb, mutpb, ngen, t_start=100.0, t_end=0.01, alpha=0.95, stats=None, halloffame=None, verbose=__debug__):
    """遗传模拟退火算法"""
    
    # 初始化统计对象
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # 评价初始种群
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    if halloffame is not None:
        halloffame.update(population)
    
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    
    # 初始温度
    t = t_start
    
    # 开始进化
    for gen in range(1, ngen + 1):
        # 选择下一代个体
        offspring = toolbox.select(population, len(population))
        
        # 克隆选中的个体
        offspring = list(map(toolbox.clone, offspring))
        
        # 应用交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # 模拟退火选择
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # 更新种群
        for i in range(len(population)):
            if offspring[i].fitness > population[i].fitness:
                # 如果子代更优，直接接受
                population[i] = offspring[i]
            else:
                # 否则按Metropolis准则接受
                delta_e = population[i].fitness.values[0] - offspring[i].fitness.values[0]
                p_accept = math.exp(delta_e / t)
                if random.random() < p_accept:
                    population[i] = offspring[i]
        
        # 降温
        t *= alpha
        if t < t_end:
            t = t_end
        
        # 更新Hall of Fame和统计信息
        if halloffame is not None:
            halloffame.update(population)
        
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    
    return population, logbook

# 4. 运行算法
def main():
    random.seed(42)
    
    # 参数设置
    pop_size = 50
    cxpb = 0.6  # 交叉概率
    mutpb = 0.2  # 变异概率
    ngen = 1000  # 迭代次数
    
    # 初始温度参数
    t_start = 100.0
    t_end = 0.01
    alpha = 0.99
    
    # 创建初始种群
    pop = toolbox.population(n=pop_size)
    
    # 设置统计
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # 运行算法
    pop, log = gasa(pop, toolbox, cxpb, mutpb, ngen, 
                   t_start=t_start, t_end=t_end, alpha=alpha,
                   stats=stats, verbose=True)
    
    # 输出结果
    best_ind = tools.selBest(pop, 1)[0]
    print("\n最佳个体:", best_ind)
    print("最佳适应度:", best_ind.fitness.values[0])
    
    # 绘制进化过程
    import matplotlib.pyplot as plt
    gen = log.select("gen")
    fit_mins = log.select("min")
    
    fig, ax = plt.subplots()
    ax.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness over Generations")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()