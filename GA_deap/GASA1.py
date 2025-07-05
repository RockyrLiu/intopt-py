import random
import numpy as np
from deap import base, creator, tools
import math
from myalgorithm import gasa_algorithm

# 问题定义：求解Rastrigin函数最小值
def rastrigin(individual):
    """测试函数"""
    n = len(individual)
    return 10 * n + sum([(x**2 - 10 * np.cos(2 * math.pi * x)) for x in individual]),

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
    # 创建初始种群
    pop = toolbox.population(n=pop_size)
    
    # 设置统计
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # 创建Hall of Fame
    hof = tools.HallOfFame(1)
    
    # 运行算法
    pop, log = gasa_algorithm(pop, toolbox, cxpb, mutpb, ngen, 
                            t_start=t_start, t_end=t_end, alpha=alpha,
                            stats=stats, halloffame=hof, verbose=True)
    
    # 输出结果
    best_ind = hof[0]
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