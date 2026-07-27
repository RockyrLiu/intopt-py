import random
import math
import numpy as np
from deap import base, creator, tools
import matplotlib.pyplot as plt
from myalgorithm import eaSimpleWithElitism

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 问题定义：求解Rastrigin函数最小值
def rastrigin(individual):
    """测试函数"""
    n = len(individual)
    return (10 * n + sum(x**2 - 10 * np.cos(2 * math.pi * x) for x in individual),)  # 修正括号问题

def setup_rastrigin_optimization(n_dim=10, bounds=(-5.12, 5.12)):
    """设置Rastrigin函数优化的遗传算法"""
    # 检查并删除已存在的类
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual
    
    # 创建适应度类和个体类
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, bounds[0], bounds[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_float, n=n_dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        return rastrigin(individual)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def optimize_rastrigin():
    """优化10维Rastrigin函数"""
    # 参数设置
    n_dim = 10
    bounds = (-5.12, 5.12)
    pop_size = 50
    n_gen = 1000
    cxpb = 0.6
    mutpb = 0.2
    hof_size = 5
    
    # 初始化
    toolbox = setup_rastrigin_optimization(n_dim, bounds)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(hof_size)
    
    # 统计设置
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # 运行算法
    pop, logbook = eaSimpleWithElitism(pop, toolbox, cxpb, mutpb, n_gen, 
                                     stats=stats, halloffame=hof, verbose=True)
    
    # 结果输出
    best_ind = hof[0]
    print(f"\n最佳解: {best_ind}")
    print(f"最佳适应度: {best_ind.fitness.values[0]}")
    
    # 绘制进化过程
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    
    plt.figure(figsize=(10, 5))
    plt.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution of Minimum Fitness (Rastrigin Function)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    optimize_rastrigin()