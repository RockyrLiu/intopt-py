import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from test_function import Rastrigin

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置随机种子
random.seed(42)
np.random.seed(42)

def setup_rastrigin_optimization(n_dim=10, bounds=(-5.12, 5.12)):
    """设置Rastrigin函数优化的遗传算法"""
    
    # 创建适应度类和个体类
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # 定义属性生成函数
    toolbox.register("attr_float", random.uniform, bounds[0], bounds[1])
    
    # 定义个体和种群生成函数
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_float, n=n_dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 定义评估函数
    def evaluate(individual):
        return Rastrigin(np.array(individual)),
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def optimize_rastrigin():
    """优化10维Rastrigin函数"""
    n_dim = 10
    bounds = (-5.12, 5.12)
    pop_size = 50
    n_gen = 1000
    cxpb = 0.6
    mutpb = 0.2
    
    toolbox = setup_rastrigin_optimization(n_dim, bounds)
    
    # 创建初始种群
    pop = toolbox.population(n=pop_size)
    
    # 注册统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # 运行算法
    result, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                         ngen=n_gen, stats=stats, verbose=True)
    
    # 获取最佳个体
    best_individual = tools.selBest(result, k=1)[0]
    best_fitness = best_individual.fitness.values[0]
    
    print(f"\n最佳解: {best_individual}")
    print(f"最佳适应度: {best_fitness}")
    
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
    optimize_rastrigin()