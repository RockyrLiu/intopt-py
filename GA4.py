# tsp_optimization.py
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from test_function import TSPProblem1, TSPProblem2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置随机种子
random.seed(42)
np.random.seed(42)

def setup_tsp_optimization(tsp_problem):
    """设置TSP问题优化的遗传算法"""
    
    # 创建适应度类和个体类
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # 定义个体生成函数
    def create_individual():
        ind = list(range(tsp_problem.N))
        random.shuffle(ind)
        return ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 定义评估函数
    def evaluate(individual):
        return tsp_problem.calculate_path_length(individual),
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOrdered)  # 有序交叉
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # 索引洗牌变异
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def optimize_tsp(tsp_problem, pop_size=100, n_gen=500, cxpb=0.8, mutpb=0.2):
    """优化TSP问题"""
    toolbox = setup_tsp_optimization(tsp_problem)
    
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
    
    print(f"\n最佳路径: {best_individual}")
    print(f"最短距离: {best_fitness}")
    
    # 绘制结果
    tsp_problem.plot_solution(best_individual, best_fitness)
    
    # 绘制进化过程
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    
    plt.figure(figsize=(10, 5))
    plt.plot(gen, fit_mins, "b-", label="Minimum Distance")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Evolution of Minimum Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("优化TSP问题1...")
    tsp1 = TSPProblem1()
    optimize_tsp(tsp1, pop_size=150, n_gen=800)
    
    print("\n优化TSP问题2...")
    tsp2 = TSPProblem2()
    optimize_tsp(tsp2, pop_size=200, n_gen=1000)