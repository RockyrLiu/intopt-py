# GA4.py
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from tsp import problem1
from myalgorithm import eaSimpleWithElitism

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def setup_tsp_optimization(tsp_problem):
    """设置TSP问题优化的遗传算法"""
    # 检查并删除已存在的类
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual
    
    # 创建适应度类和个体类
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    def create_individual():
        ind = list(range(tsp_problem.N))
        random.shuffle(ind)
        return ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        return tsp_problem.calculate_path_length(individual),
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def optimize_tsp(tsp_problem, pop_size=100, n_gen=500, cxpb=0.8, mutpb=0.2):
    """优化TSP问题"""
    # 初始化
    toolbox = setup_tsp_optimization(tsp_problem)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
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
    print(f"\n最佳路径: {best_ind}")
    print(f"最短距离: {best_ind.fitness.values[0]}")
    
    # 绘制结果
    tsp_problem.plot_solution(best_ind, best_ind.fitness.values[0])
    
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
    random.seed(42)
    np.random.seed(42)
    
    print("优化TSP问题1...")
    tsp1 = problem1
    optimize_tsp(tsp1, pop_size=150, n_gen=800)