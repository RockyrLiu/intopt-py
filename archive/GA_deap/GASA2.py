import random
import numpy as np
from deap import base, creator, tools
import matplotlib.pyplot as plt
from tsp import problem1
from myalgorithm import gasa_algorithm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def main():
    # ================== 初始化DEAP类 ==================
    # 检查并删除已存在的类
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    # 创建新的DEAP类
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # ================== 求解TSP问题 ==================
    # 求解TSPProblem1（中国31个城市）
    print("Solving TSPProblem1 (31 cities)...")
    
    # 初始化工具箱
    toolbox1 = base.Toolbox()
    toolbox1.register("indices", random.sample, range(problem1.N), problem1.N)
    toolbox1.register("individual", tools.initIterate, creator.Individual, toolbox1.indices)
    toolbox1.register("population", tools.initRepeat, list, toolbox1.individual)
    toolbox1.register("evaluate", lambda ind: (problem1.calculate_path_length(ind),))
    toolbox1.register("mate", tools.cxOrdered)
    toolbox1.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox1.register("select", tools.selTournament, tournsize=3)
    
    # 运行算法
    pop1 = toolbox1.population(n=100)
    hof1 = tools.HallOfFame(1)
    stats1 = tools.Statistics(lambda ind: ind.fitness.values)
    stats1.register("avg", np.mean)
    stats1.register("min", np.min)
    
    pop1, log1 = gasa_algorithm(pop1, toolbox1, cxpb=0.8, mutpb=0.2, ngen=1000,
                              t_start=1000, t_end=0.01, alpha=0.99,
                              stats=stats1, halloffame=hof1, verbose=True)
    
    best_path1 = hof1[0]
    best_length1 = best_path1.fitness.values[0]
    print(f"Best path length: {best_length1:.2f}")
    problem1.plot_solution(best_path1, best_length1)

if __name__ == "__main__":
    main()