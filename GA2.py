# 本代码参考Eyal Wirsansky《Hands-On Genetic Algorithms with Python》，91-103
from deap import base
from deap import creator
from deap import tools

import random
import array

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import tsp, elitism

# 设置随机种子以保证结果可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 创建旅行商问题实例：
TSP_NAME = "bayg29"  # name of problem
tsp = tsp.TravelingSalesmanProblem(TSP_NAME)

# 遗传算法常量：
POPULATION_SIZE = 300
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 30
P_CROSSOVER = 0.9  # 交叉概率
P_MUTATION = 0.1   # 个体变异概率

toolbox = base.Toolbox()

# 定义单目标最小化适应度策略：
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# 基于整数列表创建Individual类：
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

# 创建生成随机排列索引的操作符：
toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))

# 创建个体生成操作符，用随机排列的索引填充Individual实例：
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# 创建种群生成操作符来生成个体列表：
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# 适应度计算 - 计算由索引表示的城市列表的总距离：
def tpsDistance(individual):
    return tsp.getTotalDistance(individual),  # return a tuple


toolbox.register("evaluate", tpsDistance)


# 遗传操作符：
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(tsp))


# 遗传算法流程：
def main():

    # 创建初始种群(第0代)：
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # 准备统计对象：
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # 定义名人堂对象：
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # 执行含精英主义的遗传算法流程：
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # 打印最佳个体信息：
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    # 绘制最佳解决方案：
    plt.figure(1)
    tsp.plotData(best)

    # 绘制统计图表：
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # 显示两个图表：
    plt.show()


if __name__ == "__main__":
    main()
