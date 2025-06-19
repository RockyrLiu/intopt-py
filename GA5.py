from deap import base
from deap import creator
from deap import tools

import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import math
from utils import elitism

# 问题常量:
DIMENSIONS = 2  # 维度数量
BOUND_LOW, BOUND_UP = -5.0, 5.0  # 所有维度的边界值

# 遗传算法常量:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # 交叉概率
P_MUTATION = 0.5   # 个体变异概率
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0  # 交叉和变异的拥挤因子

# 共享机制常量:
DISTANCE_THRESHOLD = 0.1
SHARING_EXTENT = 5.0

# 设置随机种子:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# 定义单目标最大化适应度策略:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 基于列表创建Individual类:
creator.create("Individual", list, fitness=creator.FitnessMax)


# 辅助函数：创建在给定范围[low, up]内均匀分布的随机浮点数
# 假设所有维度的范围相同
def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]

# 创建在指定范围和维度内返回随机浮点数的算子:
toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)

# 创建填充Individual实例的个体算子:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

# 创建生成个体列表的种群算子:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# 将'反转'Himmelblau函数作为个体的适应度:
def himmelblauInverted(individual):
    x = individual[0]
    y = individual[1]
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return 2000.0 - f,  # return a tuple

toolbox.register("evaluate", himmelblauInverted)

# 用适应度共享机制包装tools.selTournament()
# 与tools.selTournament()具有相同签名
def selTournamentWithSharing(individuals, k, tournsize, fit_attr="fitness"):

    # 获取原始适应度:
    origFitnesses = [ind.fitness.values[0] for ind in individuals]

    # 对每个个体应用共享机制:
    for i in range(len(individuals)):
        sharingSum = 1

        # 遍历所有其他个体
        for j in range(len(individuals)):
            if i != j:
                # 计算个体间的欧几里得距离:
                distance = math.sqrt(
                    ((individuals[i][0] - individuals[j][0]) ** 2) + ((individuals[i][1] - individuals[j][1]) ** 2))

                if distance < DISTANCE_THRESHOLD:
                    sharingSum += (1 - distance / (SHARING_EXTENT * DISTANCE_THRESHOLD))

        # 相应减少适应度:
        individuals[i].fitness.values = origFitnesses[i] / sharingSum,

    # 使用修改后的适应度应用原始tools.selTournament():
    selected = tools.selTournament(individuals, k, tournsize, fit_attr)

    # 恢复原始适应度:
    for i, ind in enumerate(individuals):
        ind.fitness.values = origFitnesses[i],

    return selected


toolbox.register("select", selTournamentWithSharing, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/DIMENSIONS)


def main():

    # 创建初始种群(第0代):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # 准备统计对象:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # 定义名人堂对象:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # 执行带精英保留的遗传算法流程:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # 打印找到的最佳解信息:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])

    print("- Best solutions are:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i].fitness.values[0], " -> ", hof.items[i])

    # 在x-y平面上绘制解的位置:
    plt.figure(1)
    globalMaxima = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]
    plt.scatter(*zip(*globalMaxima), marker='x', color='red', zorder=1)
    plt.scatter(*zip(*population), marker='.', color='blue', zorder=0)    # plot solution locations on x-y plane:

    # 在x-y平面上绘制最佳解的位置:
    plt.figure(2)
    plt.scatter(*zip(*globalMaxima), marker='x', color='red', zorder=1)
    plt.scatter(*zip(*hof.items), marker='.', color='blue', zorder=0)

    # 提取统计数据:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # 绘制统计图表:
    plt.figure(3)
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')

    plt.show()


if __name__ == "__main__":
    main()
