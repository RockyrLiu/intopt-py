# 本代码参考Eyal Wirsansky《Hands-On Genetic Algorithms with Python》，165-170
from deap import base
from deap import creator
from deap import tools

import random
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns

from utils import elitism

# 问题常量：
DIMENSIONS = 2  # 维度数量
BOUND_LOW, BOUND_UP = -1.25, 1.25  # 所有维度的边界值

# 遗传算法常量：
POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # 交叉概率
P_MUTATION = 0.5  #0.1   # (也可尝试0.5) 变异概率
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0  # 交叉和变异的拥挤因子
PENALTY_VALUE = 10.0    # 违反约束的固定惩罚值
DISTANCE_THRESHOLD = 0.1

# 设置随机种子：
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# 定义单目标最小化适应度策略：
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# 基于列表创建Individual类：
creator.create("Individual", list, fitness=creator.FitnessMin)


# 辅助函数：创建在给定范围[low, up]内均匀分布的随机实数
# 假设每个维度的范围相同
def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]

# 创建操作符，随机返回所需范围和维度的浮点数：
toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)

# 创建个体操作符来填充Individual实例：
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

# 创建种群操作符来生成个体列表：
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# Simionescu函数作为给定个体的适应度：
def simionescu(individual):
    x = individual[0]
    y = individual[1]
    f = 0.1 * x * y
    return f,  # 返回元组

toolbox.register("evaluate", simionescu)

# 使用约束定义有效输入域：
def feasible(individual):
    """个体的可行域函数。
    如果可行返回True，否则返回False。
    """
    x = individual[0]
    y = individual[1]

    # 原始约束：
    if x**2 + y**2 > (1 + 0.2 * math.cos(8.0 * math.atan2(x, y)))**2:
        return False

    # 之前找到的解作为附加约束：
    elif (x - 0.848)**2 + (y + 0.848)**2 < DISTANCE_THRESHOLD**2:
        return False

    else:
        return True

# 用delta惩罚函数装饰适应度函数：
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, PENALTY_VALUE))

# 遗传操作符：
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/DIMENSIONS)


# 遗传算法流程：
def main():

    # 创建初始种群（第0代）：
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # 准备统计对象：
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # 定义名人堂对象：
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # 使用精英保留策略执行遗传算法流程：
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # 打印找到的最佳解信息：
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])

    # 提取统计信息：
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # 绘制统计图：
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    plt.show()


if __name__ == "__main__":
    main()
