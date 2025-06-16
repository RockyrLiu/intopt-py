# 当前代码中的所有约束均为闭区间，若想要约束到开区间需要自行修改 初始个体范围的逻辑、
# main1中边界感知算子的逻辑、main2main3中装饰器的逻辑

import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm
import multiprocessing

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义单目标最小化适应度策略
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_toolbox(objective_function, bounds):
    """创建工具箱并注册各种操作"""
    toolbox = base.Toolbox()
    dimensions = len(bounds)
    
    # 注册属性生成器（每个变量独立生成）
    for i, (low, up) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, low, up)
    
    # 注册个体创建器（组合所有属性）
    attributes = [getattr(toolbox, f"attr_{i}") for i in range(dimensions)]
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                    attributes, n=1)
    
    # 注册种群创建器
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 注册评估函数
    toolbox.register("evaluate", objective_function)
    
    # 注册选择算子（锦标赛选择）
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # 因为下面的交叉和变异操作限定了边界，所以无需担心自变量操作后超出范围
    # 注册交叉算子（模拟二进制交叉）
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                    low=[b[0] for b in bounds], 
                    up=[b[1] for b in bounds], 
                    eta=20.0)
    
    # 注册变异算子（多项式变异）
    toolbox.register("mutate", tools.mutPolynomialBounded,
                    low=[b[0] for b in bounds], 
                    up=[b[1] for b in bounds], 
                    eta=20.0, 
                    indpb=1.0/dimensions)
    
    return toolbox

def gaWithElitism(toolbox, pop_size=100, max_gen=200, 
           cx_prob=0.8, mut_prob=0.2, elite_size=10, verbose=True):
    """
    精英主义遗传算法
    
    参数:
    toolbox: 工具箱
    pop_size: 种群大小
    max_gen: 最大迭代次数
    cx_prob: 交叉概率
    mut_prob: 变异概率
    elite_size: 精英保留数量
    verbose: 是否显示进度条
    
    返回:
    population: 最终种群
    logbook: 统计日志
    hall_of_fame: 名人堂（精英集合）
    """
    
    
    # 创建初始种群
    population = toolbox.population(n=pop_size)
    
    # 创建统计对象
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)
    
    # 创建名人堂（精英保留）
    hall_of_fame = tools.HallOfFame(elite_size)
    
    # 创建日志
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields
    
    # 启用多核并行评估
    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        
        # 评估初始种群
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # 更新名人堂
        hall_of_fame.update(population)
        
        # 记录初始统计
        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        # 进度条设置
        if verbose:
            pbar = tqdm(total=max_gen, desc="进化进度", unit="gen")
        
        # 遗传算法主循环
        for gen in range(1, max_gen + 1):
            # 选择下一代个体（精英保留）
            offspring = toolbox.select(population, len(population) - elite_size)
            
            # 克隆选中的个体
            offspring = list(map(toolbox.clone, offspring))
            
            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # 变异
            for mutant in offspring:
                if random.random() < mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 添加精英个体
            offspring.extend(hall_of_fame.items)
            
            # 更新种群
            population[:] = offspring
            
            # 更新名人堂
            hall_of_fame.update(population)
            
            # 记录统计信息
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            # 更新进度条
            if verbose:
                pbar.update(1)
                pbar.set_postfix({"最佳适应度": f"{record['min']:.4f}", 
                                "平均适应度": f"{record['avg']:.4f}"})
        
        # 关闭进度条
        if verbose:
            pbar.close()
    
    return population, logbook, hall_of_fame

# 若无法确定变异和交叉是否会超出自变量范围，可使用下列装饰器(具体使用见main2和main3)
def check_bounds(min_val, max_val):
    """约束交叉和变异的范围"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                child[0] = np.clip(child[0], min_val, max_val)
            return offspring
        return wrapper
    return decorator

def plot_results(logbook):
    """绘制结果图"""
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    min_val = logbook.select("min")
    max_val = logbook.select("max")
    
    plt.figure(figsize=(12, 6))
    plt.plot(gen, avg, label="平均适应度")
    plt.plot(gen, min_val, label="最小适应度")
    plt.plot(gen, max_val, label="最大适应度")
    plt.xlabel("进化代数")
    plt.ylabel("适应度")
    plt.title("遗传算法收敛曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================================
# 测试函数
# ==============================================

def rastrigin(individual):
    """多峰测试函数，最小值为0"""
    n = len(individual)
    return 10 * n + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in individual),

def sphere(individual):
    """单峰测试函数，最小值为0"""
    return sum(x**2 for x in individual),

def ackley(individual):
    """多峰测试函数，最小值为0"""
    x, y = individual
    return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - 
            np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 
            np.e + 20),


# ==============================================
#　不同测试主函数
# ==============================================
def main1():
    """多变量多范围函数"""
    # 变量范围
    bounds = [(-5.12, 5.12), (-4.0, 4.0), (-3.0, 3.0), (-2.0, 2.0)]
    
    # 创建工具箱
    toolbox = create_toolbox(rastrigin, bounds)

    # 运行遗传算法
    population, logbook, hall_of_fame = gaWithElitism(
        toolbox,
        pop_size=100,
        max_gen=200,
        cx_prob=0.8,
        mut_prob=0.3,
        elite_size=10,
        verbose=True
    )
    
    # 获取最佳个体
    best_individual = hall_of_fame[0]
    best_fitness = best_individual.fitness.values[0]
    
    # 输出结果
    print("\n优化结果:")
    print(f"最佳适应度: {best_fitness:.6f}")
    print("最佳解:")
    for i, value in enumerate(best_individual):
        print(f"变量 {i+1}: {value:.6f} (范围: {bounds[i][0]}~{bounds[i][1]})")
    
    # 绘制收敛曲线
    plot_results(logbook)

def main2():
    """多变量单范围函数"""
    # 变量范围 (2维sphere函数)
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    
    # 创建工具箱
    toolbox = base.Toolbox()
    
    # 注册属性生成器
    toolbox.register("attr_float", random.uniform, -5.12, 5.12) # 范围
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_float, n=len(bounds))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 注册遗传操作
    toolbox.register("evaluate", sphere)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    # 使用check_bounds保证变异和交叉不会超出自变量范围
    toolbox.decorate("mate", check_bounds(-5.12, 5.12))
    toolbox.decorate("mutate", check_bounds(-5.12, 5.12))

    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # 创建初始种群
    population = toolbox.population(n=100)
    
    # 创建统计对象
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)
    
    # 创建名人堂
    hall_of_fame = tools.HallOfFame(1)
    
    # 运行简单遗传算法
    population, logbook = algorithms.eaSimple(
        population, 
        toolbox,
        cxpb=0.8,    # 交叉概率
        mutpb=0.2,   # 变异概率
        ngen=100,    # 进化代数
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True
    )
    
    # 获取最佳个体
    best_individual = hall_of_fame[0]
    best_fitness = best_individual.fitness.values[0]
    
    # 输出结果
    print("\n优化结果:")
    print(f"最佳适应度: {best_fitness:.6f}")
    print("最佳解:")
    for i, value in enumerate(best_individual):
        print(f"变量 {i+1}: {value:.6f} (范围: {bounds[i][0]}~{bounds[i][1]})")
    
    # 绘制收敛曲线
    plot_results(logbook)

def main3():
    """单变量函数"""
    toolbox = base.Toolbox()
    
    # 初始化单变量个体
    toolbox.register("attr_float", random.uniform, -5.12, 5.12) # 范围
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_float, n=1)  # n=1
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    # 使用check_bounds保证变异和交叉不会超出自变量范围
    toolbox.decorate("mate", check_bounds(-5.12, 5.12))
    toolbox.decorate("mutate", check_bounds(-5.12, 5.12))

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", sphere)
    
    population = toolbox.population(n=100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    # 运行算法（交叉概率设为0.8）
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=0.8, mutpb=0.2, ngen=100,
        stats=stats, verbose=True
    )
    
    best_individual = tools.selBest(population, k=1)[0]
    best_fitness = sphere(best_individual)[0]
    print(f"最佳解: {best_individual[0]:.6f}, 相应适应度 = {best_fitness:.6f}")

    # 绘制收敛曲线
    plot_results(logbook)


if __name__ == "__main__":
    main3()