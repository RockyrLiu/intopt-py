import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm
import multiprocessing
from itertools import chain

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TSP:
    def __init__(self, data_file):
        """初始化TSP问题"""
        self.load_data(data_file)
        self.calc_distance_matrix()
        self.num_cities = len(self.xy)
    
    def load_data(self, data_file):
        """加载城市坐标数据"""
        sj0 = np.loadtxt(data_file)
        x = sj0[:, 0:8:2].flatten()
        y = sj0[:, 1:8:2].flatten()
        sj = np.column_stack((x, y))
        d1 = np.array([70, 40])  # 起点和终点
        self.xy = np.vstack((d1, sj, d1))
        self.sj = self.xy * np.pi / 180  # 角度转弧度
    
    def calc_distance_matrix(self):
        """计算距离矩阵（球面距离）"""
        n = len(self.sj)
        self.d = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # 球面距离公式
                self.d[i, j] = 6370 * np.arccos(
                    np.cos(self.sj[i, 0]-self.sj[j, 0]) * 
                    np.cos(self.sj[i, 1]) * np.cos(self.sj[j, 1]) + 
                    np.sin(self.sj[i, 1]) * np.sin(self.sj[j, 1])
                )
        self.d = self.d + self.d.T  # 对称矩阵
    
    def calculate_path_length(self, path):
        """计算路径长度"""
        return sum(self.d[path[i], path[i+1]] for i in range(len(path)-1))

    def evaluate(self, individual):
        """DEAP评估函数"""
        return (self.calculate_path_length(individual),)

def initialize_individual(icls, tsp):
    """使用改良圈算法初始化个体"""
    c = list(np.random.permutation(tsp.num_cities-2) + 1)  # 1~N-2的排列
    c1 = [0] + c + [tsp.num_cities-1]  # 加入起点和终点
    
    # 改良圈算法优化
    improved = True
    while improved:
        improved = False
        for m in range(tsp.num_cities-2):
            for n in range(m+2, tsp.num_cities-1):
                if (tsp.d[c1[m], c1[n]] + tsp.d[c1[m+1], c1[n+1]] < 
                    tsp.d[c1[m], c1[m+1]] + tsp.d[c1[n], c1[n+1]]):
                    c1[m+1:n+1] = c1[n:m:-1]  # 反转中间部分
                    improved = True
    return icls(c1)

def cxPartialyMatched(ind1, ind2):
    """部分匹配交叉(PMX)"""
    size = len(ind1)
    if size < 3:
        return ind1, ind2
        
    # 随机选择两个交叉点（避开起点和终点）
    p1, p2 = sorted(random.sample(range(1, size-1), 2))
    
    # 创建映射关系
    mapping1 = {ind2[i]: ind1[i] for i in range(p1, p2+1)}
    mapping2 = {ind1[i]: ind2[i] for i in range(p1, p2+1)}
    
    # 修复冲突
    for i in chain(range(p1), range(p2+1, size)):
        while ind1[i] in mapping1:
            ind1[i] = mapping1[ind1[i]]
        while ind2[i] in mapping2:
            ind2[i] = mapping2[ind2[i]]
    
    # 交换中间部分
    ind1[p1:p2+1], ind2[p1:p2+1] = ind2[p1:p2+1], ind1[p1:p2+1]
    return ind1, ind2

def mutInverseIndexes(individual):
    """反转变异"""
    if len(individual) < 3:
        return (individual,)
    
    # 随机选择两个变异点（避开起点和终点）
    start, stop = sorted(random.sample(range(1, len(individual)-1), 2))
    individual[start:stop] = individual[stop-1:start-1:-1]
    return (individual,)

def plot_results(tsp, best_path, history):
    """可视化结果"""
    # 绘制最优路径
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    xx = tsp.xy[best_path, 0]
    yy = tsp.xy[best_path, 1]
    plt.plot(xx, yy, '-o', markersize=4)
    plt.scatter(tsp.xy[0, 0], tsp.xy[0, 1], s=200, c='red', marker='s', label='起点/终点')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title(f'最优路径 (长度: {history["best_fitness"][-1]:.2f} km)')
    plt.legend()
    plt.grid(True)
    
    # 绘制收敛曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['best_fitness'], label='最优适应度')
    plt.plot(history['avg_fitness'], label='平均适应度')
    plt.title('算法收敛曲线')
    plt.xlabel('进化代数')
    plt.ylabel('路径长度 (km)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # 初始化TSP问题
    tsp = TSP(r'data\obj_longitude_latitude.txt')
    
    # 创建DEAP框架
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # 注册遗传算法操作
    toolbox.register("individual", initialize_individual, creator.Individual, tsp)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", tsp.evaluate)
    toolbox.register("mate", cxPartialyMatched)
    toolbox.register("mutate", mutInverseIndexes)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # 算法参数
    population_size = 50
    ngen = 100
    cxpb = 0.7
    mutpb = 0.2
    
    # 创建初始种群
    population = toolbox.population(n=population_size)
    
    # 设置统计和跟踪
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    history = {'best_fitness': [], 'avg_fitness': []}
    
    # 运行遗传算法（带进度条）
    with tqdm(total=ngen, desc="进化进度", unit="gen") as pbar:
        def callback(gen):
            pbar.update(1)
            pbar.set_postfix({
                "当前最优": f"{history['best_fitness'][-1]:.2f}",
                "平均适应度": f"{history['avg_fitness'][-1]:.2f}"
            })
        
        # 评估初始种群
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        hof.update(population)
        history['best_fitness'].append(hof[0].fitness.values[0])
        history['avg_fitness'].append(np.mean([ind.fitness.values[0] for ind in population]))
        callback(0)
        
        # 进化循环
        for gen in range(1, ngen+1):
            # 选择
            offspring = toolbox.select(population, len(population))
            
            # 交叉和变异
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            
            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 精英保留
            hof.update(offspring)
            worst = min(offspring, key=lambda ind: ind.fitness)
            if worst.fitness < hof[0].fitness:
                offspring[offspring.index(worst)] = hof[0]
            
            # 更新种群
            population[:] = offspring
            
            # 记录统计信息
            history['best_fitness'].append(hof[0].fitness.values[0])
            history['avg_fitness'].append(np.mean([ind.fitness.values[0] for ind in population]))
            callback(gen)
    
    # 获取最优解
    best_path = list(map(int, hof[0]))  # 转换为普通整数
    best_length = hof[0].fitness.values[0]
    
    # 输出结果
    print(f"\n最优路径长度: {best_length:.2f} km")
    print("最优路径:", best_path)
    
    # 可视化
    plot_results(tsp, best_path, history)
    
    # 清理
    pool.close()

if __name__ == "__main__":
    main()