# 本代码参考司守奎《数学建模算法与应用》，375-376
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class GA:
    def __init__(self, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1):
        """
        遗传算法类
        
        参数:
        population_size: 种群规模(w=50)
        generations: 进化代数(g=100)
        crossover_rate: 交叉概率
        mutation_rate: 变异概率(0.1)
        """
        self.pop_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.history = {'best_fitness': [], 'avg_fitness': []}
    
    def initialize_population(self, tsp):
        """初始化种群"""
        population = np.zeros((self.pop_size, tsp.num_cities))
        
        for k in range(self.pop_size):
            # 产生初始解
            c = np.random.permutation(tsp.num_cities-2) + 1  # 1-100的排列
            c1 = np.concatenate(([0], c, [tsp.num_cities-1]))  # 加入起点和终点
            
            # 改良圈算法
            flag = True
            while flag:
                flag = False
                for m in range(tsp.num_cities-2):
                    for n in range(m+2, tsp.num_cities-1):
                        if (tsp.d[c1[m], c1[n]] + tsp.d[c1[m+1], c1[n+1]] < 
                            tsp.d[c1[m], c1[m+1]] + tsp.d[c1[n], c1[n+1]]):
                            c1[m+1:n+1] = c1[n:m:-1]  # 反转中间部分
                            flag = True
                if not flag:
                    population[k, c1] = np.arange(tsp.num_cities)  # 记录解
                    break
        
        return population / (tsp.num_cities-1)  # 转换为[0,1]区间的编码
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        F = 2 + np.floor((len(parent1)-2) * np.random.rand()).astype(int)
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 交换部分基因
        temp = child1[F:].copy()
        child1[F:] = child2[F:]
        child2[F:] = temp
        
        return child1, child2
    
    def mutate(self, individual):
        """变异操作"""
        if np.random.rand() > self.mutation_rate:
            return individual.copy()
            
        # 产生3个不同的变异位置    
        bw = np.sort(np.random.choice(len(individual)-2, 3, replace=False) + 2)
        # 交换片段位置
        mutated = np.concatenate((
            individual[:bw[0]], 
            individual[bw[1]:bw[2]], 
            individual[bw[0]:bw[1]], 
            individual[bw[2]:]
        ))
        
        return mutated
    
    def select(self, population, fitness):
        """
        选择操作(精英保留策略)
        """
        # 按适应度排序(越小越好)
        indices = np.argsort(fitness)[:self.pop_size]
        return population[indices]
    
    def solve(self, tsp):
        """求解TSP问题"""
        # 初始化种群
        population = self.initialize_population(tsp)
        
        # 主循环
        best_individual = None
        best_fitness = float('inf')
        
        with tqdm(range(self.generations), desc="进化进度", unit="gen") as pbar:
            for gen in pbar:
                # 计算适应度
                decoded_pop = self.decode_population(population, tsp)
                fitness = np.array([tsp.calculate_path_length(ind) for ind in decoded_pop])
                
                # 更新历史记录
                current_best = fitness.min()
                current_avg = fitness.mean()
                self.history['best_fitness'].append(current_best)
                self.history['avg_fitness'].append(current_avg)
                
                # 更新最优个体
                if current_best < best_fitness:
                    best_fitness = current_best
                    best_idx = fitness.argmin()
                    best_individual = population[best_idx]
                
                # 选择优质父代产生后代(Selection)
                selected = self.select(population, fitness)
                
                # 交叉
                offspring = []
                # 随机配对
                c = np.random.permutation(self.pop_size)
                for i in range(0, self.pop_size, 2):
                    if i+1 >= self.pop_size:
                        break
                    child1, child2 = self.crossover(selected[c[i]], selected[c[i+1]])
                    offspring.extend([child1, child2])
                
                # 变异
                mutated = []
                by = np.where(np.random.rand(self.pop_size) < self.mutation_rate)[0]
                for i in range(self.pop_size):
                    if i in by:
                        mutated.append(self.mutate(offspring[i]))
                    else:
                        mutated.append(offspring[i])
                
                # 新一代种群(合并父代和子代)
                G = np.vstack((population, np.array(mutated)))
                decoded_G = self.decode_population(G, tsp)
                fitness_G = np.array([tsp.calculate_path_length(ind) for ind in decoded_G])
                
                # 精英保留（Elitism）
                indices = np.argsort(fitness_G)[:self.pop_size]
                population = G[indices]
                
                # 更新进度条
                pbar.set_postfix({
                    "最优解": f"{best_fitness:.2f}",
                    "平均解": f"{current_avg:.2f}",
                    "当前最优": f"{current_best:.2f}"
                })
        
        # 解码最优个体
        best_path = self.decode_individual(best_individual, tsp)
        return best_path, best_fitness
    
    def decode_population(self, population, tsp):
        """
        解码整个种群
        """
        return [self.decode_individual(ind, tsp) for ind in population]
    
    def decode_individual(self, individual, tsp):
        """
        解码单个个体
        """
        return np.argsort(individual)

class TSP:
    def __init__(self, data_file):
        """
        旅行商问题类
        """
        self.load_data(data_file)
        self.calc_distance_matrix()
        self.num_cities = len(self.xy)
    
    def load_data(self, data_file):
        """加载城市坐标数据"""
        sj0 = np.loadtxt(data_file)
        x = sj0[:, 0:8:2].flatten()
        y = sj0[:, 1:8:2].flatten()
        sj = np.column_stack((x, y))
        d1 = np.array([70, 40])
        self.xy = np.vstack((d1, sj, d1))
        self.sj = self.xy * np.pi / 180  # 角度转弧度
    
    def calc_distance_matrix(self):
        """计算城市间距离矩阵"""
        n = len(self.sj)
        self.d = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                self.d[i, j] = 6370 * np.arccos(np.cos(self.sj[i, 0]-self.sj[j, 0]) * np.cos(self.sj[i, 1]) * \
                              np.cos(self.sj[j, 1]) + np.sin(self.sj[i, 1]) * np.sin(self.sj[j, 1]))
        
        self.d = self.d + self.d.T
    
    def calculate_path_length(self, path):
        """计算路径长度"""
        length = 0
        for i in range(len(path)-1):
            length += self.d[path[i], path[i+1]]
        return length

def plot_path(xy, path, title):
    """绘制路径图"""
    xx = xy[path, 0]
    yy = xy[path, 1]
    plt.figure(figsize=(10, 6))
    plt.plot(xx, yy, '-o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_convergence(history, title):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['best_fitness'], label='最优适应度')
    plt.plot(history['avg_fitness'], label='平均适应度')
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Path Length')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 初始化问题实例和算法
    tsp = TSP(r'data\obj_longitude_latitude.txt')
    ga = GA(population_size=50, generations=100, mutation_rate=0.1)
    
    # 求解问题
    best_path, best_length = ga.solve(tsp)
    
    # 输出结果
    print(f"最优路径长度: {best_length:0.4f}")
    print(f"最优路径: {best_path}")
    
    # 可视化
    plot_path(tsp.xy, best_path, '遗传算法求解旅行商问题最优路径')
    plot_convergence(ga.history, '遗传算法收敛曲线')