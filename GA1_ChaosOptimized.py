# 本代码参考司守奎《数学建模算法与应用》，375-376
import numpy as np
import matplotlib.pyplot as plt
from GA1 import GA, TSP, plot_convergence, plot_path

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 采用混沌序列可以获得比传统随机方法更均匀的空间覆盖和更强的全局搜索，避免过早收敛
class GA_Chaos(GA):
    def __init__(self, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1):
        """
        基于混沌序列的遗传算法类
        
        参数:
        population_size: 种群规模(w=50)
        generations: 进化代数(g=100)
        crossover_rate: 交叉概率
        mutation_rate: 变异概率(0.1)
        """
        super().__init__(population_size, generations, crossover_rate, mutation_rate)
    
    def crossover(self, parent1, parent2):
        """混沌序列交叉操作"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        # 生成混沌序列确定交叉位置
        ch = np.zeros(50)
        ch[0] = np.random.rand()
        for j in range(1, 50):
            ch[j] = 4 * ch[j-1] * (1 - ch[j-1])
        
        cross_points = 2 + np.floor(100 * ch[:50]).astype(int)
        
        child1, child2 = parent1.copy(), parent2.copy()
        # 在混沌序列确定的位置进行交叉
        for point in cross_points:
            if point >= len(parent1):
                continue
            child1[point], child2[point] = child2[point], child1[point]
            
        return child1, child2
    
    def mutate(self, individual):
        """混沌序列变异操作"""
        if np.random.rand() > self.mutation_rate:
            return individual.copy()
            
        # 生成混沌序列
        ch = np.zeros(2)
        ch[0] = np.random.rand()
        ch[1] = 4 * ch[0] * (1 - ch[0])
        
        # 选择两个变异位置
        mut_points = np.sort(2 + np.floor(100 * ch[:2]).astype(int))
        mut_points = np.clip(mut_points, 2, len(individual)-1)
        
        mutated = individual.copy()
        # 使用混沌值进行变异
        mutated[mut_points[0]] = ch[0]
        mutated[mut_points[1]] = ch[1]
        
        return mutated

if __name__ == "__main__":
    # 初始化问题实例和算法
    tsp = TSP(r'data\obj_longitude_latitude.txt')
    ga = GA_Chaos(population_size=50, generations=100, mutation_rate=0.1)
    
    # 求解问题
    best_path, best_length = ga.solve(tsp)
    
    # 输出结果
    print(f"最优路径长度: {best_length:0.4f}")
    print(f"最优路径: {best_path}")
    
    # 可视化
    plot_path(tsp.xy, best_path, '遗传算法求解旅行商问题最优路径')
    plot_convergence(ga.history, '遗传算法收敛曲线')