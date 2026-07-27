import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tsp import problem1  

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class IA:
    """基本免疫算法"""
    def __init__(self, func, bounds, popsize=100, maxiter=100, 
                 mut_prob=0.7, simThresh=0.2, Ncl=10, refresh_rate=0.5,
                 verbose=True, plot=True):
        """
        参数说明:
        func: 目标函数
        bounds: 边界列表，每个元素是元组 (min, max)
        popsize: 种群数量
        maxiter: 最大迭代次数
        mut_prob: 变异概率
        simThresh: 相似度阈值
        Ncl: 克隆个数
        refresh_rate: 种群刷新率
        verbose: 是否显示进度条
        plot: 是否绘制结果
        """
        # 设置目标函数
        self.function = func  
        self.verbose = verbose
        self.plot = plot
        
        # 处理边界参数
        self.dim = len(bounds)
        self.lower_bound = np.array([b[0] for b in bounds])
        self.upper_bound = np.array([b[1] for b in bounds])
        
        # 算法参数
        self.popsize = popsize
        self.maxiter = maxiter
        self.mut_prob = mut_prob
        self.simThresh = simThresh
        self.Ncl = Ncl
        self.refresh_rate = refresh_rate
        self.deta0 = (self.upper_bound - self.lower_bound).mean()  # 使用平均值作为邻域范围初值
        
        # 初始化种群
        self.population = self.init_population()
        
        # 历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': []
        }
    
    def init_population(self):
        """初始化种群"""
        # 使用正确的广播方式初始化种群
        return (np.random.rand(self.dim, self.popsize) * 
                (self.upper_bound[:, None] - self.lower_bound[:, None]) + 
                self.lower_bound[:, None])
    
    def get_concentration(self, population):
        """计算个体浓度(欧氏距离)"""
        concentration = np.zeros(self.popsize)
        for i in range(self.popsize):
            nd = np.zeros(self.popsize)
            for j in range(self.popsize):
                dist = np.sqrt(np.sum((population[:, i] - population[:, j])**2))
                nd[j] = 1 if dist < self.simThresh else 0
            concentration[i] = np.sum(nd) / self.popsize
        return concentration
    
    def calculate_fitness(self, population, alpha=1, beta=1):
        """计算适应度(激励度)"""
        # 计算抗体亲和度
        affinity = np.array([self.function(population[:, i]) for i in range(self.popsize)])
        # 计算个体浓度
        concentration = self.get_concentration(population)
        # 计算适应度(激励度)
        fitness = alpha * affinity - beta * concentration
        return fitness, affinity
    
    def mutate(self, antibody, gen):
        """变异操作"""
        clones = np.tile(antibody, (self.Ncl, 1)).T  # 保持(dim, Ncl)形状
        deta = self.deta0 / (gen + 1)  # 动态调整变异幅度
        
        for j in range(self.Ncl):
            for d in range(self.dim):
                if np.random.rand() < self.mut_prob:
                    clones[d, j] += (np.random.rand() - 0.5) * deta
                # 边界处理
                clones[d, j] = np.clip(clones[d, j], self.lower_bound[d], self.upper_bound[d])
        return clones
    
    def refresh_population(self, keep_num):
        """种群刷新"""
        refresh_num = self.popsize - keep_num
        new_pop = self.init_population()[:, :refresh_num]  # 使用init_population方法
        new_fitness = np.array([self.function(new_pop[:, i]) for i in range(refresh_num)])
        return new_pop, new_fitness
    
    def iterator(self):
        """执行优化迭代"""
        if self.verbose:
            pbar = tqdm(total=self.maxiter, desc="IA优化进度")
        
        # 初始适应度计算
        motivation, fitness = self.calculate_fitness(self.population)
        sorted_indices = np.argsort(motivation)
        sorted_pop = self.population[:, sorted_indices]
        
        self.history['best_fitness'].append(np.min(fitness))
        self.history['avg_fitness'].append(np.mean(fitness))
        
        for gen in range(self.maxiter):
            # 计算保留和刷新的个体数量
            keep_num = int(self.popsize * (1 - self.refresh_rate))
            
            # 保留种群
            elite_pop = np.zeros((self.dim, keep_num))
            elite_fitness = np.zeros(keep_num)
            
            # 选前keep_num个体进行免疫操作
            for i in range(keep_num):
                clones = self.mutate(sorted_pop[:, i], gen)
                clones[:, 0] = sorted_pop[:, i]  # 保留克隆源个体
                
                # 计算克隆体亲和度并选择最优
                clone_fitness = np.array([self.function(clones[:, j]) for j in range(self.Ncl)])
                min_idx = np.argmin(clone_fitness)
                elite_fitness[i] = clone_fitness[min_idx]
                elite_pop[:, i] = clones[:, min_idx]
            
            # 种群刷新
            new_pop, new_fitness = self.refresh_population(keep_num)
            
            # 合并种群
            self.population = np.hstack((elite_pop, new_pop))
            current_fitness = np.hstack((elite_fitness, new_fitness))
            
            # 计算新种群激励度
            motivation, _ = self.calculate_fitness(self.population)
            sorted_indices = np.argsort(motivation)
            sorted_pop = self.population[:, sorted_indices]
            current_fitness = current_fitness[sorted_indices]
            
            # 记录历史
            best_fit = np.min(current_fitness)
            self.history['best_fitness'].append(best_fit)
            self.history['avg_fitness'].append(np.mean(current_fitness))
            
            # 更新进度条
            if self.verbose:
                pbar.set_postfix({
                    '最优值': f"{best_fit:.6f}",
                    '平均值': f"{np.mean(current_fitness):.6f}"
                })
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        # 获取最优解
        best_solution = sorted_pop[:, 0]
        best_fitness = best_fit
        
        # 绘制结果
        if self.plot:
            self.plot_history()
        
        return best_solution, self.history
    
    def plot_history(self):
        """绘制适应度进化曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(self.history['avg_fitness'], 'r--', label='平均适应度')
        plt.xlabel('迭代次数', size=12)
        plt.ylabel('适应度值', size=12)
        plt.title('免疫算法优化过程 - 适应度变化', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def run(self):
        return self.iterator()

class TSP_IA(IA):
    """TSP免疫算法"""
    def __init__(self, tsp_problem, func=None, bounds=None, popsize=200, maxiter=500, 
                 mut_prob=0.7, simThresh=0.2, Ncl=10, refresh_rate=0.5,
                 init_positions=None, verbose=True, plot=True):
        """
        参数说明:
        tsp_problem: TSP问题实例，必须实现calculate_path_length()和plot_solution()方法
        func: 目标函数 (由类内部自动设置)
        bounds: 边界列表 (由类内部自动设置)
        popsize: 种群数量
        maxiter: 最大迭代次数
        mut_prob: 变异概率
        simThresh: 相似度阈值
        Ncl: 克隆个数
        refresh_rate: 种群刷新率
        init_positions: 初始解列表(可选)
        verbose: 是否显示进度条
        plot: 是否绘制结果
        """
        self.tsp_problem = tsp_problem
        self.N = tsp_problem.N  # 城市数量
        self.init_positions = init_positions
        
        # 设置边界为城市索引范围 (0到N-1)
        bounds = [(0, self.N-1)] * self.N
        
        # 定义目标函数
        def obj_func(path):
            return tsp_problem.calculate_path_length(path.astype(int))
        
        # 调用父类初始化
        super().__init__(
            func=obj_func,
            bounds=bounds,
            popsize=popsize,
            maxiter=maxiter,
            mut_prob=mut_prob,
            simThresh=simThresh,
            Ncl=Ncl,
            refresh_rate=refresh_rate,
            verbose=verbose,
            plot=plot
        )
        
    def init_population(self):
        """初始化种群 - 随机排列路径或使用提供的初始解"""
        population = np.zeros((self.dim, self.popsize), dtype=int)
        
        if self.init_positions is not None and len(self.init_positions) > 0:
            # 使用提供的初始解填充部分种群
            num_provided = min(len(self.init_positions), self.popsize)
            for i in range(num_provided):
                population[:, i] = np.array(self.init_positions[i])
        
        # 剩余种群随机生成
        for i in range(num_provided, self.popsize):
            population[:, i] = np.random.permutation(self.dim)
            
        return population
    
    def mutate(self, antibody, gen):
        """变异操作 - 交换路径中的两个城市"""
        clones = np.tile(antibody, (self.Ncl, 1)).T  # 保持(dim, Ncl)形状
        
        for j in range(1, self.Ncl):  # 跳过第一个克隆(保留原抗体)
            if np.random.rand() < self.mut_prob:
                # 随机选择两个不同的位置进行交换
                p1, p2 = np.random.choice(self.dim, 2, replace=False)
                clones[p1, j], clones[p2, j] = clones[p2, j], clones[p1, j]
        
        return clones
    
    def refresh_population(self, keep_num):
        """种群刷新 - 生成新的随机路径"""
        refresh_num = self.popsize - keep_num
        new_pop = np.zeros((self.dim, refresh_num), dtype=int)
        new_fitness = np.zeros(refresh_num)
        
        for i in range(refresh_num):
            new_pop[:, i] = np.random.permutation(self.dim)
            new_fitness[i] = self.function(new_pop[:, i])
            
        return new_pop, new_fitness
    
    def run(self):
        """执行优化并返回结果"""
        best_solution, history = super().run()  # 父类已处理适应度曲线绘制
        best_solution = best_solution.astype(int)  # 确保路径为整数
        best_fitness = history['best_fitness'][-1]  # 获取最终最优值
        
        # 只绘制TSP特有图表（路径图）
        if self.plot:
            self._plot_tsp_solution(best_solution, best_fitness)
        
        return best_solution, best_fitness
    
    def _plot_tsp_solution(self, path, length):
        """绘制TSP路径图"""
        self.tsp_problem.plot_solution(path, length)


def ia(func, bounds, popsize=100, maxiter=100, mut_prob=0.7, 
       simThresh=0.2, Ncl=10, refresh_rate=0.5, verbose=True, plot=True):
    """
    免疫算法(连续优化问题)
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        popsize: 种群数量 (默认100)
        maxiter: 最大迭代次数 (默认100)
        mut_prob: 变异概率 (默认0.7)
        simThresh: 相似度阈值 (默认0.2)
        Ncl: 克隆个数 (默认10)
        refresh_rate: 种群刷新率 (默认0.5)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 历史记录
    """
    optimizer = IA(
        func=func,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        mut_prob=mut_prob,
        simThresh=simThresh,
        Ncl=Ncl,
        refresh_rate=refresh_rate,
        verbose=verbose,
        plot=plot
    )
    best_solution, history = optimizer.run()
    return best_solution, history

def ia_tsp(tsp_problem, popsize=200, maxiter=500, mut_prob=0.7, 
           simThresh=0.2, Ncl=10, refresh_rate=0.5, init_positions=None,
           verbose=True, plot=True):
    """
    免疫算法(TSP问题)
    参数:
        tsp_problem: TSP问题实例
        popsize: 种群大小 (默认200)
        maxiter: 最大迭代次数 (默认500)
        mut_prob: 变异概率 (默认0.7)
        simThresh: 相似度阈值 (默认0.2)
        Ncl: 克隆数量 (默认10)
        refresh_rate: 种群刷新率 (默认0.5)
        init_positions: 初始解列表(可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        最优路径, 路径长度, 历史记录
    """
    optimizer = TSP_IA(
        tsp_problem=tsp_problem,
        popsize=popsize,
        maxiter=maxiter,
        mut_prob=mut_prob,
        simThresh=simThresh,
        Ncl=Ncl,
        refresh_rate=refresh_rate,
        init_positions=init_positions,
        verbose=verbose,
        plot=plot
    )
    
    best_path, best_length = optimizer.run()
    return best_path, best_length, optimizer.history


def main1():
    # 测试函数
    def sphere(X):
        return X[0] ** 2 + X[1] ** 2 + X[2] ** 2
    
    # 调用免疫算法
    best_solution, history = ia(
        func=sphere,
        bounds=[(-5.12, 5.12)] * 3,
        popsize=50,
        maxiter=100,
        verbose=True
    )
    
    print("最优解:", best_solution)

def main2():
    """执行TSP优化的主函数""" 
    # 创建TSP问题实例
    tsp_problem = problem1
    
    # 可以自定义初始解
    custom_init = [
        list(range(tsp_problem.N)),  # 顺序路径
        list(reversed(range(tsp_problem.N)))  # 逆序路径
    ]
    
    # 调用免疫算法
    best_path, best_length, history = ia_tsp(
        tsp_problem=tsp_problem,
        popsize=200,
        maxiter=500,
        init_positions=custom_init,
        verbose=True,
        plot=True
    )
    
    print("\n最优路径:", best_path)
    print(f"最短距离: {best_length:.0f}")

if __name__ == "__main__":
    main1()