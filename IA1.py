import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
from test_function import Square, Rastrigin, func2, plot_func2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class IA:
    """基本免疫算法"""
    def __init__(self, Np, dim, max_iter, mut_prob, simThresh, Ncl, obj_function, 
                 lower_bound, upper_bound, refresh_rate=0.5, verbose=False):
        self.Np = Np # 种群数
        self.dim = dim # 维数
        self.max_iter = max_iter # 最大进化代数
        self.mut_prob = mut_prob # 变异率
        self.simThresh = simThresh # 相似度阈值
        self.Ncl = Ncl # 克隆个数
        self.obj_function = obj_function # 目标函数
        self.lower_bound = lower_bound # 取值上界
        self.upper_bound = upper_bound # 取值下界
        self.refresh_rate = refresh_rate # 种群刷新率，刷新refresh_rate的个体
        self.deta0 = (upper_bound - lower_bound)  # 邻域范围初值
        self.history = {'best_fitness': [], 'avg_fitness': []}
        self.verbose = verbose # 显示进度条

    def init_population(self):
        """初始化种群"""
        f = np.random.rand(self.dim, self.Np) * (self.upper_bound - self.lower_bound) + self.lower_bound
        return f

    def get_affinity(self, f):
        """计算亲和度"""
        return np.array([self.obj_function(f[:, i]) for i in range(self.Np)])

    def get_concentration(self, f):
        """计算个体浓度(欧氏距离)"""
        concentration = np.zeros(self.Np)
        for i in range(self.Np):
            nd = np.zeros(self.Np)
            for j in range(self.Np):
                dist = np.sqrt(np.sum((f[:, i] - f[:, j])**2))
                nd[j] = 1 if dist < self.simThresh else 0
            concentration[i] = np.sum(nd) / self.Np

        return concentration

    def calculate_fitness(self, f, alpha=1, beta=1):
        """计算适应度(激励度)"""
        # 计算抗体亲和度
        affinity = self.get_affinity(f)
        assert isinstance(affinity, np.ndarray) and affinity.ndim == 1, \
        f"亲和度应为1维数组，实际为{affinity.shape}"
        # 计算个体浓度
        concentration = self.get_concentration(f)
        
        # 计算适应度(激励度)
        fitness = alpha * affinity - beta * concentration

        return fitness, affinity

    def mutate(self, a, gen):
        """变异操作"""
        Na = np.tile(a, (self.Ncl, 1)).T  # 保持(dim, Ncl)形状
        deta = self.deta0 / (gen + 1)  # 动态调整变异幅度
        
        for j in range(self.Ncl):
            for ii in range(self.dim):
                if np.random.rand() < self.mut_prob:
                    Na[ii, j] += (np.random.rand() - 0.5) * deta
                # 边界处理
                Na[ii, j] = np.clip(Na[ii, j], self.lower_bound, self.upper_bound)
        
        return Na

    def refreshPop(self, keep_num):
        """种群刷新"""
        # 计算刷新的个体数量
        refresh_num = self.Np - keep_num
        # 刷新种群
        bf = np.random.rand(self.dim, refresh_num) * (self.upper_bound - self.lower_bound) + self.lower_bound
        bFIT = np.array([self.obj_function(bf[:, i]) for i in range(refresh_num)])

        return bf, bFIT

    def iterator(self):
        """优化迭代"""
        # 初始化种群
        f = self.init_population()
        
        # 初始适应度计算
        motivation, fitness = self.calculate_fitness(f)
        self.history['best_fitness'].append(np.min(fitness))
        self.history['avg_fitness'].append(np.mean(fitness))
        
        # 按激励度排序
        sorted_indices = np.argsort(motivation)
        Sortf = f[:, sorted_indices]
        
        # 使用tqdm进度条（如果verbose为True）
        iter_range = range(self.max_iter)
        if self.verbose:
            pbar = tqdm(iter_range, desc="迭代进度")  # 创建进度条对象
        
        # 免疫循环
        for gen in iter_range:
            # 计算保留和刷新的个体数量
            keep_num = int(self.Np * (1 - self.refresh_rate))
            
            af = np.zeros((self.dim, keep_num)) # 保留的种群
            aFIT = np.zeros(keep_num) # 保留种群的亲和度
            
            # 选前keep_num个体进行免疫操作
            for i in range(keep_num):
                assert Sortf.shape == (self.dim, self.Np), f"种群矩阵应为(dim,Np)，实际为{Sortf.shape}"
                # 克隆和变异
                Na = self.mutate(Sortf[:, i], gen)
                Na[:, 0] = Sortf[:, i]  # 保留克隆源个体
                
                # 计算克隆体亲和度并选择最优
                NaFIT = np.array([self.obj_function(Na[:, j]) for j in range(self.Ncl)])
                min_index = np.argmin(NaFIT)
                aFIT[i] = NaFIT[min_index]
                af[:, i] = Na[:, min_index]
            
            # 种群刷新
            bf, bFIT = self.refreshPop(keep_num)
            
            # 合并种群
            f1 = np.hstack((af, bf))
            FIT1 = np.hstack((aFIT, bFIT))
            
            # 计算新种群激励度
            motivation1, _ = self.calculate_fitness(f1)
            
            # 重新排序
            sorted_indices = np.argsort(motivation1)
            Sortf = f1[:, sorted_indices]
            fitness = FIT1[sorted_indices]
            
            # 记录历史
            current_best = np.min(fitness)
            self.history['best_fitness'].append(current_best)
            self.history['avg_fitness'].append(np.mean(fitness))
            
            # 更新进度条显示（如果verbose为True）
            if self.verbose:
                pbar.set_postfix({'最优值': f"{current_best:.4f}"}) # 动态更新最优值
                pbar.update(1)  # 更新进度
        
        # 关闭进度条（如果verbose为True）
        if self.verbose:
            pbar.close()
        
        # 获取最优解
        best_solution = Sortf[:, 0]
        best_fitness = np.min(fitness)
        return best_solution, best_fitness
    
    def run(self):
        return self.iterator()

def plot_history(history):
    """绘制适应度进化曲线"""
    plt.figure()
    plt.plot(history['best_fitness'], label='最佳适应度')
    plt.plot(history['avg_fitness'], label='平均适应度')
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title('适应度进化曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    Np = 100       # 种群个体数量
    dim = 10      # 变量维度
    max_iter = 100  # 最大进化代数
    mut_prob = 0.7  # 变异概率
    refresh_rate = 0.5  # 种群刷新率
    
    ia = IA(Np, dim, max_iter, mut_prob, 0.2, 10, Square, lower_bound=-5.12, 
            upper_bound=5.12, refresh_rate=refresh_rate, verbose=True)
    
    best_solution, best_value = ia.run()
    print(f"最优解: {np.round(best_solution, 4)}")
    print(f"最优值: {best_value:.4f}")
    plot_history(ia.history)
    
if __name__ == "__main__":
    main()