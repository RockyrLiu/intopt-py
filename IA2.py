import numpy as np
import matplotlib.pyplot as plt
from IA1 import IA  # 导入基础免疫算法类

class TSPProblem:
    """TSP问题封装类，用于计算路径长度和适应度"""
    def __init__(self, city_coordinates):
        self.C = city_coordinates  # 城市坐标矩阵
        self.N = city_coordinates.shape[0]  # 城市数量
        self.D = self._calculate_distance_matrix()  # 距离矩阵

    def _calculate_distance_matrix(self):
        """计算城市间距离矩阵"""
        D = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                D[i, j] = np.sqrt((self.C[i, 0] - self.C[j, 0])**2 + 
                                 (self.C[i, 1] - self.C[j, 1])**2)
        return D

    def calculate_path_length(self, path):
        """计算给定路径的总长度"""
        length = self.D[path[-1], path[0]]  # 回到起点的距离
        for i in range(self.N-1):
            length += self.D[path[i], path[i+1]]
        return length

    def plot_solution(self, best_path, best_length):
        """绘制最优路径图"""
        plt.figure(figsize=(10, 6))
        # 绘制城市点
        plt.scatter(self.C[:, 0], self.C[:, 1], color='red')
        
        # 绘制路径
        for i in range(self.N-1):
            plt.plot([self.C[best_path[i], 0], self.C[best_path[i+1], 0]],
                     [self.C[best_path[i], 1], self.C[best_path[i+1], 1]], 'b-')
        # 连接最后一个城市和起点
        plt.plot([self.C[best_path[-1], 0], self.C[best_path[0], 0]],
                 [self.C[best_path[-1], 1], self.C[best_path[0], 1]], 'r-')
        
        plt.title(f"优化最短距离: {best_length:.2f}")
        
        # 标记城市编号
        for i in range(self.N):
            plt.text(self.C[i, 0], self.C[i, 1], str(i), color='red', fontsize=8)
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.show()

class TSP_IA(IA):
    """TSP专用的免疫算法类，继承自基础IA类"""
    def __init__(self, city_coordinates, Np=200, max_iter=1000, mut_prob=0.7, 
                 simThresh=0.2, Ncl=10, refresh_rate=0.5):
        self.tsp_problem = TSPProblem(city_coordinates)
        N = city_coordinates.shape[0]
        
        # 初始化父类，refresh_rate作为参数传入，默认0.5
        super().__init__(Np, N, max_iter, mut_prob, simThresh, Ncl, 
                        self._tsp_objective, 0, N-1, refresh_rate=refresh_rate)
        
        # 覆盖父类的历史记录，因为TSP需要记录路径长度而非函数值
        self.history = {'best_length': []}

    def _tsp_objective(self, path):
        """适配TSP的目标函数（IA父类需要）"""
        return self.tsp_problem.calculate_path_length(path.astype(int))

    def init_population(self):
        """覆盖父类方法：生成随机排列作为初始种群"""
        f = np.zeros((self.dim, self.Np), dtype=int)
        for i in range(self.Np):
            f[:, i] = np.random.permutation(self.dim)
        return f

    def mutate(self, a, gen):
        """覆盖父类方法：TSP专用变异（交换城市）"""
        Na = np.tile(a, (self.Ncl, 1)).T  # 保持(dim, Ncl)形状
        
        for j in range(1, self.Ncl):  # 跳过第一个(保留原个体)
            # 随机交换两个城市位置
            p1, p2 = np.random.choice(self.dim, 2, replace=False)
            Na[p1, j], Na[p2, j] = Na[p2, j], Na[p1, j]
        
        return Na

    def iterator(self):
        """覆盖父类迭代过程，适配TSP问题"""
        f = self.init_population()
        fitness = self.get_affinity(f)
        self.history['best_length'].append(np.min(fitness))
        
        sorted_indices = np.argsort(fitness)
        Sortf = f[:, sorted_indices]
        
        for gen in range(self.max_iter):
            # 使用父类的refresh_rate计算保留和刷新数量
            keep_num = int(self.Np * (1 - self.refresh_rate))
            refresh_num = self.Np - keep_num
            
            af = np.zeros((self.dim, keep_num), dtype=int)
            aFIT = np.zeros(keep_num)
            
            # 免疫操作
            for i in range(keep_num):
                Na = self.mutate(Sortf[:, i], gen)
                NaFIT = np.array([self._tsp_objective(Na[:, j]) for j in range(self.Ncl)])
                min_idx = np.argmin(NaFIT)
                af[:, i] = Na[:, min_idx]
                aFIT[i] = NaFIT[min_idx]
            
            # 种群刷新
            bf = np.zeros((self.dim, refresh_num), dtype=int)
            bFIT = np.zeros(refresh_num)
            for i in range(refresh_num):
                bf[:, i] = np.random.permutation(self.dim)
                bFIT[i] = self._tsp_objective(bf[:, i])
            
            # 合并和排序
            f1 = np.hstack((af, bf))
            FIT1 = np.hstack((aFIT, bFIT))
            
            sorted_indices = np.argsort(FIT1)
            Sortf = f1[:, sorted_indices]
            fitness = FIT1[sorted_indices]
            
            self.history['best_length'].append(np.min(fitness))
            
            if gen % 100 == 0:
                print(f"代数 {gen}, 当前最短路径: {np.min(fitness):.2f}")
        
        best_solution = Sortf[:, 0]
        best_length = np.min(fitness)
        return best_solution, best_length

    def run(self):
        """覆盖父类run方法以返回路径长度历史"""
        best_solution, best_length = self.iterator()
        return best_solution, best_length, self.history['best_length']

def immune_algorithm_tsp():
    """执行TSP优化的主函数"""
    # 中国31个省会城市坐标
    city_coordinates = np.array([
        [1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535],
        [3326, 1556], [3238, 1229], [4196, 1044], [4312, 790], [4386, 570],
        [3007, 1970], [2562, 1756], [2788, 1491], [2381, 1676], [1332, 695],
        [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578],
        [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376], [3394, 2643],
        [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
        [2370, 2975]
    ])
    
    # 创建并运行TSP免疫算法，refresh_rate默认为0.5
    tsp_ia = TSP_IA(city_coordinates, Np=200, max_iter=1000)
    best_path, best_length, trace = tsp_ia.run()
    
    # 输出结果
    print("\n最优路径:", best_path)
    print("最短距离:", best_length)
    
    # 绘制结果
    tsp_ia.tsp_problem.plot_solution(best_path, best_length)
    
    # 绘制进化曲线
    plt.figure()
    plt.plot(trace)
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.title('亲和度进化曲线')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    immune_algorithm_tsp()