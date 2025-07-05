import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class KnapsackProblem:
    """0-1背包问题封装类"""
    def __init__(self, capacity, weights, values, penalty=2):
        """
        参数:
        capacity: 背包容量
        weights: 物品重量列表
        values: 物品价值列表
        penalty: 超出容量时的惩罚系数
        """
        self.capacity = capacity
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.penalty = penalty
        self.n_items = len(weights)
        
        if len(weights) != len(values):
            raise ValueError("物品重量和价值数量必须相同")
    
    def fitness(self, X):
        """
        计算适应度(目标函数)
        X: 二进制解向量(0或1), 形状为(n_samples, n_items)
        返回: 适应度值数组, 形状为(n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        total_weight = X @ self.weights
        total_value = X @ self.values
        
        # 对超重的情况施加惩罚
        penalty = np.maximum(total_weight - self.capacity, 0) * self.penalty
        return total_value - penalty
    
    def evaluate_solution(self, solution):
        """评估单个解并返回详细信息"""
        selected = solution.astype(bool)
        total_value = np.sum(self.values[selected])
        total_weight = np.sum(self.weights[selected])
        feasible = total_weight <= self.capacity
        return {
            'solution': solution,
            'selected_indices': np.where(selected)[0],
            'total_value': total_value,
            'total_weight': total_weight,
            'is_feasible': feasible
        }


class TSPProblem:
    """TSP问题封装类"""
    def __init__(self, coordinates):
        """
        参数:
        coordinates: 城市坐标数组，形状为(n, 2)
        """
        self.coordinates = np.array(coordinates)
        self.N = len(coordinates)  # 城市数量
        self.D = self._calculate_distance_matrix()  # 距离矩阵
    
    def _calculate_distance_matrix(self):
        """计算城市间的距离矩阵"""
        dist_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    dist_matrix[i][j] = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
        return dist_matrix
    
    def calculate_path_length(self, path):
        """计算给定路径的总长度"""
        total_length = 0
        for i in range(len(path) - 1):
            total_length += self.D[path[i]][path[i+1]]
        # 回到起点
        total_length += self.D[path[-1]][path[0]]
        return total_length
    
    def plot_solution(self, path, path_length=None):
        """可视化路径"""
        plt.figure(figsize=(10, 6))
        
        # 绘制城市点
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='red', marker='o')
        
        # 绘制路径
        ordered_coords = self.coordinates[path]
        ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])  # 回到起点
        plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'b-', linewidth=1)
        
        # 标记起点
        plt.scatter(ordered_coords[0, 0], ordered_coords[0, 1], c='green', marker='*', s=200, label='起点')
        
        # 添加标题
        title = 'TSP路径规划'
        if path_length is not None:
            title += f' (总长度: {path_length:.2f})'
        plt.title(title)
        
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.legend()
        plt.grid(True)
        plt.show()


coordinates1 = np.array([
    [1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535],
    [3326, 1556], [3238, 1229], [4196, 1044], [4312, 790], [4386, 570],
    [3007, 1970], [2562, 1756], [2788, 1491], [2381, 1676], [1332, 695],
    [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578],
    [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376], [3394, 2643],
    [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
    [2370, 2975]
])
problem1 = TSPProblem(coordinates1)

# 使用示例
if __name__ == "__main__":
    # 31个城市的坐标
    coordinates = np.array([
        [1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535],
        [3326, 1556], [3238, 1229], [4196, 1044], [4312, 790], [4386, 570],
        [3007, 1970], [2562, 1756], [2788, 1491], [2381, 1676], [1332, 695],
        [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578],
        [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376], [3394, 2643],
        [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
        [2370, 2975]
    ])
    
    # 创建TSP问题实例
    tsp_problem = TSPProblem(coordinates)
    
    # 测试随机路径
    random_path = np.random.permutation(tsp_problem.N)
    path_length = tsp_problem.calculate_path_length(random_path)
    print(f"随机路径长度: {path_length:.2f}")
    
    # 可视化随机路径
    tsp_problem.plot_solution(random_path, path_length)