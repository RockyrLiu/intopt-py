import numpy as np
import matplotlib.pyplot as plt

def func1(X):
    """目标函数"""
    x1 = X[:, 0]  # 第一个维度
    x2 = X[:, 1]  # 第二个维度
    return 3 * x1**2 - 2.1 * x1**4 + (x1**6) / 3 + x1 * x2 - 3 * x2**2 + 3 * x2**4

def func2(x):
    """目标函数：f(x,y) = 3cos(xy) + x + y"""
    return 3 * np.cos(x[0] * x[1]) + x[0] + x[1]

def plot_func2():
    """绘制目标函数3D图像"""
    x = np.arange(-4, 4, 0.02)
    y = np.arange(-4, 4, 0.02)
    X, Y = np.meshgrid(x, y)
    Z = 3 * np.cos(X * Y) + X + Y
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('f(x,y) = 3cos(xy) + x + y')
    plt.show()

def func3(x):
    """目标函数：f(x,y) = -((x²+y-1)² + (x+y²-7)²)/200 + 10"""
    return (x[0]**2 + x[1] - 1)**2 + (x[0] + x[1]**2 - 7)**2 / 200 + 10

def plot_func3():
    """绘制目标函数3D图像"""
    x = np.arange(-100, 101, 1)
    y = np.arange(-100, 101, 1)
    X, Y = np.meshgrid(x, y)
    Z = -((X**2 + Y - 1)**2 + (X + Y**2 - 7)**2)/200 + 10
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', rstride=10, cstride=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('f(x,y) = -((x^2+y-1)^2 + (x+y^2-7)^2)/200 + 10')
    plt.show()

# test_function.py 中的修改
def Rastrigin(X, A=10):
    """支持1D和2D输入的Rastrigin函数，确保返回标量"""
    if X.ndim == 1:
        # 单个个体直接计算并返回标量
        return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))
    else:
        # 多个个体返回一维数组
        return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)

def Square(x):
    """平方和测试函数"""
    return np.sum(x**2)

class TSPProblem1:
    """TSP问题封装类，用于计算路径长度和适应度"""
    def __init__(self):
            # 中国31个省会城市坐标
        city_coordinates = np.array([
            [1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535],
            [3326, 1556], [3238, 1229], [4196, 1044], [4312, 790], [4386, 570],
            [3007, 1970], [2562, 1756], [2788, 1491], [2381, 1676], [1332, 695],
            [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578],
            [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376], [3394, 2643],
            [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
            [2370, 2975]])   
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
