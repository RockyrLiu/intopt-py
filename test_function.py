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