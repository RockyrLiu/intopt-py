import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import List, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def plot_function():
    """绘制函数 f(x,y) = (cos(x^2+y^2)-0.1)/(1+0.3*(x^2+y^2)^2)+3 的3D图形"""
    x = np.arange(-5, 5.01, 0.01)
    y = np.arange(-5, 5.01, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = (np.cos(X**2 + Y**2) - 0.1) / (1 + 0.3*(X**2 + Y**2)**2) + 3
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def func1(D: np.ndarray, s: List[int]) -> float:
    """计算TSP路径长度"""
    n = len(s)
    DistanV = 0.0
    for i in range(n-1):
        DistanV += D[s[i], s[i+1]]
    DistanV += D[s[n-1], s[0]]
    return DistanV

def func2(x: List[float]) -> float:
    """计算函数值 f(x) = (cos(x1^2+x2^2)-0.1)/(1+0.3*(x1^2+x2^2)^2)+3"""
    x1, x2 = x
    numerator = math.cos(x1**2 + x2**2) - 0.1
    denominator = 1 + 0.3*(x1**2 + x2**2)**2
    return numerator / denominator + 3

def TSP_TS():
    """禁忌搜索算法解决TSP问题"""
    # 城市坐标数据
    C = np.array([
        [1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399],
        [3488, 1535], [3326, 1556], [3238, 1229], [4196, 1044],
        [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
        [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678],
        [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578],
        [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376],
        [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550],
        [2545, 2357], [2778, 2826], [2370, 2975]
    ])
    
    N = C.shape[0]  # 城市数量
    D = np.zeros((N, N))  # 距离矩阵
    
    # 计算距离矩阵
    for i in range(N):
        for j in range(N):
            D[i, j] = np.sqrt((C[i, 0] - C[j, 0])**2 + (C[i, 1] - C[j, 1])**2)
    
    Tabu = np.zeros((N, N))  # 禁忌表
    TabuL = round(np.sqrt(N*(N-1)/2))  # 禁忌长度
    Ca = 200  # 候选解数量
    BestCaNum = Ca // 2  # 最优候选解数量 (保留前 Ca/2 个最好候选解)
    Gmax = 500  # 最大迭代次数
    
    # 初始化
    S0 = list(range(N)) # 当前解
    random.shuffle(S0)
    bestsofar = S0.copy()
    BestL = float('inf') # 当前最优路径长度
    ArrBestL = np.zeros(Gmax) # 记录每次迭代的最优值
    
    plt.figure(figsize=(12, 6))
    p = 0
    
    while p < Gmax:
        # 生成候选交换城市对
        A = []
        while len(A) < Ca:
            M = [random.randint(0, N-1), random.randint(0, N-1)]
            if M[0] != M[1]:
                pair = (max(M), min(M))
                if pair not in A:
                    A.append(pair)
        
        # 生成候选解
        CaNum = np.zeros((Ca, N), dtype=int)
        F = np.zeros(Ca)
        BestCa = np.full((BestCaNum, 4), float('inf')) # 最优候选解信息
        
        for i in range(Ca):
            new_route = S0.copy()
            idx1, idx2 = A[i]
            new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
            CaNum[i] = new_route
            F[i] = func1(D, new_route)
            
            # 更新最优候选解
            if i < BestCaNum:
                BestCa[i] = [i, F[i], S0[idx1], S0[idx2]]
            else:
                max_idx = np.argmax(BestCa[:, 1])
                if F[i] < BestCa[max_idx, 1]:
                    BestCa[max_idx] = [i, F[i], S0[idx1], S0[idx2]]
        
        # 排序候选解
        sorted_idx = np.argsort(BestCa[:, 1])
        BestCa = BestCa[sorted_idx]
        
        # 藐视准则和禁忌准则
        if BestCa[0, 1] < BestL:
            BestL = BestCa[0, 1]
            S0 = list(CaNum[int(BestCa[0, 0])])
            bestsofar = S0.copy()
            Tabu = np.maximum(Tabu - 1, 0) # Tabu中所有值减 1
            Tabu[int(BestCa[0, 2]), int(BestCa[0, 3])] = TabuL
        else:
            for i in range(BestCaNum):
                city1 = int(BestCa[i, 2])
                city2 = int(BestCa[i, 3])
                if Tabu[city1, city2] == 0:
                    S0 = list(CaNum[int(BestCa[i, 0])])
                    Tabu = np.maximum(Tabu - 1, 0)
                    Tabu[city1, city2] = TabuL
                    break
        
        ArrBestL[p] = BestL
        
        # 绘制当前最优路径
        plt.clf()
        plt.subplot(1, 2, 1)
        x_coords = [C[city, 0] for city in bestsofar] + [C[bestsofar[0], 0]]
        y_coords = [C[city, 1] for city in bestsofar] + [C[bestsofar[0], 1]]
        plt.plot(x_coords, y_coords, 'bo-')
        plt.title(f'优化最短距离: {BestL:.2f}')
        
        # 绘制适应度曲线
        plt.subplot(1, 2, 2)
        plt.plot(ArrBestL[:p+1])
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title('适应度进化曲线')
        plt.pause(0.005)
        
        p += 1
    
    plt.show()
    return bestsofar, BestL

# 标准禁忌算法在处理连续问题时效果欠佳
def Continuous_TS():
    """禁忌搜索算法求函数极值"""
    xu = 5.0  # 上界
    xl = -5.0  # 下界
    L = random.randint(5, 11)  # 禁忌长度
    Ca = 5  # 邻域解个数
    Gmax = 200  # 最大迭代次数
    w = 1.0  # 自适应权重系数
    tabu = []  # 禁忌表
    
    # 初始解
    x0 = [random.uniform(xl, xu) for _ in range(2)]
    bestsofar = {'key': x0, 'value': func2(x0)}
    xnow = [{'key': x0, 'value': func2(x0)}]
    trace = np.zeros(Gmax+1)  # 增加一个位置防止索引越界
    
    g = 0
    while g < Gmax:
        x_near = []
        w *= 0.998
        
        # 生成邻域解
        for i in range(Ca):
            x_temp = xnow[g]['key']
            # 生成新解并处理边界
            x1 = x_temp[0] + (2 * random.random() - 1) * w * (xu - xl)
            x1 = max(min(x1, xu), xl)
            x2 = x_temp[1] + (2 * random.random() - 1) * w * (xu - xl)
            x2 = max(min(x2, xu), xl)
            x_near.append([x1, x2])
        
        # 计算邻域解的函数值
        fitvalue_near = [func2(point) for point in x_near]
        temp = np.argmax(fitvalue_near)
        candidate = {'key': x_near[temp], 'value': fitvalue_near[temp]}
        
        # 评价函数差
        delta1 = candidate['value'] - xnow[g]['value']
        delta2 = candidate['value'] - bestsofar['value']
        
        if delta1 <= 0:  # 候选解没有改进
            xnow.append({'key': candidate['key'], 'value': func2(candidate['key'])})
            tabu.append(candidate['key'])
            if len(tabu) > L:
                tabu.pop(0)
            trace[g] = bestsofar['value']
            g += 1
        else:
            if delta2 > 0:  # 候选解优于当前最优解
                xnow.append({'key': candidate['key'], 'value': func2(candidate['key'])})
                tabu.append(candidate['key'])
                if len(tabu) > L:
                    tabu.pop(0)
                bestsofar = {'key': candidate['key'], 'value': candidate['value']}
                trace[g] = bestsofar['value']
                g += 1
            else:  # 候选解优于当前解但不如最优解
                in_tabu = any(np.allclose(candidate['key'], t) for t in tabu)
                if not in_tabu:
                    xnow.append({'key': candidate['key'], 'value': func2(candidate['key'])})
                    tabu.append(xnow[g]['key'])
                    if len(tabu) > L:
                        tabu.pop(0)
                    trace[g] = bestsofar['value']
                    g += 1
                else:
                    trace[g] = bestsofar['value']
                    g += 1
    
    # 绘制结果
    plt.figure()
    plt.plot(trace[:g])
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title('搜索过程最优值曲线')
    plt.show()
    
    return bestsofar

# 主程序
if __name__ == "__main__":
    # 选择要运行的功能
    print("1. 绘制函数图像")
    print("2. TSP禁忌搜索")
    print("3. 函数极值搜索")
    
    choice = input("请选择要运行的程序 (1-3): ")
    
    if choice == '1':
        plot_function()
    elif choice == '2':
        best_route, min_distance = TSP_TS()
        print(f"最优路径: {best_route}")
        print(f"最短距离: {min_distance:.2f}")
    elif choice == '3':
        result = Continuous_TS()
        print(f"找到最优解: x = {result['key']}, f(x) = {result['value']}")
    else:
        print("无效选择")