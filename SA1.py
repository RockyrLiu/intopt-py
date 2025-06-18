# 本代码参考秦喜文《数学建模（Python）版》，147-158
import math 
import numpy as np
from random import uniform, random
import matplotlib.pyplot as plt
from test_function import func1

class SA:
    def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = func
        self.iter = iter
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0
        self.x = [uniform(-5, 5) for _ in range(100)] # 生成100个在[-5,5]之间的随机数
        self.y = [uniform(-5, 5) for _ in range(100)]
        self.most_best = []
        self.history = {'f':[], 'T':[]}

    def generate_new(self, x, y):
        """重复产生新解， 直到满足条件"""
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) and (-5 <= y_new <= 5):
                break

        return x_new, y_new
    
    def Metrospolis(self, f, f_new):
        """Metrospolis准则"""
        if f_new <= f:
            return 1
        else:
            p = math.exp((f-f_new)/self.T)
            return 1 if random() < p else 0
        
    def best(self):
        """获取最优目标函数"""
        f_list = []
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)
        idx = f_list.index(f_best)
        return f_best, idx  # 当前温度下，迭代L次后目标函数的最优解及其下标
    
    def run(self):
        count = 0
        while self.T > self.Tf: # 终止条件
            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])
                x_new, y_new = self.generate_new(self.x[i], self.y[i])
                f_new = self.func(x_new, y_new)
                if self.Metrospolis(f, f_new):
                    self.x[i] = x_new
                    self.y[i] = y_new
            # 迭代L次后，记录在该温度下的最优解
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            # 降温
            self.T = self.T * self.alpha
            count += 1

        f_best, idx = self.best()
        print(f"F={f_best:.4f}, x1={self.x[idx]:.4f}, x2={self.y[idx]:.4f}")


if __name__ == "__main__":
    def func(x1, x2):
        """将func1适配为接受x1,x2参数的函数"""
        return func1(np.array([[x1, x2]]))[0]

    sa = SA(func)
    sa.run()

    # 可视化
    plt.plot(sa.history['T'], sa.history['f'])
    plt.title('SA')
    plt.xlabel('T')
    plt.ylabel('f')
    plt.gca().invert_xaxis() # 反转当前坐标轴 X 轴方向
    # plt.savefig(r"E:\数模\my_algorithm\优化算法\SA1.png", dpi=800,)
    plt.show()
    







