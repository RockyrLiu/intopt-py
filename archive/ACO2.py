import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_function import Rastrigin, Square, func2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class ACO_Continuous:
    """连续空间蚁群算法(仅供参考, 效果欠佳)
    
    支持以下策略:
    1. 'AS' - 基本蚁群算法
    2. 'EAS' - 精英蚁群算法
    3. 'MMAS' - 最大最小蚁群算法
    4. 'AAS' - 自适应蚁群算法
    """
    def __init__(self, func, bounds, m=50, maxiter=100, alpha=1, beta=5,
                 rho=0.1, Q=100, strategy='AS', P0=0.2, step=0.1,
                 init_positions=None, verbose=True, plot=True):
        """
        参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        m: 蚂蚁数量
        maxiter: 最大迭代次数
        alpha: 信息素重要程度
        beta: 启发式因子重要程度
        rho: 信息素蒸发系数
        Q: 信息素增量系数
        strategy: 算法策略(见类文档)
        P0: 局部搜索概率阈值
        step: 局部搜索步长
        init_positions: 初始解列表(可选)
        verbose: 是否显示进度条
        plot: 是否绘制结果
        """
        self.function = func
        self.bounds = bounds
        self.dim = len(bounds)
        self.lower_bound = np.array([b[0] for b in bounds])
        self.upper_bound = np.array([b[1] for b in bounds])
        self.m = m
        self.maxiter = maxiter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.strategy = strategy
        self.P0 = P0
        self.step = step
        self.init_positions = init_positions
        self.verbose = verbose
        self.plot = plot
        
        # 算法状态
        self.positions = None  # 蚂蚁位置矩阵(dim, m)
        self.Tau = None       # 信息素
        
        # 历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': []
        }
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _init_positions(self):
        """初始化蚂蚁位置"""
        self.positions = np.zeros((self.dim, self.m))
        
        # 如果有初始解，优先使用
        if self.init_positions is not None and len(self.init_positions) > 0:
            num_init = min(len(self.init_positions), self.m)
            for i in range(num_init):
                self.positions[:, i] = np.array(self.init_positions[i][:self.dim])
        
        # 剩余蚂蚁随机初始化
        for d in range(self.dim):
            lb = self.lower_bound[d]
            ub = self.upper_bound[d]
            for i in range(len(self.init_positions) if self.init_positions else 0, self.m):
                self.positions[d, i] = np.random.uniform(lb, ub)
    
    def _clip_position(self, position):
        """确保位置在边界范围内"""
        return np.clip(position, self.lower_bound, self.upper_bound)
    
    def _evaluate(self):
        """评估目标函数值"""
        return np.array([self.function(self.positions[:, i]) for i in range(self.m)])
    
    def _update_pheromone_AS(self):
        """基本蚁群算法信息素更新"""
        fitness = self._evaluate()
        self.Tau = (1 - self.rho) * self.Tau + self.Q / (fitness + 1e-10)
    
    def _update_pheromone_EAS(self):
        """精英蚁群算法信息素更新"""
        fitness = self._evaluate()
        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]
        
        # 普通蚂蚁信息素更新
        self.Tau = (1 - self.rho) * self.Tau + self.Q / (fitness + 1e-10)
        
        # 精英蚂蚁额外信息素
        elite_delta = np.zeros(self.m)
        elite_delta[best_idx] = 2.0 * self.Q / best_fit  # 精英权重为2.0
        self.Tau += elite_delta
    
    def _update_pheromone_MMAS(self):
        """最大最小蚁群算法信息素更新"""
        fitness = self._evaluate()
        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]
        
        # 只更新最优解路径上的信息素
        tau_max = 2.0
        tau_min = 0.001
        self.Tau = (1 - self.rho) * self.Tau
        self.Tau[best_idx] += self.Q / best_fit
        
        # 应用信息素上下限
        self.Tau = np.clip(self.Tau, tau_min, tau_max)
    
    def _update_pheromone_AAS(self):
        """自适应蚁群算法信息素更新"""
        fitness = self._evaluate()
        avg_fit = np.mean(fitness)
        best_fit = np.min(fitness)
        
        # 自适应调整rho
        ratio = best_fit / avg_fit
        self.rho = 0.01 + (0.5 - 0.01) * ratio  # rho在0.01-0.5之间自适应
        
        # 更新信息素
        self.Tau = (1 - self.rho) * self.Tau + self.Q / (fitness + 1e-10)
    
    def _move_ants(self):
        """蚂蚁移动"""
        current_iter = getattr(self, 'current_iter', 0)
        lamda = 1 / (current_iter + 1)  # 动态调整参数
        
        best_idx = np.argmin(self.Tau)
        Tau_best = self.Tau[best_idx]
        new_positions = np.copy(self.positions)
        
        for i in range(self.m):
            P = (Tau_best - self.Tau[i]) / (Tau_best + 1e-10)
            
            if P < self.P0:
                # 局部搜索
                delta = (2 * np.random.rand(self.dim) - 1)
                new_positions[:, i] += delta * self.step * lamda
            else:
                # 全局搜索
                for d in range(self.dim):
                    range_d = self.upper_bound[d] - self.lower_bound[d]
                    new_positions[d, i] += (np.random.rand() - 0.5) * range_d
            
            new_positions[:, i] = self._clip_position(new_positions[:, i])
        
        # 评估新旧位置
        new_values = np.array([self.function(new_positions[:, i]) for i in range(self.m)])
        old_values = np.array([self.function(self.positions[:, i]) for i in range(self.m)])
        
        # 只保留更好的解
        for i in range(self.m):
            if new_values[i] < old_values[i]:
                self.positions[:, i] = new_positions[:, i]
        
        self.current_iter = current_iter + 1
    
    def _update_history(self):
        """更新历史记录"""
        fitness = self._evaluate()
        current_best = np.min(fitness)
        current_avg = np.mean(fitness)
        
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            self.best_solution = self.positions[:, np.argmin(fitness)].copy()
        
        self.history['best_fitness'].append(self.best_fitness)
        self.history['avg_fitness'].append(current_avg)
    
    def iterator(self):
        """执行优化迭代"""
        # 初始化
        self._init_positions()
        self.Tau = self._evaluate()
        
        if self.verbose:
            pbar = tqdm(total=self.maxiter, desc="ACO优化进度")
        
        for _ in range(self.maxiter):
            # 蚂蚁移动
            self._move_ants()
            
            # 信息素更新
            if self.strategy == 'AS':
                self._update_pheromone_AS()
            elif self.strategy == 'EAS':
                self._update_pheromone_EAS()
            elif self.strategy == 'MMAS':
                self._update_pheromone_MMAS()
            elif self.strategy == 'AAS':
                self._update_pheromone_AAS()
            else:
                raise ValueError(f"未知策略: {self.strategy}")
            
            # 更新历史记录
            self._update_history()
            
            # 更新进度条
            if self.verbose:
                pbar.set_postfix({
                    '最优值': f"{self.best_fitness:.6f}",
                    '平均值': f"{np.mean(self._evaluate()):.6f}",
                    '策略': self.strategy
                })
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return self.history
    
    def run(self):
        """运行优化"""
        history = self.iterator()
        
        if self.plot:
            self._plot_results(history)
        
        return self.best_solution, history
    
    def _plot_results(self, history):
        """绘制结果"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(history['avg_fitness'], 'r--', label='平均适应度')
        plt.xlabel('迭代次数', size=12)
        plt.ylabel('适应度值', size=12)
        plt.title(f'ACO优化过程 ({self.strategy}) - 适应度变化', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def aco(func, bounds, m=50, maxiter=100, alpha=1, beta=5,
        rho=0.1, Q=100, strategy='AS', P0=0.2, step=0.1,
        init_positions=None, verbose=True, plot=True):
    """
    连续空间蚁群算法
    
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        m: 蚂蚁数量 (默认50)
        maxiter: 最大迭代次数 (默认100)
        alpha: 信息素重要程度 (默认1)
        beta: 启发式因子重要程度 (默认5)
        rho: 信息素蒸发系数 (默认0.1)
        Q: 信息素增量系数 (默认100)
        strategy: 算法策略 ('AS', 'EAS', 'MMAS', 'AAS') (默认'AS')
        P0: 局部搜索概率阈值 (默认0.2)
        step: 局部搜索步长 (默认0.1)
        init_positions: 初始解列表 (可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 包含'best_fitness'和'avg_fitness'的历史记录
    """
    optimizer = ACO_Continuous(
        func=func,
        bounds=bounds,
        m=m,
        maxiter=maxiter,
        alpha=alpha,
        beta=beta,
        rho=rho,
        Q=Q,
        strategy=strategy,
        P0=P0,
        step=step,
        init_positions=init_positions,
        verbose=verbose,
        plot=plot
    )
    return optimizer.run()


def main():
    """测试函数"""
    def sphere(X):
        return np.sum(X**2)
    
    # 测试所有策略
    strategies = ['AS', 'EAS', 'MMAS', 'AAS']
    
    print("连续空间ACO测试:")
    for strategy in strategies:
        print(f"\n策略: {strategy}")
        best_solution, history = aco(
            func=sphere,
            bounds=[(-5.12, 5.12), (-5.12, 5.12)],
            m=30,
            maxiter=100,
            strategy=strategy,
            verbose=True
        )
        print("最优解:", best_solution)
        print("最优值:", history['best_fitness'][-1])

if __name__ == "__main__":
    main()