import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class PSO:
    """粒子群优化算法"""
    def __init__(self, func, bounds, popsize=30, maxiter=100, 
                 w=0.8, w_min=0.4, c1=2, c2=2, v_max=2,
                 init_positions=None, verbose=True):
        """
        参数说明:
        func: 目标函数
        bounds: 边界列表，每个元素是元组 (min, max)
        popsize: 粒子数量
        maxiter: 最大迭代次数
        w: 初始惯性权重
        w_min: 最小惯性权重
        c1, c2: 学习因子
        v_max: 最大速度限制
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条
        """
        # 设置目标函数
        self.function = wrapper_black_box(func)
        self.verbose = verbose
        
        # 处理边界参数
        self.dim = len(bounds)
        self.lower_bound = np.array([b[0] for b in bounds])
        self.upper_bound = np.array([b[1] for b in bounds])
        
        # 算法参数
        self.w = w
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.popsize = popsize
        self.maxiter = maxiter
        
        # 初始化粒子位置和速度
        if init_positions is not None:
            assert init_positions.shape == (popsize, self.dim), "初始位置形状错误"
            self.X = init_positions.copy()
        else:
            self.X = np.zeros((popsize, self.dim))
            for d in range(self.dim):
                self.X[:, d] = np.random.uniform(self.lower_bound[d], self.upper_bound[d], popsize)
        
        self.V = np.random.uniform(-v_max, v_max, (popsize, self.dim))
        
        # 初始化最优位置和适应度
        self.pbest = self.X.copy()
        self.p_fit = self.function(self.X)
        min_idx = np.argmin(self.p_fit)
        self.fit = self.p_fit[min_idx]
        self.gbest = self.X[min_idx].reshape(1, -1).copy()
        
        # 历史记录（最优值和平均值）
        self.history = {
            'best_fitness': [self.fit],
            'avg_fitness': [np.mean(self.p_fit)]
        }
    
    def clamp_position(self, position):
        """位置钳制"""
        clamped = position.copy()
        for d in range(self.dim):
            clamped[:, d] = np.clip(position[:, d], self.lower_bound[d], self.upper_bound[d])
        return clamped
    
    def clamp_velocity(self, velocity):
        """速度钳制"""
        return np.clip(velocity, -self.v_max, self.v_max)
    
    def iterator(self):
        """执行优化迭代"""
        # 创建进度条
        if self.verbose:
            pbar = tqdm(total=self.maxiter, desc="PSO优化进度")
        
        for t in range(self.maxiter):
            # 动态衰减惯性权重
            self.w = self.w_min + (self.w - self.w_min) * (self.maxiter - t) / self.maxiter
            
            # 生成随机因子矩阵
            r1 = np.random.rand(self.popsize, self.dim)
            r2 = np.random.rand(self.popsize, self.dim)
            
            # 更新速度和位置
            gbest_matrix = np.repeat(self.gbest, self.popsize, axis=0)
            self.V = self.w * self.V + self.c1 * r1 * (self.pbest - self.X) + self.c2 * r2 * (gbest_matrix - self.X)
            self.V = self.clamp_velocity(self.V)
            self.X = self.clamp_position(self.X + self.V)
            
            # 计算新适应度
            current_fit = self.function(self.X)
            
            # 更新个体和全局最优
            update_mask = current_fit < self.p_fit
            self.pbest[update_mask] = self.X[update_mask]
            self.p_fit[update_mask] = current_fit[update_mask]
            
            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = current_fit[min_idx]
                self.gbest = self.X[min_idx].reshape(1, -1)
            
            # 记录历史
            self.history['best_fitness'].append(self.fit)
            self.history['avg_fitness'].append(np.mean(current_fit))
            
            # 更新进度条
            if self.verbose:
                pbar.set_postfix({
                    '最优值': f"{self.fit:.6f}",
                    '平均值': f"{np.mean(current_fit):.6f}"
                })
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return self.history
    
    def run(self):
        return self.iterator()

def wrapper_black_box(black_box_func):
    """封装黑箱函数"""
    def adapted_func(X):
        return np.array([black_box_func(x.tolist()) for x in X])
    return adapted_func

class CPSO(PSO):
    """混沌粒子群优化算法"""
    def __init__(self, func, bounds, popsize=30, maxiter=100, 
                 w=0.8, w_min=0.4, c1=2, c2=2, v_max=2, chaos_reset_ratio=0.2,
                 init_positions=None, verbose=True):
        """
        参数说明:
        func: 目标函数
        bounds: 边界列表，每个元素是元组 (min, max)
        popsize: 粒子数量
        maxiter: 最大迭代次数
        w: 初始惯性权重
        w_min: 最小惯性权重
        c1, c2: 学习因子
        v_max: 最大速度限制
        chaos_reset_ratio: 每代中重置的最差粒子比例
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条
        """
        # 混沌初始化粒子位置
        if init_positions is None:
            init_positions = self.chaos_initialization(popsize, len(bounds), bounds)
        
        # 调用父类初始化
        super().__init__(
            func=func,
            bounds=bounds,
            popsize=popsize,
            maxiter=maxiter,
            w=w,
            w_min=w_min,
            c1=c1,
            c2=c2,
            v_max=v_max,
            init_positions=init_positions,
            verbose=verbose
        )
        
        self.chaos_reset_ratio = chaos_reset_ratio
    
    def chaos_initialization(self, popsize, dim, bounds):
        """使用Logistic混沌映射生成初始种群"""
        X = np.zeros((popsize, dim))
        for i in range(popsize):
            # 生成混沌序列
            chaos_seq = self.generate_chaos_sequence(dim)
            # 映射到搜索空间 - 每个维度使用自己的范围
            for d in range(dim):
                X[i, d] = bounds[d][0] + chaos_seq[d] * (bounds[d][1] - bounds[d][0])
        return X
    
    def generate_chaos_sequence(self, dim):
        """生成混沌序列(Logistic映射)"""
        # 随机初始化混沌种子(避免0.5)
        seed = np.random.rand()
        while seed == 0.5:  # 避免不动点
            seed = np.random.rand()
        
        sequence = []
        x = seed
        for _ in range(dim):
            x = 4 * x * (1 - x)  # Logistic映射(μ=4)
            sequence.append(x)
        return np.array(sequence)
    
    def iterator(self):
        """执行优化迭代(添加混沌重置)"""
        # 创建进度条
        if self.verbose:
            pbar = tqdm(total=self.maxiter, desc="CPSO优化进度")
        
        for t in range(self.maxiter):
            # 动态衰减惯性权重
            self.w = self.w_min + (self.w - self.w_min) * (self.maxiter - t) / self.maxiter
            
            # 生成随机因子矩阵
            r1 = np.random.rand(self.popsize, self.dim)
            r2 = np.random.rand(self.popsize, self.dim)
            
            # 更新速度和位置
            gbest_matrix = np.repeat(self.gbest, self.popsize, axis=0)
            self.V = self.w * self.V + self.c1 * r1 * (self.pbest - self.X) + self.c2 * r2 * (gbest_matrix - self.X)
            self.V = self.clamp_velocity(self.V)
            self.X = self.clamp_position(self.X + self.V)
            
            # 计算新适应度
            current_fit = self.function(self.X)
            
            # 更新个体和全局最优
            update_mask = current_fit < self.p_fit
            self.pbest[update_mask] = self.X[update_mask]
            self.p_fit[update_mask] = current_fit[update_mask]
            
            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = current_fit[min_idx]
                self.gbest = self.X[min_idx].reshape(1, -1)
            
            # ======== 混沌重置步骤 ========
            # 计算需要重置的粒子数量
            num_reset = max(1, int(self.popsize * self.chaos_reset_ratio))
            
            # 选择适应度最差的粒子
            worst_indices = np.argsort(current_fit)[-num_reset:]
            
            for i in worst_indices:
                # 混沌重置粒子位置
                chaos_seq = self.generate_chaos_sequence(self.dim)
                for d in range(self.dim):
                    self.X[i, d] = self.lower_bound[d] + chaos_seq[d] * (self.upper_bound[d] - self.lower_bound[d])
                
                # 重置粒子速度
                self.V[i] = np.random.uniform(-self.v_max, self.v_max, self.dim)
                
                # 计算新位置的适应度
                new_fit = self.function(self.X[i].reshape(1, -1))[0]
                current_fit[i] = new_fit
                
                # 更新个体最优
                if new_fit < self.p_fit[i]:
                    self.p_fit[i] = new_fit
                    self.pbest[i] = self.X[i].copy()
            
            # 更新全局最优(重置后可能变化)
            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = current_fit[min_idx]
                self.gbest = self.X[min_idx].reshape(1, -1)
            # ======== 混沌重置结束 ========
            
            # 记录历史
            self.history['best_fitness'].append(self.fit)
            self.history['avg_fitness'].append(np.mean(current_fit))
            
            # 更新进度条
            if self.verbose:
                pbar.set_postfix({
                    '最优值': f"{self.fit:.6f}",
                    '平均值': f"{np.mean(current_fit):.6f}"
                })
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return self.history

class ElitePSO:
    """精英粒子群优化算法"""
    def __init__(self, func, bounds, n_runs=10, popsize=30, maxiter=100,
                 w=0.8, w_min=0.4, c1=2, c2=2, v_max=2,
                 verbose=True, level=1):
        """
        参数说明:
        func: 目标函数
        bounds: 边界列表，每个元素是元组 (min, max)
        n_runs: 独立运行次数
        popsize: 每次运行的粒子数
        maxiter: 每次运行的迭代次数
        w: 初始惯性权重
        w_min: 最小惯性权重
        c1, c2: 学习因子
        v_max: 最大速度限制
        verbose: 是否显示进度信息
        level: 当前层级(用于多层结构)
        """
        self.function = func
        self.bounds = bounds
        self.dim = len(bounds)
        self.n_runs = n_runs
        self.popsize = popsize
        self.maxiter = maxiter
        self.w = w
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.verbose = verbose
        self.level = level
        
        self.lower_bound = np.array([b[0] for b in bounds])
        self.upper_bound = np.array([b[1] for b in bounds])
        
        self.gbest = None
        self.fit = float('inf')
        self.history = {
            'best_fitness': [],
            'avg_fitness': []
        }
    
    def run_multiple_optimizers(self):
        """运行多个独立PSO优化器"""
        all_best_positions = []
        all_best_fitness = []
        
        if self.verbose and self.level == 1:
            print(f"\n{'='*50}")
            print(f"顶层开始运行 {self.n_runs} 次独立优化器")
            print("="*50)
        
        disable_progress = not self.verbose or self.level > 1
        desc = "顶层优化进度" if self.verbose else None
        
        for _ in tqdm(range(self.n_runs), desc=desc, disable=disable_progress):
            pso = PSO(
                func=self.function,
                bounds=self.bounds,
                popsize=self.popsize,
                maxiter=self.maxiter,
                w=self.w,
                w_min=self.w_min,
                c1=self.c1,
                c2=self.c2,
                v_max=self.v_max,
                verbose=False
            )
            history = pso.run()
            
            all_best_positions.append(pso.gbest[0])
            all_best_fitness.append(pso.fit)
        
        best_of_all_idx = np.argmin(all_best_fitness)
        global_best_position = all_best_positions[best_of_all_idx]
        global_best_fitness = all_best_fitness[best_of_all_idx]
        
        if self.verbose and self.level == 1:
            print(f"\n独立运行完成 (共{self.n_runs}次)")
            print(f"最佳适应度值: {global_best_fitness:.6f}")
            print("-"*50)
        
        return np.array(all_best_positions), global_best_position
    
    def run(self):
        """执行精英PSO优化"""
        best_positions, global_best = self.run_multiple_optimizers()
        
        self.gbest = global_best
        self.fit = self.function(global_best.reshape(1, -1))[0]
        
        if self.verbose and self.level == 1:
            print(f"开始精英PSO优化 (迭代次数: {self.maxiter})")
            print("-"*50)
        
        elite_pso = PSO(
            func=self.function,
            bounds=self.bounds,
            popsize=self.n_runs,  # 使用独立运行次数作为粒子数
            maxiter=self.maxiter,
            w=self.w,
            w_min=self.w_min,
            c1=self.c1,
            c2=self.c2,
            v_max=self.v_max,
            init_positions=best_positions,
            verbose=self.verbose and self.level == 1
        )
        
        history = elite_pso.run()
        self.history = history
        
        self.gbest = elite_pso.gbest[0]
        self.fit = elite_pso.fit
        
        if self.verbose and self.level == 1:
            print("\n优化完成")
            print(f"最终最优解: {np.round(self.gbest, 6)}")
            print(f"最终适应度: {self.fit:.6f}")
            print("="*50)
        
        return history


def pso(func, bounds, popsize=30, maxiter=100, 
        w=0.8, w_min=0.4, c1=2, c2=2, v_max=2,
        init_positions=None, verbose=True, plot=True):
    """
    粒子群优化(连续优化问题)
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        popsize: 粒子数量 (默认30)
        maxiter: 最大迭代次数 (默认100)
        w: 初始惯性权重 (默认0.8)
        w_min: 最小惯性权重 (默认0.4)
        c1, c2: 学习因子 (默认2, 2)
        v_max: 最大速度限制 (默认2)
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 历史记录
    """
    optimizer = PSO(
        func=func,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        w=w,
        w_min=w_min,
        c1=c1,
        c2=c2,
        v_max=v_max,
        init_positions=init_positions,
        verbose=verbose
    )
    history = optimizer.run()
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(history['avg_fitness'], 'r--', label='平均适应度')
        plt.xlabel('迭代次数', size=12)
        plt.ylabel('适应度值', size=12)
        plt.title('PSO优化过程 - 适应度变化', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return optimizer.gbest[0], history

def cpso(func, bounds, popsize=30, maxiter=100, 
        w=0.8, w_min=0.4, c1=2, c2=2, v_max=2, chaos_reset_ratio=0.2,
        init_positions=None, verbose=True, plot=True):
    """
    混沌粒子群优化(连续优化问题)
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        popsize: 粒子数量 (默认30)
        maxiter: 最大迭代次数 (默认100)
        w: 初始惯性权重 (默认0.8)
        w_min: 最小惯性权重 (默认0.4)
        c1, c2: 学习因子 (默认2, 2)
        v_max: 最大速度限制 (默认2)
        chaos_reset_ratio: 每代中重置的最差粒子比例 (默认0.2)
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 历史记录
    """
    optimizer = CPSO(
        func=func,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        w=w,
        w_min=w_min,
        c1=c1,
        c2=c2,
        v_max=v_max,
        chaos_reset_ratio=chaos_reset_ratio,
        init_positions=init_positions,
        verbose=verbose,
    )
    history = optimizer.run()
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(history['avg_fitness'], 'r--', label='平均适应度')
        plt.xlabel('迭代次数', size=12)
        plt.ylabel('适应度值', size=12)
        plt.title('CPSO优化过程 - 适应度变化', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return optimizer.gbest[0], history

def elite_pso(func, bounds, n_runs=10, popsize=30, maxiter=100,
             w=0.8, w_min=0.4, c1=2, c2=2, v_max=2,
             verbose=True, plot=True):
    """
    精英粒子群(连续优化问题)
    参数:
        func: 目标函数
        bounds: 边界列表 [(min, max), ...]
        n_runs: 独立运行次数 (默认10)
        popsize: 粒子数量 (默认30)
        maxiter: 最大迭代次数 (默认100)
        w: 初始惯性权重 (默认0.8)
        w_min: 最小惯性权重 (默认0.4)
        c1, c2: 学习因子 (默认2, 2)
        v_max: 最大速度限制 (默认2)
        init_positions: 初始位置数组 (可选)
        verbose: 是否显示进度条 (默认True)
        plot: 是否绘制结果 (默认True)
    
    返回:
        best_solution: 最优解
        history: 历史记录
    """
    optimizer = ElitePSO(
        func=func,
        bounds=bounds,
        n_runs=n_runs,
        popsize=popsize,
        maxiter=maxiter,
        w=w,
        w_min=w_min,
        c1=c1,
        c2=c2,
        v_max=v_max,
        verbose=verbose
    )
    history = optimizer.run()
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(history['best_fitness'], 'b-', label='最优适应度')
        plt.plot(history['avg_fitness'], 'r--', label='平均适应度')
        plt.xlabel('迭代次数', size=12)
        plt.ylabel('适应度值', size=12)
        plt.title('精英PSO优化过程 - 适应度变化', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return optimizer.gbest, history

def multi_layer_pso(func, bounds, n_runs=10, popsize=30, maxiter=100, layers=3,
                  w=0.8, w_min=0.4, c1=2, c2=2, v_max=2, verbose=True):
    """
    多层精英PSO优化算法
    
    参数:
        func: 目标函数 (接受向量输入，返回标量值)
        bounds: 变量边界列表 [(min, max), ...]
        n_runs: 每层独立运行次数 (默认10)
        popsize: 每个独立运行的粒子数 (默认30)
        maxiter: 每层最大迭代次数 (默认100)
        layers: 层级数 (默认3)
        w: 初始惯性权重 (默认0.8)
        w_min: 最小惯性权重 (默认0.4)
        c1: 个体学习因子 (默认2)
        c2: 社会学习因子 (默认2)
        v_max: 最大速度限制 (默认2)
        verbose: 是否显示优化过程 (默认True)
    
    返回:
        elite_history: 顶层优化历史记录 (字典包含:
            - 'best_fitness': 各代最优适应度
            - 'avg_fitness': 各代平均适应度)
        top_layer: 顶层优化器实例 (可获取全局最优解等详细信息)
    """
    print("\n" + "="*50)
    print(f"开始多层精英PSO优化 (共{layers}层)")
    print(f"配置: {n_runs}次独立运行, 每次{popsize}个粒子, {maxiter}次迭代")
    print("="*50)
    
    current_layer = None
    for layer in range(layers, 0, -1):
        is_top_layer = (layer == 1)
        if current_layer is None:
            layer_optimizer = ElitePSO(
                func=func,
                bounds=bounds,
                n_runs=n_runs,
                popsize=popsize,
                maxiter=maxiter,
                w=w,
                w_min=w_min,
                c1=c1,
                c2=c2,
                v_max=v_max,
                verbose=verbose and is_top_layer,
                level=layer
            )
        else:
            layer_optimizer = ElitePSO(
                func=func,
                bounds=bounds,
                n_runs=n_runs,
                popsize=popsize,
                maxiter=maxiter,
                w=w,
                w_min=w_min,
                c1=c1,
                c2=c2,
                v_max=v_max,
                verbose=verbose and is_top_layer,
                level=layer
            )
        current_layer = layer_optimizer
    
    top_layer = current_layer
    elite_history = top_layer.run()

    return elite_history, top_layer


def main1():
    # 定义原始黑箱函数（输入为列表）
    def square(X):
        return X[0] **2 + X[1] ** 2 + X[2] ** 2

    # 调用PSO优化
    best_solution, history = pso(
        func=square,  
        bounds=[(-5, 5), (-5, 5), (-5, 5)],  
        popsize=50,
        maxiter=100
    )

    print("最优解:", best_solution)

def main2():
    def rastrigin(X):
        A = 10
        return A * len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])
    
    # 调用CPSO优化
    best_solution, history = cpso(
        func=rastrigin,
        bounds=[(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)],
        popsize=50,
        maxiter=100,
        chaos_reset_ratio=0.3
    )
    
    print("最优解:", best_solution)

def main3():
    # 测试函数
    def rastrigin(X):
        A = 10
        return A * len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])
    
    # 调用多层精英PSO
    elite_history, top_layer = multi_layer_pso(
        func=rastrigin,
        bounds=[(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)],
        n_runs=30,
        popsize=30,
        maxiter=50,
        layers=3,
        verbose=True
    )
    
    # 运行标准PSO对比
    print("运行标准PSO作为基准对比...")
    standard_pso = PSO(
        func=rastrigin,
        bounds=[(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)],
        popsize=30,
        maxiter=50,
        verbose=False
    )
    standard_history = standard_pso.run()
    
    # 结果对比
    print("\n优化结果对比:")
    print(f"单次标准PSO最优适应度: {standard_pso.fit:.6f}")
    print(f"多层精英PSO最优适应度: {top_layer.fit:.6f}")
    improvement = (standard_pso.fit - top_layer.fit) / standard_pso.fit * 100
    print(f"优化提升: {improvement:.2f}%")
    print("="*50)
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("标准PSO优化过程", fontsize=14)
    plt.xlabel("迭代次数", size=12)
    plt.ylabel("适应度值", size=12)
    plt.plot(standard_history['best_fitness'], color='b', linewidth=2)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title("精英PSO优化过程", fontsize=14)
    plt.xlabel("迭代次数", size=12)
    plt.ylabel("适应度值", size=12)
    plt.plot(elite_history['best_fitness'], color='r', linewidth=2)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main3()