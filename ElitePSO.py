import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from PSO1 import PSO, fitness2

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ElitePSO:
    """精英初始化PSO"""
    def __init__(self, dim, n_runs=10, particles=30, iterations=50, objective_func=None,
                 base_optimizer=None, verbose=True, level=1, 
                 lower_bound=-5.12, upper_bound=5.12, v_max=2):
        """
        参数:
        dim: 搜索维度
        n_runs: 独立运行次数
        particles: 每次运行的粒子数
        iterations: 每次运行的迭代次数
        objective_func: 目标函数
        base_optimizer: 基础优化器(用于多层结构)
        verbose: 是否显示进度信息
        level: 当前层级(用于多层结构)
        lower_bound: 每个维度的下界(标量或数组)
        upper_bound: 每个维度的上界(标量或数组)
        v_max: 最大速度限制
        """
        self.dim = dim
        self.n_runs = n_runs
        self.particles = particles
        self.iterations = iterations
        self.objective_func = objective_func
        self.base_optimizer = base_optimizer
        self.verbose = verbose
        self.level = level
        self.v_max = v_max
        
        # 处理边界参数
        if np.isscalar(lower_bound):
            self.lower_bound = np.full(dim, lower_bound)
        else:
            assert len(lower_bound) == dim, "下界数组长度必须与维度匹配"
            self.lower_bound = np.array(lower_bound)
            
        if np.isscalar(upper_bound):
            self.upper_bound = np.full(dim, upper_bound)
        else:
            assert len(upper_bound) == dim, "上界数组长度必须与维度匹配"
            self.upper_bound = np.array(upper_bound)
            
        self.gbest = None
        self.fit = float('inf')
        self.fitness_history = []
        self.progress_bar = None  # 添加进度条引用

    def run_multiple_optimizers(self):
        all_best_positions = []
        all_best_fitness = []
        
        if self.verbose and self.level == 1:
            print(f"\n{'='*50}")
            print(f"顶层开始运行 {self.n_runs} 次独立优化器")
            print("="*50)

        # 只在顶层显示进度条
        disable_progress = not self.verbose or self.level > 1
        desc = "顶层优化进度" if self.verbose else None
        
        # 使用tqdm的leave参数防止重复显示
        self.progress_bar = tqdm(range(self.n_runs), desc=desc, unit="次", 
                               disable=disable_progress, position=0, leave=True)

        for i in self.progress_bar:
            if self.base_optimizer:
                optimizer = deepcopy(self.base_optimizer)
                optimizer.level = self.level + 1
                optimizer.verbose = False
                optimizer.run()
                best_position = optimizer.gbest
                best_fitness = optimizer.fit
            else:
                pso = PSO(pN=self.particles, dim=self.dim, max_iter=self.iterations, 
                         lower_bound=self.lower_bound, upper_bound=self.upper_bound, 
                         v_max=self.v_max, objective_func=self.objective_func,
                         verbose=False)
                pso.run()
                best_position = pso.gbest[0]
                best_fitness = pso.fit
            
            all_best_positions.append(best_position)
            all_best_fitness.append(best_fitness)

        # 关闭进度条
        if self.progress_bar is not None:
            self.progress_bar.close()
        
        best_of_all_idx = np.argmin(all_best_fitness)
        global_best_position = all_best_positions[best_of_all_idx]
        global_best_fitness = all_best_fitness[best_of_all_idx]
        
        if self.verbose and self.level == 1:
            print(f"\n独立运行完成 (共{self.n_runs}次)")
            print(f"最佳适应度值: {global_best_fitness:.6f}")
            print("-"*50)
        
        return np.array(all_best_positions), global_best_position
    
    def run(self):
        best_positions, global_best = self.run_multiple_optimizers()
        
        self.gbest = global_best
        self.fit = self.fitness_function(global_best.reshape(1, -1))[0]
        
        if self.verbose and self.level == 1:
            print(f"开始精英PSO优化 (迭代次数: {self.iterations})")
            print("-"*50)

        elitePSO = PSO(
            pN=self.n_runs,
            dim=self.dim,
            max_iter=self.iterations,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            v_max=self.v_max,
            initial_positions=best_positions,
            objective_func=self.objective_func,
            verbose=self.verbose and self.level == 1
        )
        
        fitness_history = elitePSO.run()
        self.fitness_history = fitness_history
        
        self.gbest = elitePSO.gbest[0]
        self.fit = elitePSO.fit
        
        if self.verbose and self.level == 1:
            print("\n优化完成")
            print(f"最终最优解: {np.round(self.gbest, 6)}")
            print(f"最终适应度: {self.fit:.6f}")
            print("="*50)
        
        return fitness_history
    
    def fitness_function(self, X):
        if self.objective_func is None:
            raise ValueError("未提供目标函数 (objective_func 不能为 None)")
        return self.objective_func(X)

def multiLayerPSO(dim, n_runs, particles, iterations, layers, objective_func,
                 lower_bound=-5.12, upper_bound=5.12, v_max=2):
    """多层PSO"""
    # 采用递归构建多层结构
    print("\n" + "="*50)
    print(f"开始多层精英PSO优化 (共{layers}层)")
    print(f"配置: {n_runs}次独立运行, 每次{particles}个粒子, {iterations}次迭代")
    print("="*50)
    
    current_layer = None
    for layer in range(layers, 0, -1):
        is_top_layer = (layer == 1)
        if current_layer is None:
            layer_optimizer = ElitePSO(
                dim=dim,
                n_runs=n_runs,
                particles=particles,
                iterations=iterations,
                objective_func=objective_func,
                verbose=is_top_layer,
                level=layer,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                v_max=v_max
            )
        else:
            layer_optimizer = ElitePSO(
                dim=dim,
                n_runs=n_runs,
                particles=particles,
                iterations=iterations,
                base_optimizer=current_layer,
                objective_func=objective_func,
                verbose=is_top_layer,
                level=layer,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                v_max=v_max
            )
        current_layer = layer_optimizer
    
    top_layer = current_layer
    elite_history = top_layer.run()

    return elite_history, top_layer

def plot_comparison(original_history, elite_history):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("标准PSO优化过程", fontsize=14)
    plt.xlabel("迭代次数", size=12)
    plt.ylabel("适应度值", size=12)
    plt.plot(original_history, color='b', linewidth=2)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title("精英PSO优化过程", fontsize=14)
    plt.xlabel("迭代次数", size=12)
    plt.ylabel("适应度值", size=12)
    plt.plot(elite_history, color='r', linewidth=2)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 参数设置
    dim = 3
    n_runs = 30  # 独立运行次数，同时也是高层粒子数
    particles = 30 # 仅用于底层粒子数
    iterations = 50
    layers = 3
    
    # 定义每个维度的不同边界, 如果边界一致可使用标量
    lower_bounds = [-5, -5.12, -10]  # 每个维度的下界
    upper_bounds = [5, 5.12, 10]     # 每个维度的上界
    
    # 运行多层精英PSO
    elite_history, top_layer = multiLayerPSO(
        dim=dim, 
        n_runs=n_runs, 
        particles=particles, 
        iterations=iterations, 
        layers=layers,
        objective_func=fitness2,
        lower_bound=lower_bounds,
        upper_bound=upper_bounds
    )
    
    # 运行标准PSO对比
    print("运行标准PSO作为基准对比...")
    standard_pso = PSO(
        pN=particles, 
        dim=dim, 
        max_iter=iterations, 
        lower_bound=lower_bounds,
        upper_bound=upper_bounds, 
        v_max=2, 
        objective_func=fitness2, 
        verbose=False
    )
    standard_history = standard_pso.run()
    
    # 结果对比
    print("\n优化结果对比:")
    print(f"单次标准PSO最优适应度: {standard_pso.fit:.6f}")
    print(f"{layers}层精英PSO最优适应度: {top_layer.fit:.6f}")
    improvement = (standard_pso.fit - top_layer.fit) / standard_pso.fit * 100
    print(f"优化提升: {improvement:.2f}%")
    print("="*50)

    plot_comparison(standard_history, elite_history)