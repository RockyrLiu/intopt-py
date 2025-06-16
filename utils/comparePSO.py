import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from PSO1 import PSO, fitness1, fitness2
from CPSO import CPSO  

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PSO_linear(PSO):
    def __init__(self, pN, dim, max_iter, lower_bound=-5.12, upper_bound=5.12, v_max=2, objective_func=None, initial_positions=None, verbose=True):
        super().__init__(pN, dim, max_iter, lower_bound, upper_bound, v_max, objective_func, initial_positions, verbose)

    def iterator(self):
        fitness_history = []
        
        for t in range(self.max_iter):
            # 动态衰减惯性权重（线性递减）
            self.w = self.w_min + (self.w - self.w_min) * (self.max_iter - t) / self.max_iter
            
            # 生成随机因子矩阵
            r1 = np.random.rand(self.pN, self.dim)
            r2 = np.random.rand(self.pN, self.dim)
            
            # 计算全局最优矩阵
            gbest_matrix = np.repeat(self.gbest, self.pN, axis=0)
            
            # 更新速度
            self.V = (self.w * self.V + 
                     self.c1 * r1 * (self.pbest - self.X) + 
                     self.c2 * r2 * (gbest_matrix - self.X))
            self.V = self.clamp_velocity(self.V)
            
            # 更新位置
            self.X = self.X + self.V
            self.X = self.clamp_position(self.X)
            
            # 计算新位置的适应度
            current_fit = self.function(self.X)
            
            # 找出需要更新的个体
            update_mask = current_fit < self.p_fit
            self.pbest[update_mask] = self.X[update_mask]
            self.p_fit[update_mask] = current_fit[update_mask]
            
            # 更新全局最优
            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = current_fit[min_idx]
                self.gbest = self.X[min_idx].reshape(1, -1)
            
            # 记录每代最优适应度
            fitness_history.append(self.fit)
            
            # 打印进度
            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.fit:.6f}")

        return fitness_history

class PSO_random(PSO):
    def __init__(self, pN, dim, max_iter, lower_bound=-5.12, upper_bound=5.12, v_max=2, objective_func=None, initial_positions=None, verbose=True):
        super().__init__(pN, dim, max_iter, lower_bound, upper_bound, v_max, objective_func, initial_positions, verbose)

    def iterator(self):
        fitness_history = []
        
        for t in range(self.max_iter):
            # 随机惯性权重
            self.w = np.random.uniform(self.w_min, self.w)
            
            # 生成随机因子矩阵
            r1 = np.random.rand(self.pN, self.dim)
            r2 = np.random.rand(self.pN, self.dim)
            
            # 计算全局最优矩阵
            gbest_matrix = np.repeat(self.gbest, self.pN, axis=0)
            
            # 更新速度
            self.V = (self.w * self.V + 
                     self.c1 * r1 * (self.pbest - self.X) + 
                     self.c2 * r2 * (gbest_matrix - self.X))
            self.V = self.clamp_velocity(self.V)
            
            # 更新位置
            self.X = self.X + self.V
            self.X = self.clamp_position(self.X)
            
            # 计算新位置的适应度
            current_fit = self.function(self.X)
            
            # 找出需要更新的个体
            update_mask = current_fit < self.p_fit
            self.pbest[update_mask] = self.X[update_mask]
            self.p_fit[update_mask] = current_fit[update_mask]
            
            # 更新全局最优
            min_idx = np.argmin(current_fit)
            if current_fit[min_idx] < self.fit:
                self.fit = current_fit[min_idx]
                self.gbest = self.X[min_idx].reshape(1, -1)
            
            # 记录每代最优适应度
            fitness_history.append(self.fit)
            
            # 打印进度
            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.fit:.6f}")

        return fitness_history

class PSOComparator:
    def __init__(self, runs=30, pN=30, dim=2, max_iter=100, 
                 lower_bound=-5.12, upper_bound=5.12, v_max=2, 
                 objective_func=fitness1, chaos_reset_ratio=0.3):  # 添加混沌重置比例参数
        """
        参数说明:
        runs: 每种策略的运行次数
        pN: 粒子数量
        dim: 问题维度
        max_iter: 最大迭代次数
        objective_func: 目标函数 (默认fitness1)
        chaos_reset_ratio: CPSO的混沌重置比例 (默认0.3)
        """
        self.runs = runs
        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.v_max = v_max
        self.objective_func = objective_func
        self.chaos_reset_ratio = chaos_reset_ratio  # 保存混沌重置比例
        
        # 存储结果的容器 - 添加CPSO相关容器
        self.linear_results = []
        self.random_results = []
        self.chaos_results = []  # 添加CPSO结果容器
        self.linear_final_fitness = []
        self.random_final_fitness = []
        self.chaos_final_fitness = []  # 添加CPSO最终适应度容器
        self.linear_convergence = []
        self.random_convergence = []
        self.chaos_convergence = []  # 添加CPSO收敛曲线容器
        self.linear_std = []
        self.random_std = []
        self.chaos_std = []  # 添加CPSO标准差容器

    def run_comparison(self):
        """执行比较实验并收集结果"""
        for run in range(self.runs):
            print(f"运行测试 {run+1}/{self.runs}")
            
            # 使用相同的初始位置和速度进行公平比较
            initial_pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.pN, self.dim))
            
            # 测试线性递减策略
            pso_linear = PSO_linear(
                pN=self.pN, 
                dim=self.dim, 
                max_iter=self.max_iter,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                v_max=self.v_max,
                objective_func=self.objective_func,
                initial_positions=initial_pos,
                verbose=False
            )
            linear_history = pso_linear.iterator()
            self.linear_results.append(linear_history)
            self.linear_final_fitness.append(linear_history[-1])
            
            # 测试随机权重策略
            pso_random = PSO_random(
                pN=self.pN, 
                dim=self.dim, 
                max_iter=self.max_iter,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                v_max=self.v_max,
                objective_func=self.objective_func,
                initial_positions=initial_pos,
                verbose=False
            )
            random_history = pso_random.iterator()
            self.random_results.append(random_history)
            self.random_final_fitness.append(random_history[-1])
            
            # 测试混沌粒子群策略 (CPSO) - 添加这部分
            cpso = CPSO(
                pN=self.pN, 
                dim=self.dim, 
                max_iter=self.max_iter,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                v_max=self.v_max,
                objective_func=self.objective_func,
                initial_positions=initial_pos,
                verbose=False,
                chaos_reset_ratio=self.chaos_reset_ratio  # 使用初始化参数
            )
            chaos_history = cpso.iterator()
            self.chaos_results.append(chaos_history)
            self.chaos_final_fitness.append(chaos_history[-1])
        
        # 计算平均收敛曲线 - 添加CPSO部分
        self.linear_convergence = np.mean(self.linear_results, axis=0)
        self.random_convergence = np.mean(self.random_results, axis=0)
        self.chaos_convergence = np.mean(self.chaos_results, axis=0)  # CPSO收敛曲线
        
        # 计算收敛曲线的标准差 - 添加CPSO部分
        self.linear_std = np.std(self.linear_results, axis=0)
        self.random_std = np.std(self.random_results, axis=0)
        self.chaos_std = np.std(self.chaos_results, axis=0)  # CPSO标准差
    
    def plot_convergence_comparison(self):
        """绘制三种策略的收敛曲线对比图"""
        plt.figure(figsize=(12, 8))
        
        # 创建迭代次数数组
        iterations = np.arange(1, self.max_iter + 1)
        
        # 绘制平均收敛曲线 - 添加CPSO曲线
        plt.plot(iterations, self.linear_convergence, 'b-', linewidth=2.5, label='线性递减权重')
        plt.plot(iterations, self.random_convergence, 'r-', linewidth=2.5, label='随机权重')
        plt.plot(iterations, self.chaos_convergence, 'g-', linewidth=2.5, label='混沌粒子群')  # 添加CPSO
        
        # 添加标准差区域 - 添加CPSO区域
        plt.fill_between(iterations, 
                         self.linear_convergence - self.linear_std, 
                         self.linear_convergence + self.linear_std,
                         color='blue', alpha=0.15)
        plt.fill_between(iterations, 
                         self.random_convergence - self.random_std, 
                         self.random_convergence + self.random_std,
                         color='red', alpha=0.15)
        plt.fill_between(iterations, 
                         self.chaos_convergence - self.chaos_std, 
                         self.chaos_convergence + self.chaos_std,
                         color='green', alpha=0.15)  # 添加CPSO
        
        plt.title('PSO策略收敛曲线对比', fontsize=16)
        plt.xlabel('迭代次数', fontsize=14)
        plt.ylabel('适应度值', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_final_fitness_distribution(self):
        """绘制最终适应度的分布对比 - 添加CPSO"""
        plt.figure(figsize=(10, 6))
        
        # 箱线图展示分布 - 添加CPSO
        box = plt.boxplot([self.linear_final_fitness, 
                          self.random_final_fitness,
                          self.chaos_final_fitness],  # 添加CPSO数据
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='darkblue'),
                        medianprops=dict(color='red'))
        
        # 设置不同箱子的颜色
        colors = ['lightblue', 'lightgreen', 'salmon']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # 设置标签 - 添加CPSO标签
        plt.xticks([1, 2, 3], ['线性递减权重', '随机权重', '混沌粒子群'])
        
        plt.title('最终适应度分布对比', fontsize=16)
        plt.ylabel('适应度值', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def statistical_analysis(self):
        """执行统计分析并打印结果 - 添加CPSO分析"""
        # 计算基本统计量 - 添加CPSO
        linear_mean = np.mean(self.linear_final_fitness)
        random_mean = np.mean(self.random_final_fitness)
        chaos_mean = np.mean(self.chaos_final_fitness)
        
        linear_std = np.std(self.linear_final_fitness)
        random_std = np.std(self.random_final_fitness)
        chaos_std = np.std(self.chaos_final_fitness)
        
        # 执行t检验 - 添加CPSO与其他策略的比较
        t_stat_lin_rand, p_value_lin_rand = stats.ttest_ind(
            self.linear_final_fitness, 
            self.random_final_fitness,
            equal_var=False
        )
        
        t_stat_lin_chaos, p_value_lin_chaos = stats.ttest_ind(
            self.linear_final_fitness, 
            self.chaos_final_fitness,
            equal_var=False
        )
        
        t_stat_rand_chaos, p_value_rand_chaos = stats.ttest_ind(
            self.random_final_fitness, 
            self.chaos_final_fitness,
            equal_var=False
        )
        
        # 打印结果 - 添加CPSO结果
        print("\n" + "="*60)
        print("PSO策略性能统计分析")
        print("="*60)
        print(f"测试配置: {self.runs}次运行, {self.pN}个粒子, {self.dim}维问题")
        print(f"目标函数: {self.objective_func.__name__}")
        print(f"CPSO混沌重置比例: {self.chaos_reset_ratio}")
        print("\n--- 线性递减权重策略 ---")
        print(f"平均最终适应度: {linear_mean:.6f} ± {linear_std:.6f}")
        print(f"最佳适应度: {np.min(self.linear_final_fitness):.6f}")
        print(f"最差适应度: {np.max(self.linear_final_fitness):.6f}")
        
        print("\n--- 随机权重策略 ---")
        print(f"平均最终适应度: {random_mean:.6f} ± {random_std:.6f}")
        print(f"最佳适应度: {np.min(self.random_final_fitness):.6f}")
        print(f"最差适应度: {np.max(self.random_final_fitness):.6f}")
        
        print("\n--- 混沌粒子群策略 ---")  # 添加CPSO部分
        print(f"平均最终适应度: {chaos_mean:.6f} ± {chaos_std:.6f}")
        print(f"最佳适应度: {np.min(self.chaos_final_fitness):.6f}")
        print(f"最差适应度: {np.max(self.chaos_final_fitness):.6f}")
        
        print("\n--- 假设检验结果 ---")
        # 线性 vs 随机
        print(f"\n[线性递减 vs 随机权重]")
        print(f"t统计量: {t_stat_lin_rand:.4f}, p值: {p_value_lin_rand:.6f}")
        self._print_comparison_result(p_value_lin_rand, linear_mean, random_mean, "线性递减", "随机权重")
        
        # 线性 vs 混沌
        print(f"\n[线性递减 vs 混沌粒子群]")
        print(f"t统计量: {t_stat_lin_chaos:.4f}, p值: {p_value_lin_chaos:.6f}")
        self._print_comparison_result(p_value_lin_chaos, linear_mean, chaos_mean, "线性递减", "混沌粒子群")
        
        # 随机 vs 混沌
        print(f"\n[随机权重 vs 混沌粒子群]")
        print(f"t统计量: {t_stat_rand_chaos:.4f}, p值: {p_value_rand_chaos:.6f}")
        self._print_comparison_result(p_value_rand_chaos, random_mean, chaos_mean, "随机权重", "混沌粒子群")
        
        print("="*60)
    
    def _print_comparison_result(self, p_value, mean1, mean2, name1, name2):
        """辅助函数：打印比较结果"""
        if p_value < 0.05:
            print(f"结论: {name1}和{name2}在0.05显著性水平下存在统计显著差异")
            if mean1 < mean2:
                print(f"      {name1}策略表现更好")
            else:
                print(f"      {name2}策略表现更好")
        else:
            print(f"结论: {name1}和{name2}在0.05显著性水平下无显著差异")

# 添加主函数用于执行比较
if __name__ == "__main__":
    # 配置比较实验参数
    comparator = PSOComparator(
        runs=1000,            # 1000次独立运行
        pN=30,              # 30个粒子
        dim=5,              # 5维问题
        max_iter=100,       # 100次迭代
        lower_bound=-5.12,  # 搜索空间下界
        upper_bound=5.12,   # 搜索空间上界
        v_max=2,            # 最大速度限制
        objective_func=fitness2,  # 使用第二个测试函数
        chaos_reset_ratio=0.2    # CPSO的混沌重置比例
    )
    
    # 执行比较实验
    print("开始执行PSO策略比较实验...")
    comparator.run_comparison()
    
    # 结果分析和可视化
    comparator.plot_convergence_comparison()
    comparator.plot_final_fitness_distribution()
    comparator.statistical_analysis()