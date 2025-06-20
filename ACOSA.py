from SA2 import TSP_SA  # 导入模拟退火TSP类
from ACO1 import ACO_TSP  # 导入蚁群算法TSP类
from test_function import TSPProblem1, TSPProblem2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Hybrid_ACO_SA(TSP_SA):
    """蚁群-模拟退火混合算法"""
    def __init__(self, tsp_problem, aco_params=None, sa_params=None):
        """
        参数:
        tsp_problem: TSP问题实例
        aco_params: ACO参数字典
        sa_params: SA参数字典
        """
        self.tsp_problem = tsp_problem
        self.N = tsp_problem.N
        
        # 设置默认参数
        aco_params = aco_params or {
            'm': 50, 'max_iter': 200, 'alpha': 1, 'beta': 5, 'rho': 0.1, 'Q': 100
        }
        sa_params = sa_params or {
            'initial_temp': 1000, 'final_temp': 1e-4, 
            'cooling_rate': 0.999, 'iter': 1000
        }
        
        # 运行ACO获得初始解
        self.aco = ACO_TSP(problem=tsp_problem, **aco_params)
        self.aco.problem = tsp_problem  # 确保问题一致
        self.aco_solution, self.aco_length = self.aco.run()
        
        # 初始化SA
        super().__init__(tsp_problem, **sa_params)
    
    def init_solution(self):
        """使用ACO的解作为初始解"""
        return self.aco_solution.tolist()  # 转换为列表

def run_hybrid_algorithm(problem_class):
    """运行混合算法"""
    print("="*50)
    print(f"蚁群-模拟退火混合算法求解TSP问题 ({problem_class.__name__})")
    print("="*50)
    
    # 初始化问题
    problem = problem_class()
    
    # 创建混合算法实例
    hybrid = Hybrid_ACO_SA(
        problem,
        aco_params={'m': 120, 'max_iter': 100},
        sa_params={'initial_temp': 200, 'cooling_rate': 0.999, 'iter': 500}
    )
    
    # 运行算法
    best_path, best_length = hybrid.run()
    
    # 输出结果
    print(f"\nACO初始解长度: {hybrid.aco_length:.2f}")
    print(f"混合算法最优长度: {best_length:.2f}")
    print(f"优化提升: {(hybrid.aco_length - best_length)/hybrid.aco_length*100:.2f}%")
    print("最优路径:", best_path)
    
    # 可视化
    problem.plot_solution(best_path, best_length)
    
    # 绘制收敛曲线
    plt.figure(figsize=(12, 5))
    
    # ACO收敛曲线
    plt.subplot(121)
    plt.plot(hybrid.aco.history['best_fitness'], 'b-', label='ACO最优值')
    plt.plot(hybrid.aco.history['avg_fitness'], 'r--', label='ACO平均值')
    plt.title('蚁群算法收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.legend()
    plt.grid(True)
    
    # SA收敛曲线
    plt.subplot(122)
    plt.plot(hybrid.history['temperature'], hybrid.history['best_fitness'])
    plt.title('模拟退火收敛曲线')
    plt.xlabel('温度')
    plt.ylabel('路径长度')
    plt.gca().invert_xaxis()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_hybrid_algorithm(TSPProblem2)