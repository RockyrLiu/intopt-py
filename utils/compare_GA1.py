import numpy as np
import time
import matplotlib.pyplot as plt
from GA1 import GA, TSP
from GA1_ChaosOptimized import GA_Chaos

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def run_comparison(num_runs=100, pop_size=50, generations=100, mutation_rate=0.1):
    """
    比较原始GA和混沌优化GA的性能
    
    参数:
    num_runs: 每种算法运行的次数
    pop_size: 种群大小
    generations: 进化代数
    mutation_rate: 变异率
    """
    # 初始化问题实例
    tsp = TSP(r'data\obj_longitude_latitude.txt')
    
    # 存储结果
    results = {
        'GA': {'lengths': [], 'times': []},
        'GA_Chaos': {'lengths': [], 'times': []}
    }
    
    print(f"开始比较测试，每种算法将运行{num_runs}次...")
    
    # 测试原始GA
    print("\n测试原始GA算法...")
    for i in range(num_runs):
        start_time = time.time()
        ga = GA(population_size=pop_size, generations=generations, mutation_rate=mutation_rate)
        _, best_length = ga.solve(tsp)
        run_time = time.time() - start_time
        
        results['GA']['lengths'].append(best_length)
        results['GA']['times'].append(run_time)
        
        if (i+1) % 10 == 0:
            print(f"已完成原始GA运行 {i+1}/{num_runs}")
    
    # 测试混沌优化GA
    print("\n测试混沌优化GA算法...")
    for i in range(num_runs):
        start_time = time.time()
        ga_chaos = GA_Chaos(population_size=pop_size, generations=generations, mutation_rate=mutation_rate)
        _, best_length = ga_chaos.solve(tsp)
        run_time = time.time() - start_time
        
        results['GA_Chaos']['lengths'].append(best_length)
        results['GA_Chaos']['times'].append(run_time)
        
        if (i+1) % 10 == 0:
            print(f"已完成混沌优化GA运行 {i+1}/{num_runs}")
    
    # 计算统计量
    stats = {
        'GA': {
            'avg_length': np.mean(results['GA']['lengths']),
            'std_length': np.std(results['GA']['lengths']),
            'avg_time': np.mean(results['GA']['times']),
            'std_time': np.std(results['GA']['times'])
        },
        'GA_Chaos': {
            'avg_length': np.mean(results['GA_Chaos']['lengths']),
            'std_length': np.std(results['GA_Chaos']['lengths']),
            'avg_time': np.mean(results['GA_Chaos']['times']),
            'std_time': np.std(results['GA_Chaos']['times'])
        }
    }
    
    # 打印结果
    print("\n比较结果:")
    print(f"{'算法':<15} {'平均路径长度':<18} {'长度标准差':<18} {'平均时间(s)':<15} {'时间标准差':<15}")
    print("-"*85)
    for algo in ['GA', 'GA_Chaos']:
        print(f"{algo:<15} {stats[algo]['avg_length']:<20.4f} {stats[algo]['std_length']:<20.4f} "
              f"{stats[algo]['avg_time']:<20.4f} {stats[algo]['std_time']:<20.4f}")
    
    # 绘制结果比较图
    plot_comparison(results, stats)
    
    return results, stats

def plot_comparison(results, stats):
    """绘制比较结果图"""
    plt.figure(figsize=(15, 6))
    
    # 路径长度比较
    plt.subplot(1, 2, 1)
    plt.boxplot([results['GA']['lengths'], results['GA_Chaos']['lengths']], 
                tick_labels=['原始GA', '混沌GA'])
    plt.title('路径长度比较')
    plt.ylabel('路径长度')
    
    # 运行时间比较
    plt.subplot(1, 2, 2)
    plt.boxplot([results['GA']['times'], results['GA_Chaos']['times']], 
            tick_labels=['原始GA', '混沌GA'])
    plt.title('运行时间比较')
    plt.ylabel('时间(s)')
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # 运行比较测试 (可以调整num_runs减少运行时间)
    results, stats = run_comparison(num_runs=100)  # 正式测试建议100次，调试时可减少
    
    # 保存结果到文件
    # np.savez('comparison_results.npz', results=results, stats=stats)
    # print("\n结果已保存到 comparison_results.npz")