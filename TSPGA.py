import numpy as np
from GA1 import BaseGA, plot_convergence
from test_function import Rastrigin, TSPProblem1, TSPProblem2

class TSPGA(BaseGA):
    """TSP问题的遗传算法"""
    def __init__(self, tsp_problem, **kwargs):
        self.tsp_problem = tsp_problem
        self.num_cities = tsp_problem.N
        super().__init__(self._tsp_objective, self.num_cities, **kwargs)
    
    def _tsp_objective(self, path):
        path = self.check_bounds(path)
        return self.tsp_problem.calculate_path_length(path)
    
    def init_population(self):
        """初始化种群"""
        pop = []
        for _ in range(self.pop_size):
            individual = np.random.permutation(self.num_cities)
            pop.append(individual)
        return np.array(pop)
    
    def check_bounds(self, path):
        """确保路径是有效的排列"""
        try:
            path = path.astype(int)
        except:
            return np.random.permutation(self.num_cities)
        
        # 检查路径长度是否正确
        if len(path) != self.num_cities:
            return np.random.permutation(self.num_cities)
        
        # 检查是否包含所有城市且不重复
        if len(np.unique(path)) != self.num_cities:
            return np.random.permutation(self.num_cities)
        
        # 确保路径中的城市索引是连续的0到N-1
        if not np.array_equal(np.sort(path), np.arange(self.num_cities)):
            return np.random.permutation(self.num_cities)
        
        return path
    
    def crossover(self, parent1, parent2):
        """顺序交叉(OX)"""
        parent1 = parent1.astype(int)
        parent2 = parent2.astype(int)
        size = len(parent1)
        cx1, cx2 = sorted(np.random.choice(size, 2, replace=False))
        
        child1 = np.full(size, -1, dtype=int)
        child2 = np.full(size, -1, dtype=int)
        
        child1[cx1:cx2+1] = parent1[cx1:cx2+1]
        child2[cx1:cx2+1] = parent2[cx1:cx2+1]
        
        current_index = (cx2 + 1) % size
        for p in [parent2, parent1]:
            for i in range(size):
                index = (i + cx2 + 1) % size
                city = p[index]
                if city not in child1:
                    child1[current_index] = city
                    current_index = (current_index + 1) % size
        
        current_index = (cx2 + 1) % size
        for p in [parent1, parent2]:
            for i in range(size):
                index = (i + cx2 + 1) % size
                city = p[index]
                if city not in child2:
                    child2[current_index] = city
                    current_index = (current_index + 1) % size
        
        return child1, child2
    
    def mutate(self, individual):
        """交换变异"""
        individual = individual.astype(int)
        idx1, idx2 = np.random.choice(self.num_cities, 2, replace=False)
        mutated = individual.copy()
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
def solve_tsp_problem1():
    """解决TSP问题1"""
    print("\n" + "="*50)
    print("TSP问题1测试")
    print("="*50)
    
    tsp_problem = TSPProblem1()
    
    # 基本TSP GA
    ga_tsp = TSPGA(
        tsp_problem=tsp_problem,
        pop_size=100,
        generations=200,
        verbose=True
    )
    best_path, best_length = ga_tsp.run(elitism=False)
    print(f"基本TSP GA - 最短路径长度: {best_length:.2f}")
    tsp_problem.plot_solution(best_path, best_length)
    plot_convergence(ga_tsp.history, '基本TSP GA收敛曲线')
    
    # 精英TSP GA
    ga_tsp_elite = TSPGA(
        tsp_problem=tsp_problem,
        pop_size=100,
        generations=200,
        verbose=True
    )
    best_path, best_length = ga_tsp_elite.run(elitism=True)
    print(f"精英TSP GA - 最短路径长度: {best_length:.2f}")
    tsp_problem.plot_solution(best_path, best_length)
    plot_convergence(ga_tsp_elite.history, '精英TSP GA收敛曲线')


def solve_tsp_problem2():
    """解决TSP问题2"""
    print("\n" + "="*50)
    print("TSP问题2测试")
    print("="*50)
    
    tsp_problem = TSPProblem2()
    
    # 基本TSP GA
    ga_tsp = TSPGA(
        tsp_problem=tsp_problem,
        pop_size=200,
        generations=500,
        verbose=True
    )
    best_path, best_length = ga_tsp.run(elitism=False)
    print(f"基本TSP GA - 最短路径长度: {best_length:.2f}")
    tsp_problem.plot_solution(best_path, best_length)
    plot_convergence(ga_tsp.history, '基本TSP GA收敛曲线')
    
    # 精英TSP GA
    ga_tsp_elite = TSPGA(
        tsp_problem=tsp_problem,
        pop_size=200,
        generations=500,
        verbose=True
    )
    best_path, best_length = ga_tsp_elite.run(elitism=True)
    print(f"精英TSP GA - 最短路径长度: {best_length:.2f}")
    tsp_problem.plot_solution(best_path, best_length)
    plot_convergence(ga_tsp_elite.history, '精英TSP GA收敛曲线')

if __name__ == "__main__":
    solve_tsp_problem1()
    solve_tsp_problem2()