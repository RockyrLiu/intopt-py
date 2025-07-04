import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 定义问题（使用ZDT1测试问题）
problem = get_problem("zdt1")

# 配置NSGA-II算法
algorithm = NSGA2(pop_size=100)

# 执行优化
res = minimize(
    problem,
    algorithm,
    ('n_gen', 100),
    seed=1,
    verbose=True
)

# 可视化结果
plot = Scatter(title="Pareto Front")
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()