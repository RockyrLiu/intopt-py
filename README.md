# 智能优化算法

当前仓库实现了一些常见的智能优化算法，采用统一的包接口设计。

## 目录

- [快速开始](#快速开始)
- [包结构](#包结构)
- [算法](#算法)
  - [模拟退火算法 (SA)](#模拟退火算法-simulated-annealing)
  - [遗传算法 (GA)](#遗传算法genetic-algorithm)
  - [遗传模拟退火算法 (GASA)](#遗传模拟退火算法-gasa)
  - [粒子群优化算法 (PSO)](#粒子群优化算法particle-swarm-optimization)
  - [免疫算法 (IA)](#免疫算法immune-algorithm)
  - [差分进化算法 (DE)](#差分进化算法differential-evolution)
  - [禁忌搜索算法 (TS)](#禁忌搜索算法tabu-search)
  - [蚁群算法 (ACO)](#蚁群算法ant-colony-optimization)
- [算子模块](#算子模块)
- [问题模块](#问题模块)
- [可视化](#可视化)
- [改进方向](#改进方向)
- [archive 目录](#archive-目录)

---

## 快速开始

```bash
# 安装依赖
uv sync

# 运行示例
uv run python examples/demo_sa.py
uv run python examples/demo_ga.py
uv run python examples/demo_de.py

# 运行测试
uv run pytest tests/ -v
```

### 基本用法

所有算法遵循统一接口：创建问题 → 创建算法子类并覆写算子 → 调用 `run()` → 获取 `OptimizeResult`。

```python
from intopt.algorithms import DE
from intopt.problems import ContinuousProblem
from intopt.operators import initRandom, mutDERand1, cxBinomial

# 1. 定义问题
problem = ContinuousProblem(func=lambda x: sum(x**2), bounds=[(-5, 5)] * 3)

# 2. 覆写算子
class MyDE(DE):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)
    def mutate(self, solution):
        return mutDERand1(solution, self.population, self.fitness, self.F)
    def crossover(self, target, donor):
        return cxBinomial(target, donor, self.CR)

# 3. 运行
result = MyDE(problem, pop_size=50, maxiter=100).run()
print(result.best_fitness)  # 最优适应度（越小越好）
```

## 包结构

```
intopt/
  algorithms/         # 算法实现
    base.py           # Optimizer 基类 + OptimizeResult
    sa.py             # 模拟退火
    ga.py             # 遗传算法
    gasa.py           # 遗传模拟退火（继承 GA）
    pso.py            # 粒子群优化
    ia.py             # 免疫算法
    de.py             # 差分进化
    ts.py             # 禁忌搜索
    aco.py            # 蚁群算法（TSP 默认，可覆写适配连续问题）
    utils.py          # 共享工具（metropolis 等）
  operators/          # 算子库（可组合复用）
    mutate.py         # 变异算子
    crossover.py      # 交叉算子
    selection.py      # 选择算子
    initialize.py     # 初始化算子
    velocity.py       # 速度更新算子
    utils.py          # 算子工具（混沌序列、适应度共享等）
  problems/           # 问题定义
    base.py           # Problem 基类
    continuous.py     # 连续优化问题
    tsp.py            # TSP 问题
    knapsack.py       # 0-1 背包问题
  visualize/          # 可视化
    plots.py          # 收敛曲线、TSP 路径图
  early_stopping.py   # 早停机制
examples/             # 示例脚本
archive/              # 重构前的旧脚本（独立运行，不从 intopt 导入）
```

## 算法

所有算法均**求最小值**（适应度越低越好）。用户通过继承算法类并覆写关键方法来实现具体算子，`run()` 始终返回 `OptimizeResult`。

### 模拟退火算法, Simulated Annealing

**覆写方法**: `mutate(solution)` — 变异操作。连续型用 `mutGaussian`，排列型用 `mutSwap`。

```python
from intopt.algorithms import SA
from intopt.operators import mutGaussian

class MySA(SA):
    def mutate(self, solution):
        return self.problem.clamp(mutGaussian(solution))

result = MySA(problem, initial_temp=100, cooling_rate=0.99).run()
```

**基本理论**: 司守奎《数学建模算法与应用》367-372; 包子阳《智能优化算法及其MATLAB实例》135-154; 秦喜文《数学建模》147-158; [wiki百科](https://zh.wikipedia.org/wiki/%E6%A8%A1%E6%8B%9F%E9%80%80%E7%81%AB)

### 遗传算法，Genetic Algorithm

**覆写方法**: `init_population()`、`crossover(p1, p2)`、`select(pop, fit, k)`、`mutate(solution)`。

```python
from intopt.algorithms import GA
from intopt.operators import initRandom, cxArithmetic, selTournament, mutGaussian

class MyGA(GA):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)
    def crossover(self, p1, p2):
        return cxArithmetic(p1, p2)
    def select(self, pop, fit, k):
        return selTournament(pop, fit, k)
    def mutate(self, solution):
        return self.problem.clamp(mutGaussian(solution))

result = MyGA(problem, pop_size=50, generations=100).run()
```

**基本理论**: 司守奎《数学建模算法与应用》373-382; 秦喜文《数学建模》159-171; Eyal Wirsansky《Hands-On Genetic Algorithms with Python》3-41

### 遗传模拟退火算法 (GASA)

继承 `GA`，变异步骤增加 Metropolis 接受准则。覆写方法与 GA 相同。

```python
from intopt.algorithms import GASA
from intopt.operators import initRandom, cxArithmetic, selTournament, mutGaussian

class MyGASA(GASA):
    def init_population(self): return initRandom(self.problem, self.pop_size)
    def crossover(self, p1, p2): return cxArithmetic(p1, p2)
    def select(self, pop, fit, k): return selTournament(pop, fit, k)
    def mutate(self, solution): return self.problem.clamp(mutGaussian(solution))

result = MyGASA(problem, pop_size=50, generations=200, T0=100).run()
```

### 粒子群优化算法，Particle Swarm Optimization

**覆写方法**: `init_population()`、`update_velocity()`。可选覆写: `update_position()`、`clamp_position()`、`clamp_velocity()`。

```python
from intopt.algorithms import PSO
from intopt.operators import initRandom, velStd

class MyPSO(PSO):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)
    def update_velocity(self):
        return velStd(self.X, self.V, self.pbest, self.gbest, self.w, self.c1, self.c2)

result = MyPSO(problem, pop_size=30, maxiter=100).run()
```

**基本理论**: 包子阳《智能优化算法及其MATLAB实例》109-134; 秦喜文《数学建模》172-184

### 免疫算法，Immune Algorithm

**覆写方法**: `init_population()`、`mutate(solution)`。可选覆写: `distance(a, b)`。可通过 `self.current_gen` 实现动态变异幅度。

```python
from intopt.algorithms import IA
from intopt.operators import initRandom, mutGaussian

class MyIA(IA):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)
    def mutate(self, solution):
        sigma = 3.0 / (1 + self.current_gen * 0.01)
        return self.problem.clamp(mutGaussian(solution, sigma=sigma))

result = MyIA(problem, pop_size=100, maxiter=500).run()
```

**基本理论**: 包子阳《智能优化算法及其MATLAB实例》57-83

### 差分进化算法，Differential Evolution

**覆写方法**: `init_population()`、`mutate(solution)`（差分变异）、`crossover(target, donor)`（二项式交叉）。支持 6 种变异策略。

```python
from intopt.algorithms import DE
from intopt.operators import initRandom, mutDERand1, cxBinomial

class MyDE(DE):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)
    def mutate(self, solution):
        return mutDERand1(solution, self.population, self.fitness, self.F)
    def crossover(self, target, donor):
        return cxBinomial(target, donor, self.CR)

result = MyDE(problem, pop_size=50, maxiter=100).run()
```

**基本理论**: 包子阳《智能优化算法及其MATLAB实例》35-56

### 禁忌搜索算法，Tabu Search

**覆写方法**: `init_solution()`、`generate_candidates(solution)`、`update_tabu(move)`、`is_tabu(move)`。可选覆写 `aspiration()`。

```python
from intopt.algorithms import TS

class MyTS(TS):
    def init_solution(self):
        return self.problem.random_solution()
    def generate_candidates(self, solution):
        # 返回 list[dict]，每项含 "solution" 和 "move"
        ...
    def update_tabu(self, move): ...
    def is_tabu(self, move) -> bool: ...

result = MyTS(problem, tabu_length=10, candidate_size=50, max_iter=500).run()
```

**基本理论**: 包子阳《智能优化算法及其MATLAB实例》155-175

### 蚁群算法，Ant Colony Optimization

**TSP 默认（零覆写，开箱即用）**：

```python
from intopt.algorithms import ACO
from intopt.problems import TSPProblem

aco = ACO(tsp_problem, m=50, max_iter=200, strategy="AS")
result = aco.run()
```

**连续问题（覆写 4 个方法）**：

覆写 `init_population()`、`initialize(population)`、`build_solutions(population)`、`update_pheromone(population, fitness)` 即可适配连续优化问题。详见 `examples/demo_aco_continuous.py`。

支持 4 种信息素更新策略：`AS`（基本）、`EAS`（精英）、`MMAS`（最大最小）、`AAS`（自适应）。

**基本理论**: 包子阳《智能优化算法及其MATLAB实例》85-107

## 算子模块

算子采用 DEAP 风格命名：`mut*`（变异）、`cx*`（交叉）、`sel*`（选择）、`init*`（初始化）、`vel*`（速度）。

### 变异算子 (`intopt.operators.mutate`)

| 算子 | 签名 | 用途 |
|---|---|---|
| `mutGaussian` | `(solution, mu, sigma, indpb)` | 高斯变异，连续型 |
| `mutSwap` | `(solution)` | 交换变异，排列型 |
| `mutFlip` | `(solution)` | 翻转变异，二值型 |
| `mutChaosSwap` | `(solution)` | 混沌交换，排列型 |
| `mutDERand1` | `(solution, population, fitness, F)` | DE/rand/1 |
| `mutDEBest1` | `(solution, population, fitness, F)` | DE/best/1 |
| `mutDERand2` | `(solution, population, fitness, F)` | DE/rand/2 |
| `mutDEBest2` | `(solution, population, fitness, F)` | DE/best/2 |
| `mutDECurrentToRand1` | `(solution, population, fitness, F)` | DE/current-to-rand/1 |
| `mutDECurrentToBest1` | `(solution, population, fitness, F)` | DE/current-to-best/1 |

### 交叉算子 (`intopt.operators.crossover`)

| 算子 | 签名 | 用途 |
|---|---|---|
| `cxArithmetic` | `(p1, p2)` | 算术交叉，连续型 |
| `cxSimulatedBinary` | `(p1, p2, eta, lower, upper)` | SBX，连续型 |
| `cxOnePoint` | `(p1, p2)` | 单点交叉，通用向量 |
| `cxTwoPoint` | `(p1, p2)` | 两点交叉，通用向量 |
| `cxUniform` | `(p1, p2, indpb)` | 均匀交叉，离散/二值 |
| `cxOrdered` | `(p1, p2)` | 顺序交叉，排列型 |
| `cxPartialyMatched` | `(p1, p2)` | 部分匹配交叉，排列型 |
| `cxBinomial` | `(target, donor, CR)` | 二项式交叉，DE 用 |
| `cxChaosArithmetic` | `(p1, p2)` | 混沌算术交叉 |
| `cxChaosOrdered` | `(p1, p2)` | 混沌顺序交叉 |

### 选择算子 (`intopt.operators.selection`)

| 算子 | 签名 |
|---|---|
| `selTournament` | `(population, fitness, k, tournsize)` |
| `selRoulette` | `(population, fitness, k)` |
| `selBest` | `(population, fitness, k)` |

### 初始化算子 (`intopt.operators.initialize`)

| 算子 | 签名 |
|---|---|
| `initRandom` | `(problem, pop_size)` |
| `initChaosContinuous` | `(problem, pop_size)` |
| `initChaosPermutation` | `(problem, pop_size)` |
| `initCustom` | `(population)` |

### 速度算子 (`intopt.operators.velocity`)

| 算子 | 签名 |
|---|---|
| `velStd` | `(X, V, pbest, gbest, w, c1, c2)` |

### 工具函数 (`intopt.operators.utils`)

| 函数 | 用途 |
|---|---|
| `chaos_sequence(length)` | Logistic 混沌序列 |
| `fitnessSharing(pop, fit, threshold, extent)` | 适应度共享（小生境） |

## 问题模块

| 类 | 描述 |
|---|---|
| `ContinuousProblem(func, bounds)` | 连续优化问题 |
| `TSPProblem(coordinates)` | TSP 问题（自动计算距离矩阵） |
| `KnapsackProblem(capacity, weights, values)` | 0-1 背包问题 |

## 可视化

`intopt.visualize` 提供 `plot_convergence()` 和 `plot_tsp_path()`，详见 `examples/demo_sa.py`。

## 改进方向

### 模拟退火
1. 选择更合理的邻域确定方式、更高效的降温策略、更好的初始状态
2. 增加回火（重升温）环节、记忆（精英保留）环节
3. 结合其他搜索算法（遗传、混沌搜索等）

### 遗传算法
1. 更合适的编码、初始化、选择交叉变异算子、终止准则
2. 精英主义算法、小生境与共享、自适应参数
3. k 种群遗传算法、高层遗传算法、混合遗传算法

### 差分进化
1. 改变变异策略（rand → best 等）、差向量个数
2. 使用指数交叉等替代交叉操作

### 免疫算法
1. 更好的浓度计算方式、激励度计算方式、变异算子
2. 与其他算法结合

### 禁忌搜索
1. 区域禁忌 (Region Tabu) 替代单点禁忌
2. 动态调整禁忌区域大小

### 蚁群算法
1. 精英蚂蚁系统、最大最小蚂蚁系统、基于排序的蚁群算法
2. 自适应蚁群算法、与其他算法融合

## archive 目录

`archive/` 存放重构前的原始脚本，使用独立 API（传递 `func`/`bounds`，内嵌绘图）。
这些脚本**独立运行**，不依赖 `intopt` 包，保留以作参考。
