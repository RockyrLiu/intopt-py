import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import Optimizer, OptimizeResult


class ACO(Optimizer):
    """蚁群算法 (Ant Colony Optimization)。

    核心流程：初始化 → 构建解 → 评估 → 更新信息素 → 迭代。

    **可重写的方法**：

    - ``init_population()`` — 初始化种群，返回 ``(m, *)``。
      TSP 默认生成随机排列路径。
    - ``initialize(population)`` — 初始化算法状态（信息素等）。
       TSP 默认设置信息素矩阵和启发式信息。
    - ``build_solutions(population)`` — 构建解，返回新种群。
       TSP 默认按概率选择城市。
    - ``update_pheromone(population, fitness)`` — 根据解质量更新信息素。
       支持 AS/EAS/MMAS/AAS 策略。

    针对非 TSP 问题，只需覆写 ``initialize``、
    ``build_solutions``、``update_pheromone`` 三个方法即可。

    Parameters
    ----------
    problem:
        优化问题实例（TSP 默认使用 TSPProblem）。
    m:
        蚂蚁数量，默认 50。
    max_iter:
        最大迭代次数，默认 200。
    alpha:
        信息素重要程度参数，默认 1。
    beta:
        启发式因子重要程度参数，默认 5。
    rho:
        信息素蒸发系数，默认 0.1。
    Q:
        信息素增加强度系数，默认 100。
    strategy:
        信息素更新策略，可选 'AS'/'EAS'/'MMAS'/'AAS'，默认 'AS'。
    verbose:
        是否显示进度条，默认 True。
    """

    def __init__(
        self,
        problem,
        m=50,
        max_iter=200,
        alpha=1,
        beta=5,
        rho=0.1,
        Q=100,
        strategy="AS",
        verbose=True,
    ):
        super().__init__(problem)
        self.m = int(m)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.Q = float(Q)
        self.strategy = strategy
        self.verbose = verbose
        self.current_gen: int = 0

    # ------------------------------------------------------------------
    # 可重写的方法
    # ------------------------------------------------------------------

    def init_population(self) -> np.ndarray:
        """初始化种群。**可重写**。

        TSP 默认：生成 ``m`` 个城市随机排列。

        Returns
        -------
        np.ndarray
            种群，shape ``(m, n_cities)``。
        """
        n = self.problem.n
        return np.array([np.random.permutation(n) for _ in range(self.m)])

    def initialize(self, population: np.ndarray):
        """初始化算法状态。**可重写**。

        TSP 默认：设置信息素矩阵 Tau (n x n) 及
        启发式信息 Eta (1 / 距离)。

        Parameters
        ----------
        population:
            初始种群。
        """
        n = self.problem.n
        if self.strategy == "MMAS":
            self.Tau = np.ones((n, n)) * 2.0
        else:
            self.Tau = np.ones((n, n))
        self.Eta = 1 / (self.problem.dist_matrix + np.eye(n) * 1e-10)

    def build_solutions(
        self, population: np.ndarray
    ) -> np.ndarray:
        """构建解。**可重写**。

        TSP 默认：每只蚂蚁从随机起点出发，按概率选择下一个城市，
        概率 = Tau^alpha * Eta^beta。

        Parameters
        ----------
        population:
            当前种群（上一代的 Tabu 表，shape ``(m, n_cities)``）。

        Returns
        -------
        np.ndarray
            新种群，shape ``(m, n_cities)``。
        """
        n = self.problem.n
        tabu = np.zeros((self.m, n), dtype=int)
        num_perms = int(np.ceil(self.m / n))
        perms = np.concatenate(
            [np.random.permutation(n) for _ in range(num_perms)]
        )
        tabu[:, 0] = perms[: self.m]

        for j in range(1, n):
            for i in range(self.m):
                visited = tabu[i, :j]
                unvisited = np.setdiff1d(np.arange(n), visited)
                tau = self.Tau[visited[-1], unvisited]
                eta = self.Eta[visited[-1], unvisited]
                P = (tau ** self.alpha) * (eta ** self.beta)
                P /= P.sum()
                tabu[i, j] = np.random.choice(unvisited, p=P)
        return tabu

    def update_pheromone(
        self, population: np.ndarray, fitness: np.ndarray
    ):
        """更新信息素。**可重写**。

        根据 ``self.strategy`` 分发到对应策略方法。
        TSP 默认实现在信息素矩阵 Tau 上按路径边沉积。

        Parameters
        ----------
        population:
            当前种群。
        fitness:
            适应度数组，shape ``(m,)``。
        """
        if self.strategy == "AS":
            self._update_as(population, fitness)
        elif self.strategy == "EAS":
            self._update_eas(population, fitness)
        elif self.strategy == "MMAS":
            self._update_mmas(population, fitness)
        elif self.strategy == "AAS":
            self._update_aas(population, fitness)
        else:
            raise ValueError(f"未知策略: {self.strategy}")

    def extract_best_solution(
        self, population: np.ndarray, best_idx: int
    ) -> np.ndarray:
        """从种群中提取最优解。**可重写**。"""
        return population[best_idx].copy()

    # ------------------------------------------------------------------
    # TSP 信息素更新策略
    # ------------------------------------------------------------------

    def _update_as(
        self, population: np.ndarray, L: np.ndarray
    ):
        n = self.problem.n
        Delta_Tau = np.zeros((n, n))
        for i in range(self.m):
            path = population[i]
            for k in range(n - 1):
                Delta_Tau[path[k], path[k + 1]] += self.Q / L[i]
            Delta_Tau[path[-1], path[0]] += self.Q / L[i]
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau

    def _update_eas(
        self, population: np.ndarray, L: np.ndarray
    ):
        n = self.problem.n
        Delta_Tau = np.zeros((n, n))
        for i in range(self.m):
            path = population[i]
            for k in range(n - 1):
                Delta_Tau[path[k], path[k + 1]] += self.Q / L[i]
            Delta_Tau[path[-1], path[0]] += self.Q / L[i]

        best_idx = np.argmin(L)
        best_path = population[best_idx]
        best_length = L[best_idx]
        for k in range(n - 1):
            Delta_Tau[best_path[k], best_path[k + 1]] += (
                2.0 * self.Q / best_length
            )
        Delta_Tau[best_path[-1], best_path[0]] += (
            2.0 * self.Q / best_length
        )
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau

    def _update_mmas(
        self, population: np.ndarray, L: np.ndarray
    ):
        n = self.problem.n
        Delta_Tau = np.zeros((n, n))
        best_idx = np.argmin(L)
        best_path = population[best_idx]
        best_length = L[best_idx]
        for k in range(n - 1):
            Delta_Tau[best_path[k], best_path[k + 1]] += (
                self.Q / best_length
            )
        Delta_Tau[best_path[-1], best_path[0]] += (
            self.Q / best_length
        )
        self.Tau = (1 - self.rho) * self.Tau + Delta_Tau
        self.Tau = np.clip(self.Tau, 0.001, 2.0)

    def _update_aas(
        self, population: np.ndarray, L: np.ndarray
    ):
        n = self.problem.n
        avg_length = np.mean(L)
        best_length = np.min(L)
        ratio = best_length / (avg_length + 1e-10)
        rho = 0.01 + (0.5 - 0.01) * ratio

        Delta_Tau = np.zeros((n, n))
        for i in range(self.m):
            path = population[i]
            for k in range(n - 1):
                Delta_Tau[path[k], path[k + 1]] += self.Q / L[i]
            Delta_Tau[path[-1], path[0]] += self.Q / L[i]
        self.Tau = (1 - rho) * self.Tau + Delta_Tau

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, early_stopping=None) -> OptimizeResult:
        """执行优化，返回 ``OptimizeResult``。

        Parameters
        ----------
        early_stopping:
            ``EarlyStopping`` 实例，默认 None 表示不启用早停。

        Returns
        -------
        OptimizeResult
            history 包含以下键：
            - ``"best"``  每代全局最优适应度
            - ``"avg"``   每代平均适应度
        """
        population = self.init_population()
        self.initialize(population)
        fitness = np.array(
            [self.problem.evaluate(ind) for ind in population]
        )

        best_idx = np.argmin(fitness)
        best_solution = self.extract_best_solution(population, best_idx)
        best_fitness = float(fitness[best_idx])

        history_best = [best_fitness]
        history_avg = [float(np.mean(fitness))]

        pbar = tqdm(
            total=self.max_iter,
            desc="ACO",
            disable=not self.verbose,
        )

        for gen in range(self.max_iter):
            self.current_gen = gen

            population = self.build_solutions(population)
            fitness = np.array(
                [self.problem.evaluate(ind) for ind in population]
            )

            current_best_idx = np.argmin(fitness)
            current_best = float(fitness[current_best_idx])

            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = self.extract_best_solution(
                    population, current_best_idx
                )

            history_best.append(best_fitness)
            history_avg.append(float(np.mean(fitness)))

            self.update_pheromone(population, fitness)

            if self.verbose:
                pbar.set_postfix(
                    {
                        "最优值": f"{best_fitness:.4f}",
                        "平均值": f"{float(np.mean(fitness)):.4f}",
                    }
                )
                pbar.update(1)

            history_dict = {"best": history_best, "avg": history_avg}
            if self._check_early_stop(early_stopping, history_dict):
                break

        if self.verbose:
            pbar.close()

        return OptimizeResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            history={"best": history_best, "avg": history_avg},
        )
