import numpy as np
from tqdm import tqdm

from intopt.algorithms.base import OptimizeResult
from intopt.algorithms.ga import GA
from intopt.algorithms.utils import metropolis


class GASA(GA):
    """遗传模拟退火算法。

    在 GA 基础上，变异步骤使用模拟退火的 Metropolis 接受准则：
    更优的突变总是接受，更差的突变以概率 exp(-ΔE/T) 接受，
    且该概率随代数递减。

    其他所有环节（初始化、选择、交叉、精英维护）与 GA 完全相同。

    Parameters
    ----------
    problem:
        待求解的优化问题实例。
    T0:
        初始温度，默认 100。
    Tf:
        终止温度，默认 1e-3。
    alpha:
        降温系数 (0, 1)，默认 0.99。
    **ga_kwargs:
        传递给 GA 的参数（pop_size、generations、cxpb、mutpb、
        elitism_size、verbose 等）。
    """

    def __init__(
        self,
        problem,
        T0: float = 100,
        Tf: float = 1e-3,
        alpha: float = 0.99,
        **ga_kwargs,
    ):
        super().__init__(problem, **ga_kwargs)
        self.T0 = float(T0)
        self.Tf = float(Tf)
        self.alpha = float(alpha)

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
        self._validate_overrides()

        population = self.init_population()
        fitness = np.array([self.problem.evaluate(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        history_best = [float(best_fitness)]
        history_avg = [float(np.mean(fitness))]

        self.elites = []
        self.update_elites(population, fitness)

        total_gen = int(
            np.floor(np.log(self.Tf / self.T0) / np.log(self.alpha))
        )
        pbar = tqdm(
            total=min(self.generations, total_gen),
            desc="GA-SA 进化",
            disable=not self.verbose,
        )

        T = self.T0
        for _gen in range(self.generations):
            n_selected = self.pop_size - self.elitism_size
            selected = (
                self.select(population, fitness, n_selected)
                if n_selected > 0
                else np.array([])
            )
            np.random.shuffle(selected)

            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 >= len(selected):
                    offspring.append(self.problem.clamp(selected[i].copy()))
                    break
                p1, p2 = selected[i], selected[i + 1]
                if np.random.rand() < self.cxpb:
                    c1, c2 = self.crossover(p1, p2)
                    c1 = self.problem.clamp(c1)
                    c2 = self.problem.clamp(c2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                offspring.append(c1)
                offspring.append(c2)

            for i in range(len(offspring)):
                if np.random.rand() < self.mutpb:
                    old_fit = self.problem.evaluate(offspring[i])
                    mutant = self.mutate(offspring[i])
                    new_fit = self.problem.evaluate(mutant)
                    if metropolis(new_fit, old_fit, T):
                        offspring[i] = mutant

            if self.elitism_size > 0:
                elite_pop = np.array([ind for ind, _ in self.elites])
                population = np.vstack([elite_pop, offspring])[: self.pop_size]
            else:
                population = np.array(offspring[: self.pop_size])

            fitness = np.array(
                [self.problem.evaluate(ind) for ind in population]
            )

            self.update_elites(population, fitness)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

            history_best.append(float(best_fitness))
            history_avg.append(float(np.mean(fitness)))

            if self.verbose:
                pbar.set_postfix(
                    {
                        "温度": f"{T:.2e}",
                        "最优值": f"{best_fitness:.4f}",
                    }
                )
                pbar.update(1)

            T = max(T * self.alpha, self.Tf)

            history_dict = {"best": history_best, "avg": history_avg}
            if self._check_early_stop(early_stopping, history_dict):
                break

        if self.verbose:
            pbar.close()

        self.population = population

        return OptimizeResult(
            best_solution=best_solution,
            best_fitness=float(best_fitness),
            history={"best": history_best, "avg": history_avg},
        )
