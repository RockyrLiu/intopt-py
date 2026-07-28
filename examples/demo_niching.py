import matplotlib.pyplot as plt
import numpy as np

from intopt.algorithms import GA
from intopt.operators.crossover import cxSimulatedBinary
from intopt.operators.initialize import initRandom
from intopt.operators.mutate import mutGaussian
from intopt.operators.selection import selTournament
from intopt.operators.utils import fitnessSharing
from intopt.problems import ContinuousProblem

plt.rcParams["font.sans-serif"] = ["SimHei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False

OPTIMA = np.array(
    [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584458, -1.848126],
    ]
)


def himmelblau(x: np.ndarray) -> float:
    return float((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)


class ContGA(GA):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)

    def crossover(self, p1, p2):
        return cxSimulatedBinary(
            p1, p2, lower=self.problem.lower, upper=self.problem.upper
        )

    def select(self, pop, fit, k):
        return selTournament(pop, fit, k, tournsize=2)

    def mutate(self, solution):
        return self.problem.clamp(mutGaussian(solution, sigma=0.1))


class SharedGA(GA):
    def init_population(self):
        return initRandom(self.problem, self.pop_size)

    def crossover(self, p1, p2):
        return cxSimulatedBinary(
            p1, p2, lower=self.problem.lower, upper=self.problem.upper
        )

    def select(self, pop, fit, k):
        return selTournament(
            pop,
            fitnessSharing(pop, fit, 0.1, 5.0),
            k,
            tournsize=2,
        )

    def mutate(self, solution):
        return self.problem.clamp(mutGaussian(solution, sigma=0.1))

    # 自行决定精英更新是用原始适应度还是共享适应度
    # def update_elites(self, population, fitness):
    #     shared = fitnessSharing(population, fitness, 0.1, 5.0)
    #     candidates = sorted(
    #         zip(population, fitness, shared),
    #         key=lambda x: x[2],
    #     )
    #     self.elites = [
    #         (ind.copy(), float(fit))
    #         for ind, fit, _ in candidates[: max(self.elitism_size, 1)]
    #     ]


def main():
    problem = ContinuousProblem(func=himmelblau, bounds=[(-5.0, 5.0)] * 2)

    ga_std = ContGA(
        problem,
        pop_size=300,
        generations=300,
        elitism_size=30,
        cxpb=0.9,
        mutpb=0.5,
        verbose=True,
    )
    ga_std.run()
    print(f"标准 GA  最优适应度: {ga_std.elites[0][1]:.6f}")

    ga_shared = SharedGA(
        problem,
        pop_size=300,
        generations=300,
        elitism_size=30,
        cxpb=0.9,
        mutpb=0.5,
        verbose=True,
    )
    ga_shared.run()
    print(f"共享 GA  最优适应度: {ga_shared.elites[0][1]:.6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ga, title in [
        (ax1, ga_std, "标准 GA"),
        (ax2, ga_shared, "共享 GA (Niching)"),
    ]:
        ax.scatter(
            ga.population[:, 0],
            ga.population[:, 1],
            marker=".",
            color="gray",
            alpha=0.15,
            label="种群",
        )
        elite_sols = np.array([sol for sol, _ in ga.elites])
        ax.scatter(
            elite_sols[:, 0],
            elite_sols[:, 1],
            marker=".",
            color="blue",
            label=f"精英 ({len(ga.elites)} 个)",
        )
        ax.scatter(*OPTIMA.T, marker="x", color="red", s=100, label="全局最优")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_title(title)

    fig.suptitle("Himmelblau 函数 — 小生境效果对比")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
