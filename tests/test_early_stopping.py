import numpy as np


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def test_triggers_after_patience_without_improvement():
    """patience 步无改善后应触发早停"""
    from intopt.early_stopping import EarlyStopping

    es = EarlyStopping(key="best", patience=3, min_delta=0.01)
    history = {"best": [10.0]}

    triggered = [es.step(history) for _ in range(10)]
    expected = [False, False, False, True, True, True, True, True, True, True]
    assert triggered == expected


def test_resets_only_on_significant_improvement():
    """显著改善重置计数器，小于 min_delta 的改善不重置"""
    from intopt.early_stopping import EarlyStopping

    es = EarlyStopping(key="best", patience=3, min_delta=1.0)

    history = {"best": [10.0]}
    assert not es.step(history)

    history["best"] = [9.9]  # 改善 0.1 < min_delta，不重置
    assert not es.step(history)

    history["best"] = [8.0]  # 改善 2.0 > min_delta，重置计数器
    assert not es.step(history)

    assert [es.step(history) for _ in range(3)] == [False, False, True]

    assert es.step(history)  # 触发后仍返回 True


def test_sa_early_stopping_terminates_early():
    """SA 集成：早停应在收敛后提前终止，而非跑完完整退火"""
    from intopt.algorithms import SA
    from intopt.early_stopping import EarlyStopping
    from intopt.problems import ContinuousProblem

    problem = ContinuousProblem(func=_sphere, bounds=[(-5.0, 5.0)] * 2)
    es = EarlyStopping(key="best", patience=50, min_delta=1e-6)
    sa = SA(
        problem,
        initial_temp=100,
        final_temp=1e-3,
        cooling_rate=0.95,
        iter_per_temp=100,
        verbose=False,
    )
    result = sa.run(early_stopping=es)

    assert result.best_fitness < 0.1
    max_possible = 1 + int(np.floor(np.log(sa.Tf / sa.T0) / np.log(sa.alpha)))
    assert len(result.history["best"]) < max_possible
