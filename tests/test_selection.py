import numpy as np


def test_selection_operators():
    """各选择算子返回合法个体"""
    from intopt.operators.selection import selBest, selRoulette, selTournament

    pop = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    fit = np.array([4.0, 3.0, 2.0, 1.0, 0.0])

    for sel in [selTournament, selRoulette, selBest]:
        selected = sel(pop, fit, k=3)
        assert selected.shape == (3, 1)

    selected = selTournament(pop, fit, k=8)
    assert selected.shape == (8, 1)
