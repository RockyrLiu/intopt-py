import numpy as np


def test_mut_chaos_swap():
    from intopt.operators.mutate import mutChaosSwap

    sol = np.array([0, 1, 2, 3, 4])
    mutated = mutChaosSwap(sol)
    assert set(mutated) == set(range(5))
    assert len(mutated) == 5
