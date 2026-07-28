import numpy as np


def test_crossover_operators():
    """各交叉算子返回合法个体"""
    from intopt.operators.crossover import (
        cxArithmetic,
        cxOnePoint,
        cxOrdered,
        cxPartialyMatched,
        cxSimulatedBinary,
        cxTwoPoint,
        cxUniform,
    )

    p1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    p2 = np.array([5.0, 6.0, 7.0, 8.0, 9.0])

    for cx in [cxArithmetic, cxOnePoint, cxTwoPoint, cxUniform]:
        c1, c2 = cx(p1, p2)
        assert c1.shape == (5,) and c2.shape == (5,)

    c1, c2 = cxSimulatedBinary(p1, p2, lower=np.zeros(5), upper=np.ones(5) * 10)
    assert c1.shape == (5,) and c2.shape == (5,)
    assert np.all(c1 >= 0) and np.all(c1 <= 10)
    assert np.all(c2 >= 0) and np.all(c2 <= 10)

    p1 = np.array([0, 1, 2, 3, 4])
    p2 = np.array([4, 3, 2, 1, 0])
    for cx in [cxOrdered, cxPartialyMatched]:
        c1, c2 = cx(p1, p2)
        assert set(c1) == set(range(5))
        assert set(c2) == set(range(5))


def test_chaos_crossover_operators():
    """混沌交叉算子返回合法个体"""
    from intopt.operators.crossover import cxChaosArithmetic, cxChaosOrdered

    p1 = np.array([0.0, 1.0, 2.0])
    p2 = np.array([3.0, 4.0, 5.0])
    c1, c2 = cxChaosArithmetic(p1, p2)
    assert c1.shape == (3,) and c2.shape == (3,)

    p1 = np.array([0, 1, 2, 3, 4])
    p2 = np.array([4, 3, 2, 1, 0])
    c1, c2 = cxChaosOrdered(p1, p2)
    assert set(c1) == set(range(5))
    assert set(c2) == set(range(5))
