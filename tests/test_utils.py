import numpy as np


def test_fitness_sharing_no_neighbors():
    """个体距离超出阈值时，适应度不变"""
    from intopt.operators.utils import fitnessSharing

    pop = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    fit = np.array([1.0, 2.0, 3.0])
    result = fitnessSharing(pop, fit, distance_threshold=1.0, sharing_extent=1.0)

    assert result.shape == fit.shape
    assert np.allclose(result, fit)


def test_fitness_sharing_reduces_close_individuals():
    """靠近的个体适应度被推高（变差）"""
    from intopt.operators.utils import fitnessSharing

    pop = np.array([[0.0, 0.0], [0.05, 0.05]])
    fit = np.array([5.0, 10.0])
    result = fitnessSharing(pop, fit, distance_threshold=0.1, sharing_extent=1.0)

    assert result.shape == (2,)
    assert result[0] > 5.0


def test_fitness_sharing_larger_extent_reduces_more():
    """sharing_extent 越大，拥挤惩罚越强"""
    from intopt.operators.utils import fitnessSharing

    pop = np.array([[0.0, 0.0], [0.05, 0.05]])
    fit = np.array([5.0, 10.0])

    result_small = fitnessSharing(pop, fit, distance_threshold=0.1, sharing_extent=1.0)
    result_large = fitnessSharing(pop, fit, distance_threshold=0.1, sharing_extent=3.0)

    assert result_large[0] > result_small[0]
