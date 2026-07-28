import numpy as np


def metropolis(new_fit: float, old_fit: float, T: float) -> bool:
    """Metropolis 接受准则。

    更优解总是接受；更差解以概率 exp(-ΔE/T) 接受。
    温度 T 越高，接受差解的概率越大，有利于跳出局部最优。

    Parameters
    ----------
    new_fit : float
        新解的适应度。
    old_fit : float
        当前解的适应度。
    T : float
        当前温度。

    Returns
    -------
    bool
        ``True`` 表示接受新解。
    """
    if new_fit < old_fit:
        return True
    if T <= 0:
        return False
    return np.random.random() < np.exp((old_fit - new_fit) / T)
