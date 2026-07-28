import numpy as np


def chaos_sequence(length: int) -> np.ndarray:
    """生成 Logistic 映射混沌序列，值域 (0, 1)。"""
    seq = np.zeros(length)
    seq[0] = np.random.rand()
    for i in range(1, length):
        seq[i] = 4 * seq[i - 1] * (1 - seq[i - 1])
    return seq
