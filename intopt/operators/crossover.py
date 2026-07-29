import numpy as np

from intopt.operators.utils import chaos_sequence


def cxArithmetic(parent1: np.ndarray, parent2: np.ndarray):
    """算术交叉：每维独立加权平均，用于连续型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代个体 1。
    parent2 : np.ndarray
        父代个体 2。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个子代个体。
    """
    alpha = np.random.random(len(parent1))
    c1 = alpha * parent1 + (1 - alpha) * parent2
    c2 = (1 - alpha) * parent1 + alpha * parent2
    return c1, c2


def cxSimulatedBinary(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta: float = 20,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
):
    """模拟二进制交叉 (SBX)，用于连续型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代个体 1。
    parent2 : np.ndarray
        父代个体 2。
    eta : float
        分布指数，越大子代越接近父代，默认 20。
    lower : np.ndarray or None
        各维度的下界，默认 None 不裁剪。
    upper : np.ndarray or None
        各维度的上界，默认 None 不裁剪。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个子代个体。
    """
    u = np.random.random(len(parent1))
    mask = u <= 0.5
    beta = np.empty(len(parent1))
    beta[mask] = (2 * u[mask]) ** (1 / (eta + 1))
    beta[~mask] = (1 / (2 * (1 - u[~mask]))) ** (1 / (eta + 1))

    c1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    c2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

    if lower is not None:
        c1 = np.clip(c1, lower, upper)
        c2 = np.clip(c2, lower, upper)
    return c1, c2


def cxOnePoint(parent1: np.ndarray, parent2: np.ndarray):
    """单点交叉：随机选一个切点，交换后半段。适用于通用向量型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代个体 1。
    parent2 : np.ndarray
        父代个体 2。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个子代个体。
    """
    n = len(parent1)
    if n < 2:
        return parent1.copy(), parent2.copy()
    point = np.random.randint(1, n)
    c1 = np.concatenate([parent1[:point], parent2[point:]])
    c2 = np.concatenate([parent2[:point], parent1[point:]])
    return c1, c2


def cxTwoPoint(parent1: np.ndarray, parent2: np.ndarray):
    """两点交叉：随机选两个切点，交换中间段。适用于通用向量型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代个体 1。
    parent2 : np.ndarray
        父代个体 2。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个子代个体。
    """
    n = len(parent1)
    if n < 3:
        return cxOnePoint(parent1, parent2)
    i, j = np.sort(np.random.choice(n + 1, 2, replace=False))
    c1 = np.concatenate([parent1[:i], parent2[i:j], parent1[j:]])
    c2 = np.concatenate([parent2[:i], parent1[i:j], parent2[j:]])
    return c1, c2


def cxUniform(
    parent1: np.ndarray, parent2: np.ndarray, indpb: float = 0.5
):
    """均匀交叉：每个位置以概率 ``indpb`` 交换，用于离散/二值型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代个体 1。
    parent2 : np.ndarray
        父代个体 2。
    indpb : float
        每个位置交换的概率，默认 0.5。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个子代个体。
    """
    mask = np.random.random(len(parent1)) < indpb
    c1 = parent1.copy()
    c2 = parent2.copy()
    c1[mask] = parent2[mask]
    c2[mask] = parent1[mask]
    return c1, c2


def cxOrdered(parent1: np.ndarray, parent2: np.ndarray):
    """顺序交叉 (OX)，用于排列型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代排列个体 1。
    parent2 : np.ndarray
        父代排列个体 2。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个排列型子代个体。
    """
    size = len(parent1)
    p1 = parent1.astype(int)
    p2 = parent2.astype(int)
    cx1, cx2 = np.sort(np.random.choice(size, 2, replace=False))

    c1 = np.full(size, -1, dtype=int)
    c2 = np.full(size, -1, dtype=int)
    c1[cx1 : cx2 + 1] = p1[cx1 : cx2 + 1]
    c2[cx1 : cx2 + 1] = p2[cx1 : cx2 + 1]

    _fill_ox(c1, p2, cx1, cx2, size)
    _fill_ox(c2, p1, cx1, cx2, size)
    return c1, c2


def _fill_ox(child, parent, cx1, cx2, size):
    pos = (cx2 + 1) % size
    for i in range(size):
        city = parent[(cx2 + 1 + i) % size]
        if city not in child:
            child[pos] = city
            pos = (pos + 1) % size


def cxPartialyMatched(parent1: np.ndarray, parent2: np.ndarray):
    """部分匹配交叉 (PMX)，用于排列型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代排列个体 1。
    parent2 : np.ndarray
        父代排列个体 2。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个排列型子代个体。
    """
    size = len(parent1)
    p1 = parent1.astype(int)
    p2 = parent2.astype(int)
    cx1, cx2 = np.sort(np.random.choice(size, 2, replace=False))

    c1 = p1.copy()
    c2 = p2.copy()

    for i in range(cx1, cx2 + 1):
        if p2[i] == p1[i]:
            continue

        pos1 = int(np.where(c1 == p2[i])[0][0])
        c1[i], c1[pos1] = c1[pos1], c1[i]

        pos2 = int(np.where(c2 == p1[i])[0][0])
        c2[i], c2[pos2] = c2[pos2], c2[i]

    return c1, c2


def cxChaosArithmetic(parent1: np.ndarray, parent2: np.ndarray):
    """混沌算术交叉：权重由 Logistic 序列生成，用于连续型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代个体 1。
    parent2 : np.ndarray
        父代个体 2。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个子代个体。
    """
    alpha = chaos_sequence(len(parent1))
    c1 = alpha * parent1 + (1 - alpha) * parent2
    c2 = (1 - alpha) * parent1 + alpha * parent2
    return c1, c2


def cxChaosOrdered(parent1: np.ndarray, parent2: np.ndarray):
    """混沌顺序交叉：切点由 Logistic 序列确定，用于排列型。

    Parameters
    ----------
    parent1 : np.ndarray
        父代排列个体 1。
    parent2 : np.ndarray
        父代排列个体 2。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        两个排列型子代个体。
    """
    size = len(parent1)
    p1 = parent1.astype(int)
    p2 = parent2.astype(int)

    ch = chaos_sequence(2)
    cx1, cx2 = np.sort(np.clip(np.floor(ch * size).astype(int), 0, size - 1))
    if cx1 == cx2:
        cx2 = min(cx1 + 1, size - 1)

    c1 = np.full(size, -1, dtype=int)
    c2 = np.full(size, -1, dtype=int)
    c1[cx1 : cx2 + 1] = p1[cx1 : cx2 + 1]
    c2[cx1 : cx2 + 1] = p2[cx1 : cx2 + 1]

    _fill_ox(c1, p2, cx1, cx2, size)
    _fill_ox(c2, p1, cx1, cx2, size)
    return c1, c2


# ---------------------------------------------------------------------------
# DE 二项式交叉
# ---------------------------------------------------------------------------


def cxBinomial(
    target: np.ndarray,
    donor: np.ndarray,
    CR: float = 0.7,
) -> np.ndarray:
    """二项式交叉（DE 用）。

    对目标向量和供体向量执行二项式交叉：
    随机选一个维度 j_rand 强制来自供体，
    其余维度以概率 ``CR`` 取供体，否则保留目标值。

    Parameters
    ----------
    target : np.ndarray
        目标向量。
    donor : np.ndarray
        供体向量。
    CR : float
        交叉概率，默认 0.7。

    Returns
    -------
    np.ndarray
        试验向量。
    """
    dim = len(target)
    j_rand = np.random.randint(0, dim)
    trial = target.copy()
    for j in range(dim):
        if np.random.rand() < CR or j == j_rand:
            trial[j] = donor[j]
    return trial
