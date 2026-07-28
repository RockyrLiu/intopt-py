class EarlyStopping:
    """早停监视器。

    监测 history 中指定键的值，若连续 ``patience`` 次未出现超过
    ``min_delta`` 的改善则触发早停。
    """

    def __init__(self, key: str = "best", patience: int = 20, min_delta: float = 1e-6):
        self.key = key
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._baseline = None
        self._triggered = False

    def step(self, history: dict) -> bool:
        """记录当前步，返回是否应停止。"""
        current = history[self.key][-1]
        if self._baseline is None:
            self._baseline = current
            return False

        if self._baseline - current > self.min_delta:
            self._baseline = current
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self._triggered = True
        return self._counter >= self.patience

    @property
    def triggered(self) -> bool:
        return self._triggered
