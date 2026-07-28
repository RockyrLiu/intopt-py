## 待实现

### 混沌版 GA

- 混沌初始化种群（`_generate_chaos_sequence` 生成 Logistic 映射序列，替代均匀随机）
- 混沌交叉（用混沌序列替代均匀随机生成交叉系数 alpha）
- 混沌变异（用混沌序列替代高斯噪声）
- 可考虑作为 GA 的 `init_strategy` / `crossover_strategy` / `mutation_strategy` 参数，或作为独立的 `ChaosWrappedProblem` 包装器
- 参考：`archive/GA.py` 中的 `ChaosContinuousGA` 和 `ChaosTSP_GA`
