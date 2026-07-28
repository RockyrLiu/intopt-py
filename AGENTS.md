# AGENTS.md

## 常用命令

```bash
# 使用 uv 管理依赖（包括uv run scripts.py、uv add 等），不要用系统 Python
uv run pytest tests/ -v               # 运行所有测试 (~40s)
uv run pytest tests/test_sa.py -v     # 运行单个测试文件
uv run pytest tests/ -v -k "sphere"   # 按名称匹配运行测试
uv run ruff check intopt/ tests/      # lint（pyproject.toml 中无自定义配置，使用默认规则）
```

## 开发原则

- 采用 TDD（测试驱动开发）：先写测试，用户审阅后再实现功能
- 测试采用函数式风格（顶层函数，不用测试类）
- 每次修改、创建 `.py` 文件后运行 `uv run ruff check intopt/ tests/` 确保无 lint 错误

## 项目架构

- `intopt/` 是库包（通过 hatchling 构建）。
- `archive/` 存放重构前的旧脚本 —— **不属于包的一部分**，禁止从该目录导入。
- `examples/` 存放从 `intopt` 导入的示例脚本。

### 包结构

```
intopt/
  algorithms/     # Optimizer 抽象基类 + 具体算法（目前仅 SA）
  operators/      # 变异算子（mutGaussian, mutSwap, mutFlip）
  problems/       # Problem 抽象基类 + ContinuousProblem, TSPProblem, KnapsackProblem
  visualize/      # 基于 matplotlib 的收敛曲线与 TSP 路径绘图
```

### 设计规则

- 所有问题均**求最小值**（适应度越低越好）。
- `Problem` 子类拥有自己的 `mutate` 方法 —— 内部委托给 `intopt.operators` 执行具体操作，算法代码只需调用 `problem.mutate()`。
- `Optimizer.run()` 始终返回 `OptimizeResult`（dataclass：`best_solution`、`best_fitness`、`history`）。
- `history` 是一个字典，至少包含键 `"best"`；算法可添加其他键（如 `"current"`、`"avg"`）。
- 算子命名采用 DEAP 风格前缀：变异 `mut*`（如 `mutGaussian`）、交叉 `cx*`（如 `cxArithmetic`）、选择 `sel*`（如 `selTournament`）。

## Git 提交规范

- 使用 conventional commits 格式：`类型: 中文描述`
- 常用类型：`feat:`（新功能）、`refactor:`（重构）、`test:`（测试）、`chore:`（杂项）、`docs:`（文档）、`fix:`（修复）等
- 描述简洁，不使用句号结尾
- **除非用户明确要求，否则不要自行 git commit**

## 注意事项

- visualize 模块硬编码了中文字体（`SimHei`、`Source Han Sans CN`）。如果系统未安装这些字体，matplotlib 会显示 tofu/方框。这是为中文 README 演示而有意设计的。
- 测试套件虽小但较慢（约 40s），因为 `test_sa_rastrigin_10d_converges` 和 `test_sa_tsp_data_improves` 运行完整的 SA 退火流程。在不理解测试意图的情况下，不要降低收敛阈值。
- 无 CI，无 pre-commit hooks。Ruff 无自定义配置 —— 仅在 pyproject.toml 中作为开发依赖。
