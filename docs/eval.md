# Mimir 有效性评估方案

> 本方案定义如何验证 Mimir（原型可塑网络）在推理、学习、遗忘、工程四个维度上的有效性。
>
> 实施方式：
> - `docs/eval.md`：评估体系设计（本文档）
> - `mimir/eval.py`：可复现的评估脚本（已实现）
> - `mimir/tests/test_convergence.py`：关键收敛性测试（已实现）
> - `mimir/tests/test_p1_features.py`：P1 功能测试（已实现）

---

## 一、评估原则

1. **可复现**：固定随机种子、固定输入文本、固定超参数。
2. **可量化**：每个维度至少有一个数值指标和明确通过标准。
3. **可解释**：失败时能从指标反推哪个模块出问题。
4. **分层**：P0 只验证学习和推理；P1 加入惊喜信号；P2 加入工程 benchmark。

---

## 二、评估维度与指标

### 2.1 学习收敛性（P0 核心）

验证：模型学到新语义后，相似输入的 embedding 距离应缩小。

| 测试 | 方法 | 通过标准 |
|---|---|---|
| **同语义收敛** | 准备 5 条同主题文本（如水果相关）。先 encode 得到基线 pairwise cosine similarity。再 learn 10 次。最后再次 encode 计算 pairwise similarity。 | 平均 similarity 提升 > 10% |
| **异语义分离** | 准备 5 条不同主题文本（水果、编程、历史、金融、体育）。学习前后分别计算跨主题 pairwise similarity。 | 跨主题 similarity 不显著上升（变化 < 5%） |
| **残差激活** | 学习前后分别记录 `norm(residual)`。 | 学习后 residual norm 显著 > 学习前 |

**反推诊断**：
- 同语义不收敛 → `OjaLearningPolicy` 学习率或 `update_nearest` 逻辑有问题。
- 异语义不分离 → softmax 温度过高或原型数太少。
- 残差不增长 → `residual_scale` 太小或原型未被激活。

---

### 2.2 遗忘行为（P0 核心）

验证：系统能按预期衰减旧记忆，并在容量满时稳定运行。

| 测试 | 方法 | 通过标准 |
|---|---|---|
| **强度衰减曲线** | 学习一条文本后，连续调用 100 次 `decay()`（模拟 100 步无访问），记录 strength。 | strength 单调递减，最终趋近于 0 |
| **容量淘汰** | 设置 `num_prototypes=8`，学习 16 条完全不同主题的文本。检查最早学习的文本对应的 prototype 是否被覆盖。 | 旧 prototype 被覆盖，系统不 crash |
| **遗忘后稳定性** | 容量满后，encode 任意文本，输出无 NaN/Inf。 | 输出合法（`torch.isfinite` 全为 True） |

**反推诊断**：
- strength 不衰减 → `forgetting_decay` 未生效或 strength 更新逻辑错误。
- 容量满后 crash → `update_nearest` 或 `decay` 越界。

---

### 2.3 惊喜信号与预测（P1）

验证：`surprise_score` 能有效区分熟悉输入与新输入，且重复序列的下一原型可被预测。

| 测试 | 方法 | 通过标准 |
|---|---|---|
| **重复序列预测** | 用 3 条同主题文本构造 A→B→C 循环，学习 10 轮。记录最后一次 C 之后预测的原型。 | 预测结果等于 A（即序列中的下一个原型） |
| **熟悉 vs 新奇** | 文本 A 学习 10 次，文本 B 从未学习。分别调用 `learn()` 并记录 `surprise_score`。 | 文本 B 的 `surprise_score` 显著高于文本 A |
| **连续对话预测** | 构造 20 轮对话，前 10 轮反复出现同一主题，后 10 轮切换主题。记录每轮 `surprise_score`。 | 主题切换时 `surprise_score` 出现明显峰值 |

新增评估字段：

| 字段 | 含义 |
|---|---|
| `predicted_after_c` | 重复序列中 C 之后预测出的下一个原型 id（等于 A 则预测正确） |
| `proto_a` / `proto_b` / `proto_c` | 序列 A→B→C 实际对应的原型 id |
| `surprise_expected` | 期望转移（A→B）的 surprise 分数，应接近 0 |
| `surprise_unexpected` | 非期望转移（A→C）的 surprise 分数，应接近 1 |
| `mean_random_surprise` | 随机历史主题序列的平均 surprise 分数，应较高 |

**反推诊断**：
- 无法预测重复序列 → 转移矩阵未正确更新，或同一文本映射到不同原型。
- surprise 无法区分期望/非期望 → 平滑系数过大，或学习率过低导致原型未稳定。

---

### 2.4 工程性能（P0/P1）

验证：系统在实际使用场景下具备可接受的延迟和资源占用。

| 指标 | 方法 | 目标 |
|---|---|---|
| **推理延迟增量** | 对 1000 条文本分别用 base 模型和 `mimir.encode()` 计时，计算中位数延迟。 | Mimir 中位数延迟 < base 模型延迟 × 1.2 |
| **批量推理吞吐** | 对 batch_size=32/64/128 分别测试 100 次 encode。 | 吞吐随 batch 线性增长，无异常下降 |
| **状态文件体积** | `save()` 后检查 checkpoint 文件大小。 | < 100 MB（k=1024, dim=768） |
| **内存占用** | 连续 1000 次 `learn()` 后观察内存 RSS。 | 无持续增长，无 OOM |

**反推诊断**：
- 延迟过高 → `PrototypeStore.lookup()` 未优化或 CPU/GPU 数据搬运过多。
- 内存泄漏 → EventBus 或 checkpoint 持有未释放引用。

---

## 三、评估数据集

为避免引入外部依赖，评估使用内置的小型合成数据集：

```text
fruit_theme = [
    "苹果富含维生素 C",
    "香蕉是钾的良好来源",
    "橙子味道酸甜",
    "葡萄可以直接食用",
    "草莓是红色的浆果",
]

code_theme = [
    "Python 支持异步编程",
    "Rust 的所有权系统保证内存安全",
    "函数式编程强调不可变性",
    "单元测试能提高代码质量",
    "Docker 容器化简化了部署",
]

history_theme = [
    "秦始皇统一了六国",
    "唐朝是中国古代强盛的朝代",
    "丝绸之路连接了东西方",
    "明朝郑和下西洋",
    "清朝末年发生了鸦片战争",
]
```

评估时固定使用这些文本，便于横向比较。

```bash
python -m mimir.eval
python -m mimir.eval --backend llama-server
python -m mimir.eval --backend llama-server --top-k 4
python -m mimir.eval --backend sentence-transformer
```

### 当前报告字段

| 字段 | 类型 | 含义 | 出现条件 |
|---|---|---|---|
| `fruit_sim_delta` | float | 水果主题学习前后平均 pairwise cosine similarity 变化 |  always |
| `code_sim_delta` | float | 代码主题学习前后平均 pairwise cosine similarity 变化 | always |
| `cross_sim_delta` | float | 跨主题 pairwise cosine similarity 变化 | always |
| `residual_norm_before` | float | 学习前残差 L2 norm | always |
| `residual_norm_after` | float | 学习后残差 L2 norm | always |
| `residual_norm_delta` | float | 学习前后残差 L2 norm 变化 | always |
| `embedding_shift` | float | 学习前后同文本 embedding 平均 L2 距离 | always |
| `initial_strength` | float | 遗忘测试初始 strength 总和 | always |
| `final_strength` | float | 遗忘测试最终 strength 总和 | always |
| `strength_decay_ratio` | float | 最终 strength / 初始 strength | always |
| `state_finite` | float | 输出是否合法（无 NaN/Inf），合法为 1.0 | always |
| `base_time_ms` | float | base 编码耗时（毫秒） | always |
| `mimir_time_ms` | float | Mimir 编码耗时（毫秒） | always |
| `overhead_ratio` | float | Mimir 总耗时 / base 编码耗时 | always |
| `predicted_after_c` | float | 重复序列中 C 之后预测的下一个原型 id | always |
| `proto_a` / `proto_b` / `proto_c` | float | 重复序列 A→B→C 实际对应的原型 id | always |
| `surprise_expected` | float | 期望转移的 surprise 分数 | always |
| `surprise_unexpected` | float | 非期望转移的 surprise 分数 | always |
| `mean_random_surprise` | float | 随机序列的平均 surprise 分数 | always |

---

## 实施计划

### 当前阶段（P0/P1）

- [x] 在 `mimir/eval.py` 中实现收敛、遗忘、延迟评估
- [x] 在 `mimir/eval.py` 中实现预测准确率与 surprise 评估
- [x] 在 `mimir/tests/test_convergence.py` 中实现同语义收敛回归测试
- [x] 支持 `--backend` 切换 FakeEngine / llama-server / sentence-transformer
- [x] 支持 `--top-k` 评估稀疏激活效果
- [x] 记录基线结果到 `docs/benchmarks.md`

### P1 后续

- [ ] 加入 `test_surprise()` 评估惊喜信号
- [ ] 加入 `test_prediction_accuracy()` 评估转移矩阵预测准确率

### P2 阶段

- [ ] 加入 `test_latency()`、`test_throughput()`、`test_memory()` 工程 benchmark
- [ ] 将 benchmark 结果写入 `docs/benchmarks.md`

---

*创建于 2026-06-27 | 对应 P0/P1 评估体系*
