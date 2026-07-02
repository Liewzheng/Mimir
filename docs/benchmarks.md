# Mimir Benchmarks

> 可复现的评估结果记录。
>
> 符号说明：> - `PASS`：达到该维度预设通过标准。> - `INFO`：无预设通过标准，仅作参考。> - `state_finite`：输出是否合法（无 NaN/Inf），在所有实际运行中均为 `1.0`。

---

## 运行命令

```bash
# 合成数据（FakeEngine，dim=16）
python -m mimir.eval

# 本地 llama-server 真实模型；不指定 --top-k 时使用全 softmax 激活
python -m mimir.eval --backend llama-server

# 本地 llama-server + top-k=4 稀疏激活
python -m mimir.eval --backend llama-server --top-k 4

# sentence-transformers
python -m mimir.eval --backend sentence-transformer
```

---

## 结果 1：合成数据（FakeEngine，dim=16）

评估配置：

| 参数 | 值 |
|---|---|
| `num_prototypes` | 8 |
| `learning_rate_base` | 0.1 |
| `learning_rate_decay` | 0.1 |
| `temperature` | 0.5 |
| `residual_scale` | 0.3 |
| `forgetting_decay` | 0.995 |
| `learn_iterations` | 20 |
| `random_seed` | 42（固定） |

结果：

| 维度 | 指标 | 结果 | 状态 |
|---|---|---|---|
| 收敛性 | 水果主题 similarity delta | +0.037504 | PASS |
| 收敛性 | 代码主题 similarity delta | +0.031480 | INFO |
| 收敛性 | residual norm delta | +0.027725 | PASS |
| 收敛性 | embedding shift (L2) | 0.078467 | INFO |
| 分离性 | 跨主题 similarity delta | +0.010457 | PASS |
| 遗忘 | strength 衰减比例 | 0.606 | PASS |
| 状态 | 输出合法（state_finite） | 1.000 | PASS |
| 延迟 | Mimir / base 延迟比 | 1.138 | PASS |
| 预测 | predicted_after_c | 0.000000 | PASS |
| 惊喜 | surprise_expected | 0.471774 | PASS |
| 惊喜 | surprise_unexpected | 0.552419 | PASS |
| 惊喜 | mean_random_surprise | 0.821637 | INFO |

**Overall: PASS**

---

## 结果 2：真实模型（Qwen3-Embedding-8B，dim=4096）

使用本地 `llama-server --embeddings` 服务，模型为 `dengcao/Qwen3-Embedding-8B:Q5_K_M`。
服务启动示例：

```bash
llama-server \
  --model /path/to/your/ollama/models/blobs/sha256-... \
  --port 11435 --host 127.0.0.1 --embeddings --ctx-size 4096
```

> 注意：请将 `/path/to/your/...` 替换为你本地的模型路径。

评估配置（真实模型已高度聚类，因此使用更激进的学习率）：

| 参数 | 值 |
|---|---|
| `num_prototypes` | 16 |
| `learning_rate_base` | 0.5 |
| `learning_rate_decay` | 0.05 |
| `temperature` | 0.3 |
| `residual_scale` | 1.0 |
| `forgetting_decay` | 0.995 |
| `learn_iterations` | 20 |
| `random_seed` | 42（固定） |

结果：

| 维度 | 指标 | 结果 | 状态 |
|---|---|---|---|
| 收敛性 | 水果主题 similarity delta | +0.002318 | PASS |
| 收敛性 | 代码主题 similarity delta | +0.002582 | INFO |
| 收敛性 | residual norm delta | +0.202645 | PASS |
| 收敛性 | embedding shift (L2) | 0.454766 | INFO |
| 分离性 | 跨主题 similarity delta | +0.003144 | PASS |
| 遗忘 | strength 衰减比例 | 0.606 | PASS |
| 状态 | 输出合法（state_finite） | 1.000 | PASS |
| 延迟 | Mimir / base 延迟比 | 1.004 | PASS |
| 预测 | predicted_after_c | 5.000000 | PASS |
| 预测 | proto_a / proto_b / proto_c | 5 / 3 / 12 | PASS |
| 惊喜 | surprise_expected | 0.102740 | PASS |
| 惊喜 | surprise_unexpected | 0.993151 | PASS |
| 惊喜 | mean_random_surprise | 0.864328 | INFO |

**Overall: PASS**

### 关键观察

1. **真实模型已经聚类很好**：基础水果/代码主题内部相似度已达 0.62，Mimir 的提升空间被压缩到 ~0.002。
2. **残差显著增长**：residual norm 从 0.25 增至 0.45，说明原型矩阵确实在学习并影响输出。
3. **embedding shift 明显**：学习前后水果文本的 L2 距离平均 0.45，证明 `learn()` 改变了编码。
4. **延迟几乎无 overhead**：llama-server 的推理耗时占主导，Mimir 的矩阵运算可忽略。
5. **预测与惊喜可区分**：重复序列 A→B→C 映射到三个不同原型，期望转移 surprise 显著低于非期望转移。

---

## 结果 3：真实模型 + top-k=4 稀疏激活

评估配置：在结果 2 基础上增加 `top_k=4`。

结果：

| 维度 | 指标 | 结果 | 状态 |
|---|---|---|---|
| 收敛性 | 水果主题 similarity delta | +0.004332 | PASS |
| 收敛性 | 代码主题 similarity delta | +0.004735 | INFO |
| 收敛性 | residual norm delta | +0.290026 | PASS |
| 收敛性 | embedding shift (L2) | 0.894969 | INFO |
| 分离性 | 跨主题 similarity delta | +0.005803 | PASS |
| 遗忘 | strength 衰减比例 | 0.606 | PASS |
| 状态 | 输出合法（state_finite） | 1.000 | PASS |
| 延迟 | Mimir / base 延迟比 | 1.023 | PASS |
| 预测 | predicted_after_c | 5.000000 | PASS |
| 预测 | proto_a / proto_b / proto_c | 5 / 3 / 12 | PASS |
| 惊喜 | surprise_expected | 0.102740 | PASS |
| 惊喜 | surprise_unexpected | 0.993151 | PASS |
| 惊喜 | mean_random_surprise | 0.864328 | INFO |

**Overall: PASS**

### 与全 softmax 对比

| 指标 | 全 softmax | top-k=4 | 变化 |
|---|---|---|---|
| fruit_sim_delta | +0.002318 | +0.004332 | +87% |
| residual_norm_delta | +0.202645 | +0.290026 | +43% |
| embedding_shift | 0.454766 | 0.894969 | +97% |
| overhead_ratio | 0.933 | 1.037 | 可忽略 |

- top-k 稀疏激活在真实模型上显著增强了学习效果。
- 原因可能是：限制参与残差计算的原型数量后，更新更集中，避免了大量弱激活原型的“平均化”效应。
- **当前默认激活方式仍为全 softmax**；top-k 在 P1 验证有效，计划在 P1 末段或 P2 切换为默认。

---

## 结论

- P0 核心机制在合成数据和真实模型上均通过验证。
- 真实模型上 Mimir 的作用是**微调**而非**重构**编码，符合“快权重调制”的设计定位。
- **P1 top-k 稀疏激活**在真实模型上进一步提升了学习信号强度，推荐后续作为默认配置。
- **P1 prediction / surprise 评估通过**：重复序列可被正确预测，期望转移与非期望转移的 surprise 显著可区分。

---

## 可复现性

- 所有结果使用 `eval.py` 中固定的随机种子（`seed=42`）生成。
- 合成数据结果为单次运行值；真实模型结果受本地 llama-server 负载影响，可能有小幅波动。
- 历史基线（包括旧版详细原始数据）可在 Git 历史 `docs/benchmarks.md` 的早期版本中查阅。

---

*最后更新：2026-06-28*
