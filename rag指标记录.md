# RAG 检索指标记录

> 评测数据集：[qihoo360/InduOCRBench](https://huggingface.co/datasets/qihoo360/InduOCRBench)
> 中文企业技术文档（12 行业 / 570 份 PDF / 2071 题），标注格式为 Hybrid Markdown（含 HTML 表格 + LaTeX）。
>
> 评测脚本：`tests/eval/run_eval.py`，原始结果：`tests/eval/eval_result.json`

---

## 评测一：Hybrid 检索（dense+sparse RRF）vs 纯 Dense

**日期**：2026-06-25
**样本**：50 题（从 2071 题中筛选"检索友好"题型，剔除对抗性/统计类）
**灌库**：46 份文档 → 959 chunks（用 doc_md 标准标注直接灌库，排除 OCR 误差）
**分块**：recursive 策略，chunk_size=500，overlap=100（与生产一致，含 `</table>` 保护）

### 选题口径

保留（检索可命中）：Basic Recognition / Structural Alignment / Cross-Field Continuity / Complex Reasoning
剔除（需 LLM 推理，非检索能解决）：Statistical/Counting / 各种 *Attack / Aggregation

### 结果

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR |
|------|:-----:|:-----:|:-----:|:---:|
| dense（无 rerank） | 0.66 | 0.86 | 0.90 | 0.759 |
| **hybrid（无 rerank）** | 0.66 | 0.84 | 0.86 | 0.751 |
| dense + rerank | 0.78 | 0.94 | 0.96 | 0.857 |
| **hybrid + rerank** | 0.78 | 0.94 | 0.96 | 0.857 |

### 结论

1. **sparse 在本数据集无提升，甚至略降（Hit@5: 0.90→0.86）**。诚实记录为负面结果。
2. **reranker 是召回质量的关键**：把 Hit@3 从 0.86 拉到 0.94（+9.3%），贡献远大于 sparse 路。
3. **reranker 抹平了召回顺序差异**：hybrid 改变了进 reranker 的候选顺序，但 cross-encoder 对候选独立打分，顺序不影响最终结果，故 hybrid+rerank 与 dense+rerank 完全一致。

### 负面结果的技术分析

为何 sparse 没发挥作用：

- **数据集特性**：InduOCRBench 的题 90%+ 是表格精确查找（evidence 是 `<tr><td>` HTML 片段）。同一表格的 chunk 语义高度集中，dense 向量已能精确定位，sparse 的词项匹配反而是噪声——含相同词项但不同行的 chunk 会被提前。
- **中文 tokenizer 局限**：BGE-M3 的 sparse 基于 XLM-RoBERTa 子词分词，中文一字多 token，词项匹配的精确度不如英文，sparse 信号弱。
- **sparse 的真实价值场景**：英文为主、术语/缩写密集（如 "CIoU"、"BERT"、"RESTful"）的技术文档，或 reranker 缺席/候选量极大来不及全量的场景。本数据集（中文表格）不满足。

### 对项目的启示

- **保留 hybrid 代码但承认当前无实测收益**：sparse 路对英文术语场景仍有理论价值，代码已实现且经测试，但简历叙事需调整（不能声称"提升召回率"）。
- **真正的提升点是 reranker**：Hit@3 从 0.86→0.94 是实测数据，简历应强调"两阶段检索（召回→rerank 精排）将 Top-3 命中率提升至 94%"。
- **sparse 路记录为"已实现、待英文场景验证"**：诚实记录在项目不足里。

---

## 评测二：Reranker 价值（dense 召回 → rerank 精排）

从上表提取的完整对比（行=方法，列=指标，与评测一一致）：

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR |
|------|:-----:|:-----:|:-----:|:---:|
| dense（无 rerank） | 0.66 | 0.86 | 0.90 | 0.759 |
| hybrid（无 rerank） | 0.66 | 0.84 | 0.86 | 0.751 |
| dense + rerank | 0.78 | 0.94 | 0.96 | 0.857 |
| hybrid + rerank | 0.78 | 0.94 | 0.96 | 0.857 |
| **reranker 提升（dense）** | **+18.2%** | **+9.3%** | **+6.7%** | **+12.9%** |

**关键发现**：

1. **reranker 是核心**：dense 加 rerank 后 Hit@3 从 0.86→0.94（+9.3%），MRR +12.9%。两阶段检索（召回 Top-20 → 精排 Top-3）是本项目检索质量的核心保障。
2. **hybrid+rerank 与 dense+rerank 完全相同**：这不是 sparse 无用的证据,而是 reranker 太强——cross-encoder 对候选独立打分后重排,抹平了召回阶段的顺序差异。sparse 改变了进 reranker 的候选顺序,但 reranker 会把对的重新排上来。
3. **真正要看 sparse 价值,看"无 rerank"两行**：hybrid(无rerank) Hit@5=0.86 反而低于 dense(无rerank) 的 0.90——在本数据集(中文表格)sparse 是负收益(评测一已分析原因)。

---

## 环境与复现

```bash
# 确保已下载 InduOCRBench 到项目根目录（仅 RAG_eval 部分）
# huggingface-cli download qihoo360/InduOCRBench --repo-type dataset \
#   --local-dir ./InduOCRBench --include "RAG_eval/*"

# 运行评测（默认抽样 50 题）
.venv/Scripts/python.exe tests/eval/run_eval.py --sample 50
```

- 评测用独立 Chroma 库（`data/chroma_eval/`），不污染生产数据
- 命中判定：检索 chunk 去空白后是否包含 evidence 前 30 字符指纹
- GPU 加速：BGE-M3/Reranker 走 CUDA（encode <1s/batch）

---

## 评测三：Query 改写端到端实测

**日期**：2026-06-25
**目的**：验证多轮指代场景下，rewrite_query 节点是否能正确消解指代并提升检索质量。
**脚本**：`tests/eval/test_rewrite_e2e.py`
**数据**：doc_user_2（简历文档，76 chunks）

### 测试设计

构造多轮指代对话（轮1完整问题建立上下文，轮2用指代词）：

```
轮1(完整): AI驱动的数据处理平台的项目背景是什么？
轮2(指代): 它的技术栈有哪些？    ← "它"指代数据处理平台
```

对比"绕过改写（直接用原指代问题检索）" vs "走改写（rewrite_query 消解后检索）"。

### 结果

| | 绕过改写 | 走改写 |
|---|:---:|:---:|
| 改写后 query | （原文）"它的技术栈有哪些？" | "AI驱动的数据处理平台的技术栈有哪些？" |
| Top-1 rerank 分数 | 0.3367 | **0.9914** |
| Top-2 内容 | 无关（问答系统概述） | 技术栈相关（Python/Vue3/全栈） |

**改写使 Top-1 rerank 分数提升 194%（0.34 → 0.99）。**

### 结论

1. **指代消解正确**：LLM 准确把"它"消解成"AI驱动的数据处理平台"，改写后 query 语义完整。
2. **检索质量显著提升**：rerank 分数从 0.34 飙到 0.99，Top-2 从无关内容变成技术栈相关。
3. **Query 改写的价值在"命中质量"而非"是否命中"**：本例中 dense 语义够强，改写前后 Top-1 都命中了同一文档，但改写后的语义匹配精准度大幅提高——这对后续生成质量（答案准确性）有直接影响。

### 评测边界说明

Query 改写**不能用 InduOCRBench 的 Hit/MRR 指标评测**，因为：
- InduOCRBench 是单轮独立查询（无历史），rewrite_query 会直接跳过（空历史）
- 改写的作用依赖多轮上下文，单轮评测测不到

因此 query 改写用**功能性实测**（指代消解 + rerank 分数对比）背书，而非 Hit Rate 数字。
