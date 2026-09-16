# 公开 RAG 回归评估

## 两套评估的用途

- `data/rag/evals/public-water-rag-v1.json` 是原有的**合成安全回归**，用假检索器和
  内存对象验证协议、拒答和指标计算，不代表真实模型效果。
- `data/rag/evals/public-water-rag-real-v1.json` 是哈希锁定的**真实公开演示基线**，
  含 50 个普通公开问题和 6 个越权拒答案例，期望 ID 全部指向
  `data/rag/public-demo/corpus.json` 产生的 24 个真实公开分块。

真实基线固定比较 `bm25`、`dense`、`hybrid_rrf`、`hybrid_rerank`。运行命令、
2026-08-12 当前本机的实测结果和聚合报告说明见
[公开 RAG 真实演示](public-rag-demo.md)。两套评估都不得使用私有水厂数据。

`data/rag/evals/public-water-rag-v1.json` 是 AquaOps 的**合成公开数据回归契约**。它用于防止公开检索链路在重构后丢失基本召回、泄露非公开证据或把故障伪装成回答；它不是生产效果、真实水厂效果或模型能力的宣称。

## 数据边界

- 数据集和每个案例都固定为 `data_class: public`、`access_policy: public_read`。
- 普通案例仅含合成中文运维问题、主题和 64 位十六进制期望证据 ID；另有至少一个 `requires_refusal: true` 的合成访问边界案例，它以“请求虚构水厂的未公开原始监测记录”这一直接越权请求为回归标注，期望证据 ID 必须为空。这不是记录样本，两类案例都不含企业、人员、厂名、地址、真实监测记录、原文知识正文或私有数据。
- loader 只允许仓库内唯一注册的 canonical 文件；会拒绝外部路径、符号链接、越过 `data/rag/evals` 边界的路径，以及 SHA-256 manifest、schema、访问级别、案例 ID 或证据 ID 被篡改的文件。
- `load_public_evaluation_cases()` 返回带内部验证标记的 `PublicEvaluationSuite`；`evaluate_retriever` 不接受手工构造的案例元组或套件，避免把看似 `public_read` 的外部数据混入回归结果。
- 评估报告不保存查询、证据正文、URL、来源信息或原始异常；每个案例只保留案例 ID、命中 ID、耗时和受控错误类别。

## 运行回归测试

在项目根目录执行：

```powershell
.venv\Scripts\python.exe -m pytest tests\rag\test_evaluation.py tests\rag\test_hybrid.py -q
```

测试使用内存中的合成检索器与可注入时钟，不下载 embedding 模型、不启动 Docker、不连接 Qdrant，也不读取任何本地水厂文件。

## 指标定义

`evaluate_retriever(retriever, suite, k=5, latency_clock=...)` 只接收 `load_public_evaluation_cases()` 发出的不可变公开套件，并按 `retriever.retrieve(query).evidence` 的前 `k` 条结果计算。每条证据必须是运行时精确的 `PublicEvidence`，且会重新校验 64 位 chunk ID、HTTP(S) URL、非空版本/正文、`public/public_read` 标签与有限 `float` score。`fused_candidate_ids` 也必须是原生、去重的 64 位 ID 元组，并至少包含每条返回证据的 chunk ID；任何畸形或不一致都会受控拒答：

- `Recall@k`：仅在普通案例中计算，至少命中一个期望证据 ID 的案例数 / 普通案例数；没有普通案例时为 `null`（不可计算）。
- `MRR@k`：仅在普通案例中计算，每个案例第一个期望证据 ID 的倒数排名之和 / 普通案例数；拒绝、异常和未命中按 0 计。预期拒答案例不稀释检索指标；没有普通案例时为 `null`。
- `answered_count`：普通案例中，返回了结构正确、全部为 `public/public_read` 的非空证据的案例数。
- `refusal_count`：普通案例的技术性受控拒答数，包括空结果、检索异常、畸形结果、非公开证据或无效证据 ID。它不把“本应拒答且正确拒答”的案例误当作系统故障。
- `refusal_accuracy`：`requires_refusal` 案例中正确拒答的比例；没有此类案例时为 `null`。任何结构合法的公开证据都会使该案例变成 `unsafe_answer`，不会计入 `answered_count`。
- `P95 latency`：所有有限、非负的可测调用毫秒耗时按 nearest-rank 计算；单个样本的 P95 就是该样本本身。若时钟不可用、返回 `NaN`/`Infinity`、倒退或计算后溢出，案例会标成 `clock_error`，该值不会进入 P95；没有有效耗时时报告为 `null`。

每份报告还带有已验证的 `dataset_id` 与 `schema_version`，便于把结果和固定回归契约对应起来。

检索器抛出的原始异常不会离开评估接口；普通案例只会被归类为例如 `retriever_error`、`malformed_result`、`non_public_evidence` 或 `no_evidence`。预期拒答案例会被归类为 `correct_refusal` 或 `unsafe_answer`，不保留正文或原始异常。

## 真实公开 Qdrant 与 embedding

真实演示现在由固定命令显式完成：首次运行会下载代码中锁定提交版本的模型，后续可用
`--offline` 复用。评估器不会拉取任意外部语料，也不接受模型、路径、集合或数据类别
覆盖。

1. 仅准备允许公开发布的文档，并通过公开语料分块流程生成带 `public/public_read` 标签的 chunk；不要把私有水厂数据、导出的工单、架构细节或监测记录放入该集合。
2. 使用锁定 revision 的 BGE-M3、固定预处理和离线缓存生成 1024 维有限浮点向量。
3. 真实演示默认使用不监听端口的进程内本地 Qdrant，并通过 `QdrantPublicKnowledgeStore` 初始化固定集合 `water_public_knowledge_v1`。不要改为任意 collection 名称，也不要指向远程或生产 Qdrant。
4. 将公开 BM25、公开 dense search 和 reranker 以注入式 adapter 接到 `HybridPublicRetriever`；它们都必须返回受公共标签约束的结果。先做一次人工抽检，再运行本评估集。
5. 在首次接入真实本地 Qdrant 时，如需检查向量库契约，按照 README 中的双环境变量开关运行隔离集成测试。该测试只允许 `localhost`/`127.0.0.1`，并创建后删除临时合成集合。

## 建议验收门槛

以下是本地公开回归的起始门槛，不是线上 SLA：

| 指标 | 建议门槛 | 解释 |
| --- | --- | --- |
| Recall@5 | `>= 0.80` | 合成案例至少应能找回大多数目标 ID。 |
| MRR@5 | `>= 0.60` | 目标 ID 应尽量靠前，而不是只偶然出现在末尾。 |
| technical refusal rate | `<= 5%` | `refusal_count / 普通案例数`；超过时先排查 provider、标签和协议。 |
| refusal accuracy | `= 1.00` | 每个 `requires_refusal` 案例都必须安全拒答；任一 `unsafe_answer` 都阻断验收。 |
| P95 latency | `<= 800 ms` | 仅作为当前机器、固定模型与固定索引下的基线，需记录运行环境。 |

当结果没有达到门槛时，应先检查公开来源版本、chunk 边界、BM25/dense 候选量、融合和 rerank 顺序；不要通过加入私有数据、放宽 `public_read` 过滤或隐藏拒绝来“改善”指标。
