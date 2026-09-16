# AquaOps 公开架构说明

本文只解释公开演示面，不代表生产部署设计，也不描述私有水厂数据处理流程。当前实现的
目标是让 Agent、HTTP API 和 stdio MCP 在相同的公开检索器上得到一致、可追溯、只读的证据。

## 一条可复现的文字链路

```text
公开问题
  -> 证据约束的 Agent / API / stdio MCP 入口
  -> 混合召回（BM25 + dense vector）
  -> RRF 融合
  -> 可选 reranker（默认 Torch；公开演示可显式选择 OpenVINO）
  -> chunk_id/source_url/source_version 校验
  -> 固定 ABC 决策支持输出 + 人工复核标记
```

入口层不各自实现一套检索逻辑：公开演示组合根先构造一个经过语料注册和边界校验的
`PublicDemoRuntime`，再把同一个 retriever 注入 Agent graph、ASGI API 和 MCP handler。
因此三入口的引用 ID 集合可以在 smoke test 中比较；任一入口没有公开证据或覆盖门禁失败，
都按受控失败处理，而不是把另一入口的结果拼接过来。

## 公开语料、chunk 与召回边界

- 公开运行时只接受注册的 `public/public_read` 语料；当前演示的来源、版本、归档哈希和
  分块清单位于 `data/public/sources.json` 与 `data/rag/public-demo/corpus.json`。
- 每个 chunk 必须带稳定的 `chunk_id`、HTTP(S) `source_url`、非空 `source_version`、
  公开访问级别和正文。检索器会再次校验这些字段，避免向量库 payload 被误当成可信证据。
- BM25 负责词面召回，dense vector 负责语义相似，RRF 在不依赖单一分数尺度的情况下合并
  候选；reranker 只处理固定候选窗口。公开 CLI 不接受任意模型 ID、路径、collection 名称、
  数据类别或远程 Qdrant 地址。
- 公开评估比较 BM25、dense、Hybrid RRF 和 Hybrid + rerank。指标必须连同评估集版本、
  模型 revision、命令和聚合报告哈希解释；历史本机结果不是生产 SLA。

## ABC 输出与证据覆盖

公开 Agent 的固定输出包含三个维度：

- **A：历史/现状**。只有存在相应历史证据时填写；不能把 C 的公开知识当作 A 的现场数据。
- **B：异常/风险**。只有异常证据或受控诊断结果覆盖时填写；缺少覆盖就明确标记缺失。
- **C：公开知识/处置参考**。引用公开来源、版本和分数，但仍需要人工复核适用条件。

覆盖门禁要求 ABC 三个维度的结构保持稳定；引用只保留 `chunk_id`、`source_url`、
`source_version` 等可追溯元数据，且 `human_review_required` 固定为 `true`。当问题越权、
证据为空、版本不一致或返回内容不满足公开协议时，系统返回受控拒答/降级原因，不用相邻
维度的证据“补齐”缺失维度。

## 最小可观察输出

公开 CLI 只输出聚合或安全摘要，便于复制到日志而不泄漏查询和证据正文：

| 命令 | 可用于验收的字段 | 含义 |
| --- | --- | --- |
| `prepare` | `ok`、`chunk_count`、`corpus_sha256`、模型 ID/revision、reranker 状态 | 公开索引和固定模型是否准备完成 |
| `evaluate` | `case_count`、`k`、四路变体、数据/报告哈希 | 固定公开评估集上的可追溯聚合结果 |
| `smoke` | `ok`、三个入口证据数量、`citation_sets_match`、`abc_dimension_count`、`human_review_required` | Agent/API/MCP 是否共享同一公开证据协议 |

报告不会保存问题、来源正文、本机路径、原始异常或凭据。真实数字应以当前命令生成的
`artifacts/public-demo/public-demo-report.json` 为准；README 中的历史表格只作为已记录实验，
必须按文档中的日期、机器和模型 revision 复验后才能对外引用。

## API、MCP 与安全边界

- HTTP API 的公开演示入口只接收固定模式和受约束的问题，返回证据、覆盖状态和人工复核标记。
- stdio MCP 只暴露 `retrieve_public_water_knowledge`、`aggregate_public_history` 和
  `screen_public_anomaly` 三个公开只读工具；参数和来源域名在 handler 层再次验证。
- 公开演示不提供数据库写入、设备控制、任意路径、远程 collection 或任意插件执行能力。
  高风险写操作不能复用当前 Python 进程中的能力对象，未来必须拆到最小权限服务并增加人工审批。
- 工具错误被归类为固定错误码/降级原因，不把底层异常、密钥或内部路径直接写进响应。

## 预测输出与私有边界

预测输出契约只保证结构化、版本化、可追溯字段（例如模型版本、输入窗口、预测区间、
不确定性、质量门禁和人工复核状态）；它不证明真实水厂预测效果，也不允许公开演示读取私有
监测 CSV/XLS。私有数据导入、私有分析 Agent、私有测试和本地 artifacts 均属于本地隔离面，
不进入公开 RAG、公开 MCP、Docker build context 或 CI。发布候选需要把这些路径从独立公开树中
排除，而不是依赖 README 的口头约定。

## 运行与复核入口

从项目根目录开始，先运行 README 的合成回归，再按[真实公开 RAG 演示](public-rag-demo.md)
准备固定模型并执行离线 evaluate/smoke。发生失败时保留受控聚合报告，先检查公开来源版本、
chunk 边界、候选窗口和 reranker 后端；不要通过加入私有数据、放宽 `public_read` 过滤或
隐藏拒答来改善指标。最终发布前还要完成许可证、历史审计、独立仓库抽取和外部发布授权。
