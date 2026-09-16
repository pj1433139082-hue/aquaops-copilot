# AquaOps 公开 RAG 真实演示

## 交付范围

这套演示只使用经过仓库哈希锁定的公开资料：8 个美国 EPA 官方来源、24 个原创
中文摘要分块、50 个普通检索问题和 6 个越权拒答案例。运行链路为：

`公开语料 → BGE-M3 1024 维向量 → 本地 Qdrant + BM25 → RRF → BGE reranker → Agent/API/MCP`

Agent、ASGI API 和 stdio MCP 共享同一个注入式公开检索器。默认应用仍是安全不可用
状态，只有公开演示组合入口会注入检索能力。演示没有读取、挂载、索引或返回私有水厂
数据，也不提供数据库写入、设备控制或任意路径参数。

本演示的最终回答是确定性的证据摘录与固定 ABC 结构，并不是 LLM 生成或验证的工程
建议。只有 C（公开知识）有证据时，A（历史）和 B（异常）会明确标为缺失，系统不会
把 C 的证据借给 A/B；所有输出都要求人工复核。

## 环境与首次准备

需要 Python 3.12、[uv](https://docs.astral.sh/uv/) 和可用网络。首次运行会下载
固定提交版本的 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) 与
[BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)。Windows 未启用
开发者模式时，Hugging Face 缓存会因不能创建符号链接而额外占用空间；建议预留至少
6 GB。下载和 CPU 推理时间取决于网络、CPU 和磁盘，首次通常需要数分钟。

在项目根目录执行：

```powershell
uv run --locked python scripts/prepare_public_rag_demo.py
```

成功输出只包含 `ok`、公开分块数、语料哈希、模型 ID 与固定提交哈希。模型已缓存后，
可以禁止联网复用：

```powershell
uv run --locked python scripts/prepare_public_rag_demo.py --offline
```

`--offline` 会在命令生命周期内同时设置 Hugging Face 与 Transformers 的离线标志，
命令结束后恢复调用者原有环境；不会因为依赖解析附属文件而偷偷联网。

本地 Qdrant 使用固定的仓库内公开演示目录，不启动网络服务、不暴露端口；命令结束会
显式关闭句柄。Local Qdrant 对 payload index 会给出无效提示，这是嵌入式运行模式的
已知警告，不会放宽检索结果的 `public/public_read` 二次校验。

## 真实评测与聚合报告

联网首次运行或离线复用固定缓存：

```powershell
uv run --locked python scripts/evaluate_public_rag_demo.py
uv run --locked python scripts/evaluate_public_rag_demo.py --offline
```

该命令比较 BM25、dense、RRF 融合和 RRF + cross-encoder 重排，并在评测后执行一次
Agent/API/MCP 烟雾测试。报告原子写入忽略目录
`artifacts/public-demo/public-demo-report.json`。文件只含聚合指标、模型与数据哈希、
`uv.lock` 哈希和 UTC 时间，不含问题、证据正文、来源 URL、本地模型路径、原始异常或
凭据。

2026-08-12 在当前 Windows 本机、CPU 推理、`k=5` 下测得：

| 方案 | Recall@5 | MRR@5 | 拒答准确率 | 技术拒答率 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.98 | 0.9267 | 1.00 | 0.00 | 1.49 ms |
| Dense | 1.00 | 0.9700 | 1.00 | 0.00 | 199.01 ms |
| Hybrid RRF | 1.00 | 0.9567 | 1.00 | 0.00 | 9.15 ms |
| Hybrid + rerank | 1.00 | 0.9700 | 1.00 | 0.00 | 885.66 ms |

这些是 50 个公开问题与 6 个拒答案例上的本机可复现实验，不是生产 SLA、真实水厂
效果或回答正确率承诺。优化后 CPU rerank P95 仍略高于原 800 ms 建议门槛；演示保留
真实结果，不通过加入私有数据、隐藏拒答或修改评测集来改善数字。实际部署可评估
ONNX/OpenVINO、GPU 或异步服务，但必须重新测量并保留相同安全门。

### Reranker 延迟优化记录

原始配置把 RRF 前 20 个候选全部送入 cross-encoder，正式 P95 为 `2259.37 ms`。
诊断确认延迟随候选数近似线性增长，而公开分块成对输入仅 178–220 tokens，缩短最大
长度没有实际收益。完整 56 例对照中，6、8、12 个候选的 Recall@5、MRR@5、拒答准确率
均与 20 个候选相同。

最终固定 `candidate_limit=24`、`rerank_limit=8`、`answer_limit<=6`：保留两个候选晋升
位，同时将正式 P95 降至 `885.66 ms`，相对原基线下降约 `60.8%`。调用者不能覆盖候选
窗口，报告 schema v3 会记录这两个固定值和实际 reranker 后端。主依赖仍不强制安装
ONNX Runtime、Optimum 或 OpenVINO。经过独立评测和真实烟雾后，项目新增了精确锁定的
`public-openvino` 可选组；Torch 仍是默认后端。

### 可选 OpenVINO 后端

只在需要本机 CPU 低延迟演示时安装可选组：

```powershell
uv sync --extra public-openvino
uv run --locked --extra public-openvino python -m aquaops.public_demo_cli prepare --offline --reranker-backend=openvino
uv run --locked --extra public-openvino python -m aquaops.public_demo_cli smoke --offline --reranker-backend=openvino
```

入口只接受精确的 `torch/openvino`，不接受 ONNX、任意模型 ID、模型目录或设备参数。
OpenVINO 使用仓库忽略目录中固定的公开模型工件；启动时逐文件校验固定集合和 SHA-256，
强制离线、`trust_remote_code=false` 与 CPU，然后用一组公开文本做有限分数自检。缺依赖、
缺工件、哈希不符或自检失败时自动回退到固定 Torch 模型。`prepare` 的
`reranker_backend` 字段会明确给出 requested、active、fallback_used 和安全状态码，
聚合交付报告 v3 也记录同样状态，不会把回退伪装成 OpenVINO 成功。

约 1.13 GB 的 OpenVINO 工件位于 `.gitignore` 覆盖的公开演示目录，不进入 Git；因此
新机器仅安装可选依赖并不会自动拥有该工件。正式迁移时应从固定 revision 在受控环境
重新导出并核对文档中的文件哈希，或通过制品库分发完全相同的目录。缺工件时系统会安全
回退 Torch，不会联网临时导出。

2026-08-12 项目锁定环境真实复验：`prepare` 显示 `active=openvino`、
`fallback_used=false`；完整 smoke 的 Agent、API、MCP 各返回 6 条公开证据，引用集合一致，
ABC 为 3 个维度并要求人工复核。随后在相同 50 个普通问题和 6 个拒答案例上运行完整
`evaluate`，Hybrid + OpenVINO rerank 的 Recall@5 为 1.00、MRR@5 为 0.97、P95 为
`253.24 ms`；相对上表 Torch 端到端基线 `885.66 ms` 降低约 `71.4%`。命令只输出
一行聚合 JSON，生成的 v3 报告哈希为
`cb9619514a4d306d6ae35128ab4da08d660de81d35bcfbcbaa5a06619c2c89d1`。

## 三入口真实烟雾测试

```powershell
uv run --locked python scripts/smoke_public_rag_demo.py --offline
```

验收条件：

- `ok=true`；
- Agent、API、MCP 均获得非空公开证据，且引用 ID 集合一致；
- ABC 固定为三个维度；
- `human_review_required=true`；
- 输出不出现问题、正文、URL、本地路径或原始错误。

本次实测三个入口各返回 6 条公开证据，引用集合一致，所有条件通过。

## 失败处理与清理

- `public_demo_request_invalid`：命令或参数不是固定的 `prepare/evaluate/smoke`、可选
  `--offline` 与可选 `--reranker-backend=torch|openvino`；入口不接受模型 ID、集合名、
  路径、数据类别或报告目标。
- `public_demo_failed`：模型缓存、公开索引、证据协议或烟雾门禁失败。外部输出不会携带
  底层异常；在开发环境通过合成测试和受控本地日志定位。
- `--offline` 首次运行失败：先联网完成固定版本准备，不要改为浮动 revision。
- Qdrant 文件锁：不要并行运行多个演示命令；正常命令会显式关闭资源。异常终止后等待
  上一个进程完全退出再重试。

不需要停止后台服务，因为真实演示使用进程内本地 Qdrant。若要删除可重建的公开演示
索引和聚合报告，可在确认没有演示进程运行后手工删除 `artifacts/public-demo/`；模型
缓存由 Hugging Face 管理，不在仓库中。绝不能用私有数据目录替换该目标。

## 复验门禁

只跑真实演示交付相关合成契约：

```powershell
uv run --locked pytest tests/rag/test_demo_corpus.py tests/rag/test_local_models.py tests/rag/test_demo_runtime.py tests/rag/test_demo_evaluation.py tests/test_demo.py tests/test_demo_delivery.py tests/test_public_demo_cli.py -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

本次最终验证结果：公开演示范围 `582 passed, 1 skipped`；全仓库
`1143 passed, 10 skipped`；Ruff 规则检查、Python 编译、私有耦合扫描、报告泄露扫描
均通过。本次触及的 23 个 Python 文件全部通过格式检查。全仓库 `ruff format --check .`
仍指出 48 个既有文件需要格式化；为了不在脏工作区机械改写无关用户文件，本次没有扩大
修改范围，这是一项独立的代码卫生待办，不影响真实公开演示运行结果。

评测定义、合成回归与真实基线的区别见[公开 RAG 评估说明](public-rag-evaluation.md)。

## Reranker CPU 后端评测

已在公开 50 问、每问固定 8 个相同候选上独立比较 Torch、ONNX Runtime 与
OpenVINO。OpenVINO 在本机保持 Recall@5=1.00、MRR@5=0.97 和 Top-5 排序完全一致，
三次 P95 中位数由 Torch 的 813.535 ms 降至 319.215 ms；ONNX FP32 没有稳定收益。
Torch 仍是默认后端；OpenVINO 已作为显式可选后端接入，并具备离线自检、工件哈希
校验、安全状态和 Torch 自动回退。完整方法、风险与决策见
[公开 RAG Reranker CPU 后端独立评测](public-reranker-backend-benchmark.md)。
