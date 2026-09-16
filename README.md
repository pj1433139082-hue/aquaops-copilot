# AquaOps Copilot

## Public portfolio release boundary

AquaOps is a decision-support portfolio project built from public EPA-derived summaries and
synthetic test data. It does not connect to, control, or claim validated performance at any real
water plant. Retrieval and Agent outputs retain source/version evidence for human review; they are
not automatic operating instructions. Known limitations include a curated public demo corpus,
a single-worker demonstration profile, and no production availability or security certification.
Compose requires locally supplied PostgreSQL and JWT secrets; no credential default is stored in
the repository.

The AquaOps source is offered under the Apache-2.0 license ([license text](LICENSE)). Public source
rights, attribution, and versions are recorded in the registered corpus metadata and the [public RAG
evaluation notes](docs/public-rag-evaluation.md). Dependency and source notices are listed in the
[public license notices](docs/public-license-notices.md); security reporting and safe disclosure
guidance are in [SECURITY.md](SECURITY.md). Release evidence and mandatory privacy checks are listed
in the [release checklist](docs/release-checklist.md).

## 公开演示（真实输出）

下面两张图来自公开合成语料的离线 smoke/API 运行，只展示聚合字段和安全边界，不展示本机
路径、模型路径、原始证据正文或任何私有水厂数据。图片用于快速了解结果，README 中的命令
和报告哈希才是可复现事实来源。

![公开 RAG 三入口 smoke 输出](docs/assets/public-rag-smoke.svg)

![公开 Agent API ABC 响应契约](docs/assets/public-api-abc.svg)

## 60 秒公开路径

如果只想验证公开边界和最小检索闭环，不需要启动 Docker，也不需要访问任何水厂文件：

```powershell
Copy-Item .env.example .env
uv run --locked python -m pytest tests\rag\test_public_corpus.py tests\rag\test_hybrid.py tests\rag\test_evaluation.py -q
```

要运行真实公开语料演示，第一次准备需要联网下载固定 revision 的公开模型；模型缓存就绪后，
评估和三入口烟雾测试可以离线复用：

```powershell
uv run --locked python scripts/prepare_public_rag_demo.py
uv run --locked python scripts/evaluate_public_rag_demo.py --offline
uv run --locked python scripts/smoke_public_rag_demo.py --offline
```

如果把公开模型缓存放在默认 Hugging Face 目录之外，请在运行前把 `HF_HOME` 和
`HF_HUB_CACHE` 指向外置的公开缓存目录；缓存不应进入 Git、staging、Docker build context
或截图。`--offline` 会在命令生命周期内禁止模型联网探测。

命令只输出聚合结果或安全摘要。你应能看到 `ok`、公开分块/案例数量、语料与评估哈希、
`abc_dimension_count=3`、`human_review_required=true` 和三入口引用集合一致等字段；它不会
回显查询正文、原始证据、本机模型路径或异常堆栈。输出字段和失败处理见
[公开架构说明](docs/public-architecture.md) 与[真实演示说明](docs/public-rag-demo.md)。

## 本地公开演示：前置条件与数据边界

本项目的演示边界是**公开资料或合成数据**。不要把私有水厂数据、生产工单、
架构细节、监测原始记录或令牌提交、挂载或导入本仓库。Qdrant 仅在 Compose
内部网络可达；它不发布未经认证的宿主机端口。当前公开服务没有写入、删除、控制或私有数据工具；私有数据处理不属于公开演示范围，边界见[公开架构说明](docs/public-architecture.md)。

需要 Python 3.12、已同步的本地 `.venv`（或 uv）以及 Docker Compose（仅在
启动 API/Qdrant 时需要）。开始前，从安全模板创建本地环境文件：

```sh
cp .env.example .env
```

On PowerShell, use:

```powershell
Copy-Item .env.example .env
```

若确实需要外部搜索，在本机 `.env` 中填写真实 `AQUAOPS_SEARCH_API_KEY`；绝不
把真实密钥或 `.env` 提交到 Git。合成测试与公开 RAG 回归不需要该密钥。
启动 Compose 前还必须在本机 `.env` 填写随机的 `AQUAOPS_POSTGRES_PASSWORD` 和
至少 32 字符的 `AQUAOPS_JWT_SECRET`。

## 仅公开 RAG 的合成回归

下面的回归只使用固定的合成公开评估集、假检索器和内存对象；不会下载模型、
启动 Docker/Qdrant、访问网络，也不会读取本地水厂文件：

```powershell
.venv\Scripts\python.exe -m pytest tests\rag\test_public_corpus.py tests\rag\test_hybrid.py tests\rag\test_evaluation.py -q
```

指标定义、拒答规则、数据集版本锁定及其“非生产承诺”见
[公开 RAG 评估说明](docs/public-rag-evaluation.md)。

## 真实公开 RAG 演示

仓库现已提供固定版本的真实公开演示：8 个 EPA 官方来源、24 个公开中文摘要分块、
BGE-M3、进程内本地 Qdrant、BM25、RRF、BGE reranker，以及共享检索器的
Agent/API/MCP 三入口。首次联网准备后可完全离线复用：

```powershell
uv run --locked python scripts/prepare_public_rag_demo.py
uv run --locked python scripts/evaluate_public_rag_demo.py --offline
uv run --locked python scripts/smoke_public_rag_demo.py --offline
```

真实命令、固定模型 revision、实测四路指标、聚合报告、失败处理和清理方式见
[公开 RAG 真实演示](docs/public-rag-demo.md)。演示只索引注册的
`public/public_read` 语料；不得指向远程/生产 Qdrant，也不得加入任何私有水厂数据。

完整的文字架构、入口共用关系、ABC 覆盖门禁、预测输出边界和安全降级策略见
[公开架构说明](docs/public-architecture.md)。

## 启动本地 API 与内部 Qdrant

首次构建或依赖变动时可执行：

```sh
docker compose up --build -d
```

已有镜像时，启动同一组本地服务：

```sh
docker compose up -d
```

Once the services are running, check the API:

```sh
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Stop the local services:

```sh
docker compose down
```

To also remove any named volumes and orphaned containers created by the stack:

```sh
docker compose down --volumes --remove-orphans
```

### 可选模型 worker

默认 Compose 闭环启动 PostgreSQL、Redis、Qdrant、迁移与 API，不下载模型。需要测试
公开资料异步摄取时，显式启用 `model-worker` profile：

```sh
docker compose --profile model-worker up --build -d
```

worker 镜像使用独立带哈希的依赖锁文件；BGE-M3 仍绑定代码中的固定 revision。
首次启用需要联网下载公开模型，之后复用 `huggingface-model-cache` 命名卷。该 profile
只接受仓库注册且哈希校验通过的公开语料，不得挂载私有数据或主机模型目录。

## 本地 stdio MCP

公开 MCP 不会作为 Compose 服务或监听端口启动。需要 MCP 客户端连接时，直接以
stdio 运行与锁定依赖一致的入口：

```powershell
uv run --locked python -m aquaops.mcp.public_server
```

它只提供 `retrieve_public_water_knowledge`、`aggregate_public_history` 与
`screen_public_anomaly` 三个只读工具。原始 JSON 入参、固定错误、来源 URL 限制
及威胁模型见 [公开 MCP 说明](docs/public-mcp.md)。

## 证据与人工复核

检索结果会保留 `chunk_id`、`source_url`、`source_version` 和分数；它们用于追溯
公开证据，不是自动控制依据。任何运维判断都应由人员复核原始公开来源、时间范围和
适用条件。当前实现也不防御能在同一 Python 进程内任意执行代码的恶意插件；未来的
高风险写操作必须放进独立、最小权限并带人工审批的服务，不能复用本进程能力对象。

## Agent-run development contract

`POST /v1/agent/runs` runs the selected safety graph synchronously and returns
the completed, citation-gated result with HTTP 200. The server generates an
unguessable `run_id`; use `GET /v1/agent/runs/{run_id}` to retrieve that same
result. `X-Request-ID` is trace-only and never selects a stored result.

Results live only in a thread-safe in-process store for single-process local
development. It retains at most 100 completed results and evicts the oldest
when full. Results are lost on restart, are not shared by multiple workers,
and this is not a reliable task queue. A later backend implementation must
replace it with durable execution and result storage.

## Image digest provenance

The following tag-and-digest references were verified on 2026-07-19 using the
official Docker Hub Registry v2 manifest metadata for Qdrant and Python, and
the official GitHub Container Registry v2 manifest metadata for uv. The tag is
included for readability; the digest fixes the image content used by builds.

- `qdrant/qdrant:v1.18.2@sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c`
- `python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de`
- `ghcr.io/astral-sh/uv:0.5.30@sha256:bb74263127d6451222fe7f71b330edfb189ab1c98d7898df2401fbf4f272d9b9`

## Optional local Qdrant contract test

The Qdrant contract test never starts Docker and is skipped unless both of the
following flags are set. It permits a write only to an already-running local
Qdrant endpoint at `localhost` or `127.0.0.1`.

```powershell
$env:AQUAOPS_RUN_QDRANT_CONTRACT = "1"
$env:AQUAOPS_QDRANT_CONTRACT_ALLOW_WRITE = "1"
.venv\Scripts\python.exe -m pytest tests/integration/test_qdrant_contract.py -q
```

Each run creates a UUID-derived temporary collection, topic, and point using
only synthetic public data, then deletes that collection in `finally`. It never
uses the fixed production public collection; do not enable it for non-local or
production Qdrant URLs.
