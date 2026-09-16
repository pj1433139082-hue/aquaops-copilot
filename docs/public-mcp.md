# AquaOps public MCP（仅本地 stdio）

`aquaops.mcp.public_server` 是 AquaOps 的**公开资料只读边界**。它不是
Web API，也不连接或代理任何私有水厂数据。

## 本地启动

仅在本机需要 MCP 客户端连接时运行：

```powershell
uv run --locked python -m aquaops.mcp.public_server
```

入口固定使用 stdio；没有 HTTP、SSE 或 Streamable HTTP 传输，没有监听端口，
也没有 `uvicorn`。不要把此命令作为公网服务启动。

## 工具清单

服务只注册下列三个工具，工具名称由代码中的 `PUBLIC_MCP_TOOL_NAMES` 固定：

| 工具 | 输入 | 安全输出 |
| --- | --- | --- |
| `retrieve_public_water_knowledge` | `query`（非空、最多 512 字符）、`limit`（1–6） | `chunk_id`、`source_url`、`source_version`、`score` |
| `aggregate_public_history` | `indicator`（`nh4`/`cond`/`q`）、`hours`（1–168）、`end_at`（规范 UTC `Z`） | 经过既有公开历史契约校验的窗口聚合 |
| `screen_public_anomaly` | `indicator`（`nh4`/`cond`/`q`）、`baseline_hours`（1–168）、`end_at`（规范 UTC `Z`） | 经过既有公开异常契约校验的聚合筛查结果 |

没有 write、delete、admin、SQL、路径、文件或私有数据工具。未配置检索依赖时，
检索工具受控返回不可用；它不会自行创建 Qdrant 客户端或下载嵌入/重排模型。
数值输入在 MCP 边界保留为原始 JSON 值，并在每个工具的第一层以
`type(value) is int` 和范围校验处理。这样 `true`、`"1"` 等不会被 SDK 强转为
`1`，而会稳定返回 `invalid_request`，同时不会将 SDK 的验证异常交给客户端。

同样，`query`、`indicator` 与 `end_at` 以原始 JSON 值进入工具，并要求
`type(value) is str`、非空、长度和格式都满足约束。对象、列表、布尔值与缺失字段
统一返回 `invalid_request`；不会将 Pydantic、ValidationError、SDK 错误或原始输入
回显给客户端。

为同时保留固定 typed error（而不是由 FastMCP 在函数前抛出验证文本）与可审计的输入
规范，代码将不可变的 `PUBLIC_TOOL_INPUT_CONTRACTS` 作为规范注册表。它精确列出三个工具
的参数名、JSON 类型、必填性、整数边界、枚举、默认值及 UTC 格式；工具描述也重复该约束。
Python 进程内的集成方可以导入该常量，且业务运行时不能修改它。远程或 stdio MCP 客户端
不能通过 MCP 自动读取该模块常量，应依据每个工具的 description 构造请求；无论客户端如何
构造输入，运行时仍会以 fail-closed 校验返回固定 typed error。

## 数据与传输边界

- 仅允许经调用方显式注入的公开、`public_read` 检索能力；MCP 默认不配置检索器。
- 历史聚合与异常筛查只复用既有的固定公开历史函数，并在 MCP 输出前再次做严格
  类型与业务语义校验。
- 响应采用结构化白名单。原文 `text`、逐行观测、`values`、`rows`、向量、本地
  路径、SQL、命令、令牌和密钥均不会被序列化。检索来源 URL 必须为无用户凭据、
  无查询参数和无片段的 HTTPS URL；来源版本仅允许受控标识符字符，并拒绝
  `token`、`key`、`secret`、`credential` 等敏感标记。
- 上述 URL 与版本标记规则只适用于**检索证据**。公开历史聚合和公开异常筛查的
  `source_version` 是既有固定公开源的受控 literal（当前为
  `v1.0.0 (2025-04-26)`），由其既有严格模型校验后原样返回；它不走检索版本标识符
  规则。
- 检索来源不做 DNS 解析。只接受格式受限的公开 DNS 名称；所有 IP literal，以及
  `localhost`、`.localhost`、`.local`、`.internal`、`.lan`、`.corp`、
  `.home.arpa` 等本地/内部名称都会被拒绝。
- 不存在私有 MCP 端点。私有水厂数据应在独立、本地受控工作流中聚合后再决定是否
  形成可共享结果，绝不能接入此服务器。

## Agent 授权与威胁模型

operations 图对公开只读工具采用不可变注册表、主机创建的预算会话、一次性执行收据和
按工具来源绑定的公开证据能力。它们的目标是防止模型输出、JSON/MCP 参数、请求状态或
Pydantic 复制对象伪造调用额度、工具来源或 A/B/C 证据归属；它们不是 Python 进程内的
权限隔离机制。尤其是，以下划线命名的符号只是实现约定，不能抵御可在同一进程任意导入、
反射或执行代码的恶意插件。

当前服务器只有公开只读能力，且不执行自动控制。未来如增加任何高风险写工具，必须将
执行放在独立受控进程或审批服务中，并采用独立身份、最小权限、明确人工审批和可撤销的
审计边界；不得把当前进程内能力对象或私有 Python 符号作为写操作的安全依据。

## 错误格式

无论参数错误、能力未配置、依赖故障或不安全的依赖结果，工具均返回固定 JSON
形状，不把异常、查询全文、路径、令牌或堆栈交给 MCP 客户端：

```json
{
  "ok": false,
  "error": {
    "code": "invalid_request",
    "message": "Request does not satisfy the public read-only contract."
  }
}
```

其他代码为 `capability_unavailable` 和 `unsafe_public_result`；消息同样是固定的。

## 合成测试

测试使用注入的假检索器和合成公开汇总对象，不读取真实水厂数据、不启动 Docker、
Qdrant、模型或网络监听：

```powershell
uv run --locked python -m pytest tests/mcp/test_public_server.py tests/agent/test_tools.py -q
```

在测试前确保依赖已同步（`mcp` 的解析版本由 `uv.lock` 固定）。
