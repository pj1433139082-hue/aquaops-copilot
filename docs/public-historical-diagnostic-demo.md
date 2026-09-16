# 公开历史入流诊断演示

本演示用于展示 AquaOps 对**公开历史入流监测数据**做的描述性窗口聚合诊断。它不是实时生产系统：
结果不代表当前或最近工况，不能用于自动控制，所有输出均需人工复核。

## 前置条件与数据边界

当前仅允许从登记的 Co-UDlabs 数据源获取本地数据；下载和质量画像依次执行：

~~~powershell
uv run python scripts/acquire_public_source.py co-udlabs-wwtp-lpicm-2025
uv run python scripts/profile_public_source.py co-udlabs-wwtp-lpicm-2025
~~~

公开原始数据、获取收据和质量画像都位于 data/public/ 下的 Git 忽略路径，不会提交进仓库。
不得将私有水厂数据、生产数据或本地私有文件路径加入本演示。数据来源与许可边界见
[public-data-sources.md](public-data-sources.md)。

## 启动与请求

在项目根目录启动本地 API：

~~~powershell
uv run --locked uvicorn aquaops.api.app:create_app --factory --reload
~~~

请保持这个 uvicorn 前台 PowerShell 窗口运行；随后另开一个 Windows PowerShell 窗口，再执行
下面任一请求示例。

PowerShell 示例：

~~~powershell
$body = @{
  mode = "operations"
  question = "公开历史诊断 氨氮 24h 截止 2020-06-11T23:55:00Z"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/agent/runs" `
  -Headers @{ "X-Request-ID" = "public-history-demo" } `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
~~~

curl.exe 的 PowerShell 单行示例：

~~~powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/agent/runs" -H "Content-Type: application/json" -H "X-Request-ID: public-history-demo" -d '{"mode":"operations","question":"公开历史诊断 氨氮 24h 截止 2020-06-11T23:55:00Z"}'
~~~

演示只返回窗口聚合统计和来源证据，不展示 CSV 原始行或原始监测值。

## 公开历史异常筛查请求

下列请求在同一受控公开数据源上，对**截止时刻之前**的 24 小时历史基线做单指标异常候选筛查：

~~~powershell
$body = @{
  mode = "operations"
  question = "公开异常筛查 氨氮 24h 截止 2020-06-11T23:55:00Z"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/agent/runs" `
  -Headers @{ "X-Request-ID" = "public-anomaly-demo" } `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
~~~

目标时刻的观测不会进入自己的基线，因此结果只能表述为“公开历史中的异常候选”。稳健得分阈值 3.5 是
公开演示的初始筛查阈值，不是工艺报警阈值；任何候选都必须由人工结合数据质量、维护记录和现场工况复核。
实现的方法选择、回退条件和工程约束见
[public-anomaly-method-selection.md](public-anomaly-method-selection.md)。响应不会返回目标原始值、原始行、完整数组或文件路径。

## 响应与安全含义

成功响应中的字段含义如下：

- **answer**：公开历史窗口的聚合统计或异常筛查结论，并固定说明“非实时、不可用于自动控制”。
- **evidence**：唯一来源证据，含 chunk_id、Co-UDlabs Zenodo source_url 和
  source_version；当前登记来源为
  https://zenodo.org/records/15285089，版本为 v1.0.0 (2025-04-26)。
- **requires_human_review**：始终为 true，提醒操作人员人工复核。
- **request_id**：调用方的追踪标识；**run_id**：服务器生成的本次结果标识，可用于后续查询。

即使返回成功，也只能据此描述该公开历史窗口；它不能替代在线监测、报警、加药、曝气或其他生产
控制决策。

## 允许的显式请求

请求必须完整匹配下列格式，其中小时数为 1..168，截止时间为带 Z 的 UTC 时间：

~~~text
公开历史诊断 <指标> <小时数>h 截止 <YYYY-MM-DDTHH:MM:SSZ>
~~~

异常筛查必须完整匹配下列格式，并使用相同的指标、小时数和 UTC 时间边界：

~~~text
公开异常筛查 <指标> <小时数>h 截止 <YYYY-MM-DDTHH:MM:SSZ>
~~~

指标映射为：

- 氨氮 或 nh4 → nh4（mg/L）
- 电导 或 cond → cond（µS/cm）
- 流量 或 q → q（L/s）

## 会被安全拒答的请求

以下类型不会被当作公开历史诊断执行，而会安全拒答：

- “当前氨氮是多少？”、“最近是否异常？”等实时或最近工况请求；
- 加药、曝气、阀门、泵等控制命令；
- 传入或要求读取私有水厂路径、私有 CSV、SCADA 或生产数据的请求；
- 缺少指标、小时数或 UTC 截止时间，或超出 1–168 小时边界的请求。

## 面试讲解要点

可以这样说明该能力的工程取舍：

- 我先限定数据许可范围，并为获准公开 CSV 登记 SHA-256；每次读取先校验同一份字节，再解析。
- 获取阶段只提取白名单路径，并防护路径穿越与符号链接风险，避免把压缩包内未获准内容带入项目。
- 工具边界采用严格证据 schema：只允许来源、版本和聚合窗口标识，避免泄露原始行、文件路径或私有数据。
- LangGraph 的 operations 模式只接受完整、受限的公开历史请求；其他模式、异常结果或渲染失败都会回落到安全拒答。
- 输出固定要求人工复核，并明确不是实时系统、不是生产控制能力；不把本地演示夸大成线上性能或自动化运维成果。
