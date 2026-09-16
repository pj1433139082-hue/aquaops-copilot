# 公共数据源登记与使用边界

本表是 AquaOps 公共数据阶段的来源登记。`data/public/sources.json` 是机器可读的
权威清单；本文件说明首轮用途及数据处理边界。

| Source ID | 角色 | 来源 | 初始用途 | 许可证状态 | 提交边界 |
| --- | --- | --- | --- | --- | --- |
| `co-udlabs-wwtp-lpicm-2025` | 数据集 | [Co-UDlabs LPICM（Zenodo）](https://zenodo.org/records/15285089) | 本地探索与后续受控摄取 | 已验证：`WWTP_Langmatt/**` 为 CC0 1.0 | **唯一获准本地获取的来源**；仅限 `WWTP_Langmatt/**`；原始文件不得提交。 |
| `iwa-benchmark-simulation-models` | 仿真模型 | [IWA Benchmark Simulation Models](https://github.com/wwtmodels/Benchmark-Simulation-Models) | 识别可用于基准对比的模型与接口 | 待审查 | 仅参考；不得下载、索引或提交。 |
| `dhi-waterbench-inflow` | 数据集 | [WaterBench — Inflow to a Wastewater Treatment Plant](https://github.com/DHI/WaterBench-TimeSeries-WWTPINflow) | 评估入流水序列的公开基准候选项 | 待审查：仅为搜索/仓库中观察到的 CC-BY-NC-4.0 提示，非许可证确认 | 仅参考；许可证未经直接核验前不得下载、索引或提交。 |
| `mee-gb-18918-2025-amendment` | 知识文档 | [GB 18918—2002（含修订）](https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/shjbh/swrwpfbz/200307/W020260206765128897192.pdf) | 标注排放标准与修订版本的参考边界 | 待审查 | 仅参考；不得下载、索引或提交。 |
| `epa-nutrient-control-design-manual` | 知识文档 | [EPA Nutrient Control Design Manual](https://www.epa.gov/sites/default/files/2019-02/documents/nutrient-control-design-manual.pdf) | 营养盐控制设计知识的参考边界 | 待审查 | 仅参考；不得下载、索引或提交。 |

除 Co-UDlabs 外，所有 `reference_only` 资源在许可证得到确认前一律不得下载，
也不得建立索引。任何来源的原始数据都不得提交到仓库；本地获取的数据只能放在
被忽略的 `data/public/<source_id>/...` 中。

对 Co-UDlabs 的本地临时检查已证实压缩包内部许可证证据：`common/license.txt` 与
`WWTP_Langmatt/documentation/README_WWTP_Langmatt.md` 均将 WWTP Langmatt 标为
CC0 1.0。因此只允许摄取 `WWTP_Langmatt/**`。`Rainfall_Lenzburg/**` 另为 CC BY，
明确排除且不得提取。此结论只适用于上述已核验的路径，严禁扩展到压缩包内的其他路径。

正式归因：Rieckermann, J. (2025): Electrical conductivity and ammonium monitoring
data from WWTP Langmatt inlet, Lenzburg (Version 1) [Data set]. Zenodo. DOI:
10.5281/zenodo.15285089。

为保证公开历史诊断所依据的内容未被本地替换，登记表还保存
`WWTP_Langmatt/data/WWTP_Langmatt.csv` 的文件级 SHA-256。每次只读窗口汇总都会先将
同一份读取到的字节与该登记哈希比对，校验通过后才以 `utf-8-sig` 解码并聚合；哈希不符、
缺失或登记校验失败时一律拒绝输出统计结果。该机制只校验获准公开 CSV，不会读取、输出或
推断任何私有水厂数据。

`WWTP_Langmatt/documentation/README_WWTP_Langmatt.md` 还将该来源的 `date_vec`
说明为 ISO UTC 时间。因此在且仅在这个已登记、哈希校验通过的 Co-UDlabs CSV 中，形如
`YYYY-MM-DD HH:MM:SS` 或 `YYYY-MM-DDTHH:MM:SS` 的无时区观测时间按 UTC 解释；带
偏移的观测时间归一为 UTC。接口请求的 `end_at` 仍必须是严格的 canonical UTC `Z` 格式。
这项解释不适用于其他公开来源，更不适用于任何私有数据。
