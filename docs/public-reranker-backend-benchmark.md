# 公开 RAG Reranker CPU 后端独立评测

评测日期：2026-08-12
结论：**当前机器已将 OpenVINO 接为显式可选后端；不接入 ONNX FP32；Torch 继续作为默认与回退后端。**

## 评测目的

在不改变公开语料、召回候选和模型权重的前提下，单独比较 BGE Reranker 在
PyTorch、ONNX Runtime 和 OpenVINO 三种 CPU 后端上的排序质量与延迟。评测只回答
“哪个推理后端适合这台演示机”，不改变现有公开演示的 Torch 基线。

## 固定条件

- 机器：AMD Ryzen 9 7945HX，16 核 32 线程，CPU 推理。
- 模型：`BAAI/bge-reranker-base`。
- 模型 revision：`2cfc18c9415c912f9d8155881c133215df768a70`。
- 数据：公开演示评测集中的 50 个普通问题；6 个策略拒答问题不进入模型评测。
- 输入：生产公开 RRF 为每个问题生成完全相同的 8 个候选，三种后端只负责重排。
- 计时：每个后端先执行一个 8 对文本的预热批次，再独立运行 50 次重排；完整流程重复
  3 次，以三次 P95 的中位数作为主要决策指标。
- 质量门禁：Recall@5、MRR@5、Top-1 一致率、Top-5 集合与顺序一致率。
- 隔离：使用 `artifacts/public-demo/backend-benchmark/.venv` 独立环境，不修改项目
  `pyproject.toml` 或 `uv.lock`，不读取私有水厂数据。

公平比较环境版本：SentenceTransformers 5.6.1、Torch 2.13.0 CPU、Transformers
4.57.6；ONNX Runtime 1.28.0；OpenVINO 2026.3.0。生产环境仍使用原有依赖和 Torch
后端，因此下表是同一隔离环境中的横向比较，不把它冒充生产端到端延迟。

## 结果

| 后端 | 三次 P95（ms） | P95 中位数 | P50 中位数 | 初始化中位数 | Recall@5 | MRR@5 | 稳态 RSS 约值 | 工件大小 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Torch FP32 | 824.546 / 809.014 / 813.535 | 813.535 | 782.142 | 1,075.705 ms | 1.00 | 0.97 | 1.14 GB | 1.129 GB |
| ONNX FP32 | 786.366 / 833.821 / 821.214 | 821.214 | 767.214 | 3,449.955 ms | 1.00 | 0.97 | 2.01 GB | 1.129 GB |
| OpenVINO FP32 | 323.655 / 319.215 / 314.368 | **319.215** | **230.303** | 3,404.375 ms | 1.00 | 0.97 | 2.13 GB | 1.130 GB |

以三次 P95 中位数计算：

- OpenVINO 相对 Torch 加速约 **2.55 倍**，P95 降低约 **60.8%**。
- ONNX 相对 Torch 没有稳定收益，P95 中位数反而约高 **0.9%**。
- 三者 Recall@5 与 MRR@5 完全一致。
- ONNX 与 Torch 的完整 8 项排序一致。
- OpenVINO 与 Torch 的 Top-1、Top-5 集合和 Top-5 顺序均为 100% 一致；全部 8 项
  的平均绝对名次差为 0.01，只发生在 Top-5 之外，不影响当前回答证据。

## 决策

1. **不接入 ONNX FP32。** 当前硬件上没有稳定尾延迟收益，同时初始化和内存占用更高。
2. **OpenVINO 已进入显式可选路径。** 它达到了质量不回退和 P95 明显下降两个门禁。
3. **不直接替换 Torch 默认值。** OpenVINO 隔离环境的稳态 RSS 约增加 1 GB，初始化约增加
   2.3 秒；Windows 下 Optimum Intel 加载已保存模型时还输出路径兼容警告，虽未阻止三次
   离线运行，但需要在正式接入前通过固定依赖、启动烟雾和回退测试消除风险。
4. 已实现固定 `torch/openvino` 白名单、逐文件哈希、启动离线自检、失败自动回退 Torch、
   聚合状态审计和精确锁定的可选依赖；不能用自动下载或浮动 revision 作为演示依赖。

## 可复现工件

评测脚本、固定输入、各后端模型和逐次结果保存在忽略目录：

`artifacts/public-demo/backend-benchmark/`

其中：

- `prepare_inputs.py`：从公开评测集和生产 RRF 固化 50×8 相同输入。
- `run_backend.py`：强制离线、固定模型 revision、CPU、单后端独立进程评测。
- `compare_rankings.py`：比较完整排序、Top-1 与 Top-5 一致性。
- `summary.json`：不含问题正文、URL、本地私有路径或密钥的聚合结论。

该目录被 `.gitignore` 排除，约 3.4 GB 的模型副本不会进入版本库。复验前必须确认
输入仍标记为 `data_class=public`，不得把私有目录或私有指标映射给这些脚本。

## 官方依据与解释边界

SentenceTransformers 官方建议在实际模型、数据和硬件上分别基准测试各后端，并明确指出
ONNX/OpenVINO 不保证对每个模型都更快；这正是本次没有凭后端名称直接做技术选型的原因。
官方对 CPU 的一般建议偏向 Intel 使用 OpenVINO、其他 CPU 使用 ONNX，但本机是 AMD，
所以本报告只依据这台 AMD 演示机上的实测，不把结论外推到其他部署机器。

- SentenceTransformers CrossEncoder 推理效率：<https://sbert.net/docs/cross_encoder/usage/efficiency.html>
- SentenceTransformers CrossEncoder API：<https://www.sbert.net/docs/package_reference/cross_encoder/model.html>
- BGE Reranker 模型：<https://huggingface.co/BAAI/bge-reranker-base>
