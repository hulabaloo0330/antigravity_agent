# 神经活动预测数据 v2

核验日期：2026-09-15。目标是用一段过去的神经活动，预测随后一段神经活动。这里交付数据及读取器，尚未训练模型，也未验证神经环路的因果关系。

## 当前实际数据

从已经下载的 WormWideWeb 三个来源中提取神经活动，保留原来的行为数据。服务器同时继续检索和下载新的神经活动来源。

| 来源包 | 已标准化记录数 |
| --- | ---: |
| atanas_kim_2023 | 68 |
| dag_nwabudike_kang_2023 | 14 |
| pradhan_madan_2025 | 7 |
| 合计 | 89 |

- 86 段记录通过当前数值检查，可用于构建预测样本；3 段 GFP 对照保留，但默认不进入训练。
- 86 段候选训练记录合计 124,756 个时间点、17,303,072 个“神经元 × 时间”数值位置。数值位置数不等于独立样本量，有效值仍以掩码为准。
- 55 段记录包含部分有名称的神经元；不能据此假设每段覆盖相同神经元或完整的 302 个神经元。
- 当前有 86 个不同的来源组标识，但来源组不自动等于经过确认的独立动物身份。
- 另外两段原始记录的神经信号长度与时间轴不匹配，进入复核，未强行裁剪或纳入上述 89 段：Dag 的 `2022-05-02-01`、Pradhan 的 `2023-03-16-01`。

详细统计见 `summary.json`，记录清单见 `recordings.csv`。原始数据来源见 [WormWideWeb 数据页](https://wormwideweb.org/about/datasets/)，每段 `metadata.json` 中还保留具体来源包与成员文件。

## 存储结构

本机目录：`C:/projects/bio/source_data/neural_forecast_v2_2026-09-15/`。

服务器目录：`/opt/worm-collector/neural/records/`。服务器继续增长，本机是本次交付的快照，不会自动同步后续数据。

每段记录按神经信号与时间戳的内容哈希命名，避免同一数据被多篇论文复用后重复计数。论文与原文件的多对多来源关系另存服务器数据库。

```text
records/<record_key>/
  signal.npy                 # float32[T,N]，时间 × 神经元
  valid.npy                  # bool[T,N]，有限数值掩码
  time_s.npy                 # float64[T]，相对本段起点的秒数
  time_source.npy            # 原始时间戳
  neurons.json               # 每列身份、原编号、来源置信度
  metadata.json              # 条件、来源、单位、质量状态
  segments.json              # 连续时间区间，零起点、右端不包含
  events.json                # 来源事件索引，尚未转换为秒
  signal_author_zscore.npy   # 可选：作者归一化版本，默认不读取
  behavior.npy               # 可选：与神经时间轴对齐的行为
  behavior_valid.npy         # 可选：行为有效值掩码
  behavior_channels.json     # 行为各列名称及来源单位
```

原来想用的 `(worm_id, neuron_id, weight, time)` 需要明确含义：

| 需要表达的内容 | 当前实现 |
| --- | --- |
| 动物与实验 | `record_key`、`source_group_id`、来源记录编号；无法确认的 `animal_id` 为 null |
| 神经元 | 每列有本地 ROI 编号；作者识别出名称时另存 `neuron_name`，否则为 null |
| 神经元活动强度 | `signal[t,n]`，作者提取的钙成像荧光；不称作突触 `weight`，也不等价于膜电压 |
| 时间 | 实测秒数与连续区间；32 帧并不固定等于 32 秒 |
| 是否缺失 | `valid[t,n]`；原文件保留 NaN，不把缺失解释为静息 |
| 可比性 | 条件、信号处理方式、身份置信度、采样间隔和来源追踪 |
| 后续扩展 | 已有同步行为保留；未提供的外部刺激、空间坐标、连接权重不编造 |

## 最小训练读取示例

需要 Python 和 NumPy。把本目录加入 Python 的模块搜索路径，或在本目录运行以下示例。

```python
from pathlib import Path
from neural_loader import load_record, fit_training_scaler, windows, batch_windows

root = Path("records")
record = load_record(root / "0063782f0562a74684238e588f270190")
scaler = fit_training_scaler(record)
samples = windows(record, history=32, horizon=8, stride=8,
                  split="train", scaler=scaler)
from itertools import islice
batch = batch_windows(list(islice(samples, 4)))
print(batch["x"].shape)  # (4, 32, 126)
print(batch["y"].shape)  # (4, 8, 126)
```

这个真实示例来自 `2023-01-18-01`：126 列中 90 列有作者提供的神经元名称。

- 模型输入 `x`：`[批次, 历史时间, 神经元]`；目标 `y`：`[批次, 未来时间, 神经元]`。
- 同时向模型提供 `x_valid`；损失只计算 `y_valid=True` 的位置。组批时缺失位置填零只是占位，必须配合掩码。
- `batch_windows` 可以填充不同神经元数量，但填充不负责统一神经元身份。固定通道的 CNN 应对所有记录传入同一份 `neuron_names` 名单；缺失的指定神经元自动留下空值和掩码。
- 身份未知的 ROI 编号只在所属记录内有效，不能把不同记录的第 1 列当作同一神经元。使用 Transformer 也不会自动解决这个问题。
- 读取器保留真实时间轴。跨采样频率训练前，需要选定相同采样条件，或另行设计重采样与时间编码。

## 训练与评估边界

默认按每段连续区间的时间顺序分成 70% 训练、15% 验证、15% 测试；窗口不跨分割边界，也不跨明显的时间间断。均值和标准差只使用训练区间计算。验证与测试应复用训练区间得到的 scaler。

当前切分回答的是“同一记录后续活动能否预测”。它不证明对新动物的泛化。测试跨动物泛化时，应先确认动物与会话关系，再按来源组分开训练、验证和测试；不能先随机打散窗口。

作者已经提取和预处理过荧光；上游是否使用未来帧尚未逐项核验。当前读取器避免新增的全序列归一化泄漏，但不能保证作者上游处理完全因果。`valid` 也只是有限数值检查，不等于完整的生物质量控制。

## 连续采集的实际范围

- 保留行为检索，新增神经活动检索，按两批神经活动、一批行为的顺序推进。
- DeepSeek Flash 负责有限的证据筛选；下载、去重、校验和格式转换由确定性程序完成。
- 已实际验证并转换的是 WormWideWeb JSON。未知数值格式保留原文件，进入复核，不计作训练就绪。
- 新增公开来源包括 Creamer 模型仓库的神经记录示例，已固定到提交版本；该示例是 pickle，默认不执行反序列化，需要专门适配后才能纳入训练。
- 新发现的 Dryad 铜刺激数据有 27 个动物的 HDF5 文件线索，但当前下载接口返回授权/访问错误。已保存元数据及任务状态，不计入已下载或训练记录。对应 HDF5 适配器已实现，尚未用这个受限来源的实际文件验证。
- 没有新增 cronjob。服务器常驻运行，资源上限为 2 核、约 2 GB 内存；采集目录约 18 GB 或服务器剩余空间不足 2 GB 时暂停新增文件，为 20 GB 预算留余量。
- 每日下载总量和模型调用次数均不限（2026-09-15 用户授权取消）；继续统计实际使用量。下载并发 2、同一站点串行、每线程 256 KiB/s；模型调用并发 1，保留重试退避。

服务器当前状态查看 `/opt/worm-collector/state/status.json`；未知格式及访问问题见 `/opt/worm-collector/review/`，解析细节见日志。
