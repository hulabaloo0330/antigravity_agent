# 环路预测原型：代码与使用入口

此目录包含真实解剖图、日期分组验证、未来神经活动预测、神经到行为读出及串联。最终以 `results/REPORT_完整原型与验证.md` 为准；启动程序不代表实验成功，须检查status及复核文件。

## 数据约定

- 输入是提取后的钙荧光 `[时间,神经元列]`、逐列名称、秒数时间戳。不是膜电位。
- 每条新记录最初160帧只用于神经均值/标准差校准，不更新模型参数，不用未来或行为标注校准。
- 从校准后取32帧历史，预测8帧，训练采样约0.6秒/帧。陌生采样频率/断点须另行验证，不静默接受。
- 神经输出按固定49名称排列，未观测节点的原单位输出是NaN，不冒充完整302神经元预测。
- 行为输出：速度mm/s、头角rad、角速度rad/s、咽泵1/s。线性回归未保证物理范围，不能直接当作湿实验刺激参数。

## Python 调用

自动选第0折的一条真实留出记录。环境与文件由助手部署，用户的学习重点是输入/输出和模型代码。

```python
from pathlib import Path
import sys

experiment = Path(r"C:\projects\worm-circuit\experiments\circuit_forecast_v2")
sys.path.insert(0, str(experiment))
from common import read_json
from predict import Predictor, record_input

manifest = read_json(experiment / "prepared/manifest.json")
record_index = manifest["folds"][0]["test"][0]
record_key = manifest["records"][record_index]["key"]
record_dir = experiment.parents[1] / "source_data/neural_forecast_v2_2026-09-15/records" / record_key

model = Predictor(experiment, fold=0, variant="anatomical", seed=17, device="cuda")
signal, neuron_names, time_s = record_input(record_dir)
prediction = model.forecast(signal, neuron_names, time_s, left=160)

print(prediction["neural_normalized"].shape)  # (8, 49)
print(prediction["behavior"].shape)           # (8, 4)
print(prediction["behavior_units"])
```

评估已有记录时应匹配其留出折，不能拿包含该记录训练的其他折代替。新实验室或新条件的使用仍需独立验证。

## 文件职责

| 文件 | 用途 |
|---|---|
| PROTOCOL.md / config.json | 训练前固定设计和配置 |
| SOURCE_AUDIT.md / sources/ | 数据来源、原始连接表、单位及身份限制 |
| prepare.py / prepared/ | 对齐、初始校准、日期分组、真实/保度随机图及哈希 |
| models.py | 历史与时间编码、两层图注意力、相同参数四变体 |
| baselines.py | 神经线性预测、行为自身历史/神经历史直接预测、行为读出 |
| run.py | 36次训练、内层选择、冻结后外层测试 |
| results/runs/*/best.pt | 各折/变体/种子的最优验证模型 |
| results/fold*/linear_weights.npz | 各折线性模型、读出、训练行为尺度 |
| results/predictions/ | 外层预测，顺序对应prepared/windows.npz |
| predict.py | 加载已训模型的推理接口 |
| verify_inference.py | 留出样本复现与未来信息隔离验证 |
| analyze.py | 独立指标复算、画图和报告 |

## 环境和复现

训练在台式机项目`.venv`：Python3.12.13、torch2.13.0+cu130、NumPy2.5.3、RTX5070。准备连接表需openpyxl。报告用Matplotlib，本机独立依赖放在v1/.analysisdeps；没有修改台式机Anaconda。实际版本/设备/源码/准备数据哈希见results/provenance.json。

重跑使用新的输出目录，勿覆盖原结果或测试后挑选超参数。同设备、同精度和batch可检验精确复现；跨设备可能有小浮点差异。

## 完成范围

公开baseline数据上的两段预测与结构对照属于此次软件实验。任意外界刺激到活动、导师实验室材料迁移、独立动物复测关系核验、严格因果影像处理和湿实验验证，仍需要相应数据与独立实验。未提供材料、未测量连接、未运行验证均不计作完成。
