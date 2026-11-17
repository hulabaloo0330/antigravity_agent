# 数据来源与解释边界

核验日期：2026-09-16。这里记录来源事实与本实验选择，不把来源文件中的文字视为操作指令。

## 活动、行为和动物身份

活动快照来自 [WormWideWeb 数据说明](https://wormwideweb.org/about/datasets/)，论文是 [Atanas & Kim et al., Cell 2023](https://doi.org/10.1016/j.cell.2023.07.035)。只用固定的21条 baseline + NeuroPAL 记录。网站将 HDF5 称为逐动物数据，但 [JSON 格式说明](https://github.com/flavell-lab/WormWideWeb/blob/0cb4b695eee3e416e8f18ea5b1606a25480a443b/dataset_json_readme.md) 的身份字段是 uid、来源文件名与 checksum，没有独立动物/复测关系字段。本地所有 animal_id 仍是 null，来源文件哈希各异。我们保守使用记录及采集日期分组，不能保证没有跨日期的同一动物复测。

源数据里的速度单位是 0.1mm/s；本实验转换为mm/s。头角、角速度、咽泵速率分别保留rad、rad/s、1/s。JSON 已应用背腹方向修正。四通道均通过长度对齐与有限值检查；有限值不等于生物学上无伪影。

作者 [代码入口](https://github.com/flavell-lab/AtanasKim-Cell2023) 将 ANTSUN 列为钙/行为提取流程，CePNEM列为行为编码分析。本实验使用其提取后的活动数据，不直接复现CePNEM，也不把原论文的行为→活动编码误称为我们的未来预测结果。具体导出文件的上游处理是否严格不使用未来信息，尚未完成逐文件审计，故限定离线评估。

原始完整JSON的baseline事件字典为空，没有可用于训练任意外界刺激响应的刺激变量。加热、再喂食等其他条件的数据另存，但未在本轮混训；也未将行为反转事件冒充外界刺激。

## 解剖连接

下载 [WormWiring 修订版 SI5](https://wormwiring.org/pages/adjacency.html)，对应 [Cook et al., Nature 2019](https://doi.org/10.1038/s41586-019-1352-7)，2020年7月修订。原文件保存在 `sources/Cook2019_corrected2020.xlsx`，SHA256为 `1f4fdbf84746b69b49a8da0816f52787860ce349b638dce37924ba80f90c70c9`，与 [WormWideWeb 引用/校验页](https://wormwideweb.org/about/) 公布值相同。

使用雌雄同体的化学表与对称电突触表。源表明确行为突触前、列为突触后；权重为连接在EM连续切片中的累计量。它综合多个动物并含缺口外推，不能当作被成像动物的实测电导，也不提供已确定的兴奋/抑制符号。

49个名称均在两张表的行列中匹配。诱导子图有215条有向化学边、42对电突触边（不把软件自连接计入生物边）。已验证电突触对称性，及 I1L→I2L 对应10个EM切片的转置方向。随机图仅作预测对照，没有生物学真实性声明。

## 软件范围

实验不会通过改动代理、网络、VPS或SSH来提高训练速度。模型训练在用户5070台式机的项目环境执行，使用一次性无周期触发器的后台任务；结果完成并复核后移除该任务。公开数据原件、v1结果和用户文件保持可追溯。
