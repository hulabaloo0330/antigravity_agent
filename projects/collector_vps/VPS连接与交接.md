# VPS 采集任务交接（神经活动 v2 已运行）

最后核验：2026-09-15 23:25:10（香港时间）。神经活动v2和共享停止条件的常驻整理均已运行。下方早期数据数字为历史快照，最新运行结构与统计见文末“常驻整理更新”；恢复上下文先读 memory/PROJECT_STATE.md。

## 当前运行方式

- SSH：root@72.11.133.206，端口 18788。本机使用现有密钥 C:/Users/xusiy/.ssh/id_rsa_racknerd；私钥没有上传。
- 常驻服务：worm-collector.service；active/running，开机自动启动。
- 根目录：/opt/worm-collector。程序 2.0.1，独立 Python 执行器调用 deepseek-flash。
- 连续检索、筛选、下载、转换，每批结束后等待 15 秒继续。没有创建 cronjob 或 Codex 自动任务。
- 保留行为采集，新增神经活动采集；检索按两批 neural、一批 behavior 推进，分别保存游标。
- 凭据仅存 VPS /etc/worm-collector/deepseek.key，未回传本机。不得在日志、交接或聊天中输出凭据。
- 服务使用 /usr/bin/python3.8。默认 /usr/bin/python3 实际指向 root 私有 pyenv，独立服务账号不能使用。
- NumPy 1.24.4 与 h5py 3.11.0 放在 /opt/worm-collector/vendor，仅此服务通过 PYTHONPATH 使用。

## 本次已完成的实际数据

- 原有五篇种子来源的 10 个文件均已下载并校验，原文件 366,243,877 字节。
- 三个 WormWideWeb 来源中的神经数据已转换为 89 段标准记录，其中 86 段可用于构建训练样本，3 段 GFP 对照默认排除。
- 86 段候选训练记录共 124,756 个时间点、17,303,072 个神经元×时间数值位置；缺失值以掩码为准。
- 55 段记录有部分命名神经元。未知 ROI 不当作跨个体统一身份，来源组不直接当作已确认的动物身份。
- Dag 的 2022-05-02-01 和 Pradhan 的 2023-03-16-01 因神经信号与时间长度不符进入复核，原文件保留。
- 新增 Creamer 仓库 example_recording.pkl 已实际下载并校验：9,118,164 字节；提交 bba43302d50a4947804d98b01779856e648237cc。默认不执行 pickle 反序列化，尚未计入训练记录。
- 当前已验证原文件总计 11 个、375,362,041 字节。后续会继续变化。
- 新发现 Dryad 铜刺激数据的 27 个动物 HDF5 及标签、说明共 29 个文件线索；下载返回 401/403，已保留元数据并进入 review，不能声称已下载。HDF5 适配器尚未用这个来源的实物验证。
- 未训练模型，也未证明跨动物预测或环路因果推断有效。

## 训练存储结构

服务器 /opt/worm-collector/neural/records/<record_key>/：

- signal.npy：float32[时间,神经元]，作者提取的钙成像荧光，不是膜电压或突触权重。
- valid.npy：有限数值掩码；time_s.npy 与 time_source.npy 保存相对及原始时间。
- neurons.json：每列 ROI、可选神经元名称、来源置信度。
- metadata.json：条件、来源、质量、身份不确定性及内容哈希。
- segments.json：连续时间区间，窗口不跨明显时间间断。
- 可选 behavior.npy 与行为通道、掩码；可选作者归一化信号独立保存，默认读取器不使用。
- app/neural_loader.py：按过去 32 帧→未来 8 帧读取，默认在每个连续区间按时间 70/15/15 切分；归一化只拟合训练区间。
- 可生成 [批次,时间,神经元] 并携带缺失掩码和身份。不同记录的通道对齐需要明确的同名神经元名单；填充本身不代表身份对齐。

本机已交付 C:/projects/bio/source_data/neural_forecast_v2_2026-09-15/，包含同批 89 段数据、recordings.csv、summary.json、neural_loader.py 与 README_训练数据说明.md。本机是交付快照，不会自动同步 VPS 后续文件。

## 资源与现有代理

- cgroup CPU 配额最多 2 核，MemoryLimit=2,000,000,000 字节；核验时内存约 77.6 MB。
- 两个下载线程，同一站点串行；每线程 256 KiB/s，总下载上限 512 KiB/s。数值解析串行，数值库线程数为 1。
- 数据目录约 18 GB 或服务器可用空间不足 2 GB 时暂停新增文件，给 20 GB 预算留余量；这是应用预算，并非独立文件系统硬配额。
- daily_transfer_max_bytes=null、daily_llm_calls_max=null：每日下载总量和模型调用次数上限均已取消（2026-09-15 用户授权）。实际传输量与模型请求仍按 UTC 日期计数；单模型调用并发及重试退避保持生效。
- 单文件最大 512 MiB，不自动运行论文仓库代码，不自动下载原始视频。未知格式保留并进入 review。
- xray、hysteria、wireproxy、nginx 均 active，本次未修改或重启代理服务，未修改 SSH、端口、防火墙或路由。
- 原有 OpenClaw、其已有 8 个定时任务和聊天路由保持原状。

## 目录和检查

- state/catalog.sqlite：论文、数据集、文件、任务、神经记录、神经元、来源关系。
- state/status.json：最近发布的统计；长任务期间以数据库和 logs/events.jsonl 更实时。
- state/control.json：运行开关，enabled=true。
- papers/：论文与证据；datasets/：仓库清单及回执；objects/sha256/：原文件内容去重。
- neural/records/：训练结构；review/：访问受限、证据不足、未知格式等。
- exports/current.json：最近 CSV 快照位置，保留最近 7 份。
- config/policy.json、config/AGENT_PROMPT.md、config/decision.schema.json：实际生效规则。

仅操作本专用服务：

```sh
systemctl status worm-collector.service --no-pager
cat /opt/worm-collector/state/status.json
tail -n 20 /opt/worm-collector/logs/events.jsonl
systemctl stop worm-collector.service
systemctl start worm-collector.service
```

关闭 SSH 或本地对话不会停止采集。

## 实现及修复记录

- 本机当前程序 C:/projects/bio/collector_runtime/；服务源码以 /opt/worm-collector/app/ 为准。
- 部署材料 /root/bio-collector-inbox/20260915/runtime/，原始规范包在同级 collector_spec_v1/。旧 cron 设计和旧 v1 deploy.py 不能代表当前部署方式，不要直接重跑旧安装脚本。
- upgrade_neural.py 是已执行过的 v1→v2 一次迁移，有旧版哈希保护，不能重跑。
- 已修复 PubMed ArticleId 提取范围：仅读取当前论文的 PubmedData/ArticleIdList，避免误读参考文献 DOI/PMCID。
- 49 条已有 PubMed 元数据已按当前论文重新核验并重新排队筛选，检索游标重置回填；五篇已验证种子的原文件和关联未受影响。
- 修复前数据库、代码与论文证据备份位于 runtime/before-pubmed-repair-2026-09-15T081711Z/。一次性修复脚本 repair_pubmed_v2.py 不要重复执行。
- 测试已通过：来源去重、任务独占、证据约束、资源预算、真实状态、形状与缺失掩码、身份未知保留、时间间断、时间切分、训练区间归一化、命名对齐、组批填充、GFP 排除。
- 实测读取示例：2023-01-18-01，126 列中 90 个有名称，X=(4,32,126)，Y=(4,8,126)。
- 最新部署核验文件 runtime/neural-v2-verification.json；它是一次快照，不是持续状态。
- 2026-09-15 16:37（香港时间）：用户要求取消模型调用上限，已将 daily_llm_calls_max 设为 null，并同步取消筛选、重试和检索入口的每日次数门槛。计数继续保留，其他资源限制不变。部署证据见 runtime/unlimited-model-deployment.json；一次性脚本 remove_model_cap.py 不要重复执行。

## 常驻整理更新（2026-09-15 23:25 香港时间）

用户授权另开常驻整理agent，明确要求停止条件与原采集器共享。已实际部署：

- 仍只有一个专用 `worm-collector.service`。drop-in `/etc/systemd/system/worm-collector.service.d/20-organizer.conf` 让轻量supervisor管理采集和整理两个工作进程；原collector.py未修改。
- 三个进程在同一CPU/内存cgroup，合计最多2核和2,000,000,000字节，不是每个进程各有2GB。
- 共用 `state/control.json` 的 enabled 开关和原存储预算。禁用时两个工作进程退出，监督进程保留以检测重新启用；停止systemd服务会停止全部三个进程。
- 整理worker每30秒增量读取已verified文件，分类并生成适配优先清单；不新增LLM调用，不使用cron，不执行pickle/仓库代码，不改原始文件与catalog。
- 新脚本位于 `/opt/worm-collector/app/organizer_*.py`，本机对应 `C:/projects/bio/organization/`。
- 整理结果位于 `/opt/worm-collector/organization/`：entries、categories、index.json、adapter_priority.json、supervisor.json。分类不会自动增加训练记录。
- 已逐文件分类141条：发现其他物种MAT、Git LFS指针、README/license、图像视频和模型输出混入下载；这些不计作有效训练序列。候选神经CSV、NPZ、神经+运动MAT另列待适配。
- 23:25:10独立核验：服务active/running，两工作进程均在；合计内存约122MB。论文条目2845，verified文件141，内容去重137，去重861,791,132字节；标准记录仍89/eligible86。
- 已验证共享禁用/恢复、整个服务停止/重启、三进程同cgroup、磁盘门槛、纯缓存周期不重复读取内容或重写entry；四项代理服务均保持active。
- 本地证据：`organization/verification.json`、`shared_control_verification.json`、`service_stop_restart_verification.json`、`cache_cycle_verification.json`；中文报告 `organization/下载整理报告.md`。
- 远端部署前备份：`/opt/worm-collector/runtime/before-organizer-20260915T152102Z`。
