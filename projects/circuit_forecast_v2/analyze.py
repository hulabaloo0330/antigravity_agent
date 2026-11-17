"""独立复核全部外层预测，生成中文科研报告及图表；不训练、不改选型。"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read_json,write_json,sha256

LABELS={'persistence':'保持最后值','ridge':'线性自回归','self':'仅自身历史','dense':'全连接注意力',
        'anatomical':'真实解剖图','rewired':'保度随机图'}
COLORS={'persistence':'#d48f29','ridge':'#238a79','self':'#74859a','dense':'#6b79bc','anatomical':'#bd4d7b','rewired':'#9c795c'}
BEHAVIOR_LABELS=['速度（mm/s）','头部角度（rad）','角速度（rad/s）','咽泵速率（1/s）']

def main(root):
    root=Path(root).resolve();out=root/'results';figures=out/'figures';figures.mkdir(exist_ok=True)
    manifest=read_json(root/'prepared/manifest.json');config=manifest['config'];r=read_json(out/'results.json')
    data=dict(np.load(root/'prepared/windows.npz'));records=manifest['records'];scales=np.load(out/'behavior_scales.npy')
    verification={}
    for name,saved in r['neural_scores'].items():
        pred=np.load(out/'predictions'/f'neural_{name}.npy');assert np.isfinite(pred).all()
        per_record=[]
        for rid in range(len(records)):
            selected=data['record']==rid
            mask=np.broadcast_to(data['present'][selected,None,:],data['y'][selected].shape)
            difference=pred[selected].astype(np.float64)-data['y'][selected]
            per_record.append(float(np.mean(difference[mask]**2)))
        np.testing.assert_allclose(per_record,[x['mse'] for x in saved['per_record']],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.mean(per_record),saved['mse'],rtol=1e-12,atol=1e-12)
        verification['neural_'+name]=sha256(out/'predictions'/f'neural_{name}.npy')
    for name,saved in r['behavior_scores'].items():
        pred=np.load(out/'predictions'/f'behavior_{name}.npy');assert np.isfinite(pred).all()
        per_record=[]
        for rid in range(len(records)):
            s=data['record']==rid;difference=pred[s].astype(np.float64)-data['behavior_y'][s]
            per_record.append(np.mean(difference**2,axis=(0,1)))
        np.testing.assert_allclose(per_record,[x['mse'] for x in saved['per_record']],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.mean(per_record,0)),saved['rmse'],rtol=1e-12,atol=1e-12)
        verification['behavior_'+name]=sha256(out/'predictions'/f'behavior_{name}.npy')
    provenance=read_json(out/'provenance.json')
    for name,digest in provenance['source_sha256'].items():assert sha256(root/name)==digest
    for name,digest in provenance['prepared_sha256'].items():assert sha256(root/'prepared'/name)==digest
    np.testing.assert_array_equal(np.load(out/'predictions/neural_persistence.npy'),np.repeat(data['x'][:,-1:,:],8,axis=1))
    write_json(out/'independent_verification.json',{'passed':True,'prediction_sha256':verification,
        'checks':['all record-level neural MSE recomputed independently','all behavior physical RMSE recomputed independently',
                  'code/config/prepared hashes match trained version','persistence exactly reproduces last observation']})
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,
                         'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    order=list(LABELS);s=r['neural_summary']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    heights=[s[k]['mse_mean'] for k in order]
    bars=axes[0].bar(range(len(order)),heights,color=[COLORS[k] for k in order])
    axes[0].set_xticks(range(len(order)),[LABELS[k].replace('注意力','\n注意力').replace('解剖图','\n解剖图') for k in order],fontsize=9)
    axes[0].set(ylabel='留出记录 MSE（初始校准标准差单位）',title='按采集日期留出：21 条记录，三折')
    for b,v in zip(bars,heights):axes[0].text(b.get_x()+b.get_width()/2,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
    axes[0].set_ylim(0,max(heights)*1.16)
    for k in order:axes[1].plot(np.arange(1,9)*.6,s[k]['horizon_mse'],'.-',label=LABELS[k],color=COLORS[k])
    axes[1].set(xlabel='名义预测提前时间（秒）',ylabel='MSE',title='未来八帧的预测误差');axes[1].legend(fontsize=8,frameon=False)
    fig.savefig(figures/'neural_comparison.png');plt.close(fig)

    graph=np.load(root/'prepared/graphs.npz');fig,axes=plt.subplots(1,2,figsize=(13,6.5),layout='constrained')
    for ax,name,label in zip(axes,['chemical','electrical'],['化学连接：215 条有向边','电突触：42 对对称边']):
        im=ax.imshow(np.log1p(graph[name]),cmap='viridis',interpolation='nearest')
        ax.set_xticks(range(49),manifest['names'],rotation=90,fontsize=5);ax.set_yticks(range(49),manifest['names'],fontsize=5)
        ax.set(xlabel='输入神经元（突触前）',ylabel='接收神经元（突触后）',title=label)
        fig.colorbar(im,ax=ax,shrink=.65,label='log(1 + EM 切片量)')
    fig.savefig(figures/'anatomical_graph.png');plt.close(fig)

    behavior_group={}
    for name in ['behavior_persistence','behavior_ar','direct_neural_history','observed_future_neural_readout','cascade_ridge']:
        behavior_group[name]=r['behavior_scores'][name]
    for variant in config['variants']:
        scores=[r['behavior_scores'][f'cascade_{variant}_s{seed}'] for seed in config['seeds']]
        mse=np.mean([a['mse'] for a in scores],0);norm=np.mean([a['normalized_mse'] for a in scores],0)
        behavior_group['cascade_'+variant]={'mse':mse.tolist(),'rmse':np.sqrt(mse).tolist(),'normalized_mse':norm.tolist(),'score':float(norm.mean())}
    bnames={'behavior_persistence':'行为保持最后值','behavior_ar':'行为自身历史预测','direct_neural_history':'神经历史直接预测行为',
            'cascade_ridge':'线性神经预测→行为','cascade_anatomical':'真实图神经预测→行为','observed_future_neural_readout':'真实未来神经活动→行为（读出诊断）'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for k,ax in enumerate(axes.flat):
        names=list(bnames);v=[behavior_group[n]['rmse'][k] for n in names]
        ax.barh(range(len(names)),v,color=['#d48f29','#238a79','#6b79bc','#729e93','#bd4d7b','#aaaabb'])
        ax.set_yticks(range(len(names)),[bnames[n] for n in names],fontsize=8)
        ax.set(xlabel='RMSE，越小越好',title=BEHAVIOR_LABELS[k]);ax.invert_yaxis()
    fig.savefig(figures/'behavior_comparison.png');plt.close(fig)

    rid=next(i for i,v in enumerate(records) if v['key']=='ff8040a03d224b3215ddc5761224ce86')
    indices=np.flatnonzero(data['record']==rid)[:32];target=data['time_y'][indices,-1];target-=target[0]
    fig,axes=plt.subplots(4,1,figsize=(11,8),sharex=True,layout='constrained')
    predictions={n:np.load(out/'predictions'/f'behavior_{n}.npy') for n in ['behavior_ar','direct_neural_history','cascade_anatomical_s17']}
    for k,ax in enumerate(axes):
        ax.plot(target,data['behavior_y'][indices,-1,k],'k.-',label='实际行为',linewidth=1.3)
        for name,color,label in [('behavior_ar','#238a79','行为自身历史'),('direct_neural_history','#6b79bc','神经历史直接预测'),
                                 ('cascade_anatomical_s17','#bd4d7b','真实图串联预测')]:
            ax.plot(target,predictions[name][indices,-1,k],'.--',label=label,color=color,linewidth=.9)
        ax.set_ylabel(BEHAVIOR_LABELS[k])
    axes[0].legend(ncol=4,fontsize=8,frameon=False)
    axes[0].set_title('预先固定记录 2023-01-23-08：约提前4.8秒预测行为；每个点重新读取真实历史')
    axes[-1].set_xlabel('相对首个展示目标点的时间（秒）');fig.savefig(figures/'heldout_behavior_example.png');plt.close(fig)

    neural_rows=[]
    for k in order:
        d=s[k];deviation='' if d['mse_seed_std'] is None else f" ± {d['mse_seed_std']:.4f}"
        neural_rows.append(f"| {LABELS[k]} | {d['mse_mean']:.4f}{deviation} | {d['improvement_over_persistence']*100:+.1f}% |")
    comp_rows=[]
    for name,d in r['graph_comparisons'].items():
        other=name.removeprefix('anatomical_minus_');lo,hi=d['date_bootstrap_95_interval']
        comp_rows.append(f"| 真实图 − {LABELS[other]} | {d['record_macro_mse_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {d['records_a_better']}/21 |")
    behavior_rows=[]
    for name,label in bnames.items():behavior_rows.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in behavior_group[name]['rmse'])+' |')
    comparison=r['graph_comparisons']['anatomical_minus_rewired'];lo,hi=comparison['date_bootstrap_95_interval']
    if hi<0:graph_conclusion='真实图相对保度随机图的误差更低，日期分组重采样区间也低于零；这是本批数据下的预测证据，不能升级为因果机制证明。'
    elif lo>0:graph_conclusion='真实图相对保度随机图的误差更高，日期分组重采样区间也高于零；当前架构与观测子图下，解剖约束没有带来预期收益。'
    else:graph_conclusion='真实图与保度随机图的差异区间跨过零，本轮没有建立真实拓扑优于随机拓扑的稳定证据。'
    neural_best=min(s,key=lambda k:s[k]['mse_mean'])
    negative_pumping=float((np.load(out/'predictions/behavior_cascade_anatomical_s17.npy')[:,:,3]<0).mean())
    behavior_summary={'groups':behavior_group,'cascade_anatomical_s17_negative_pumping_fraction':negative_pumping}
    write_json(out/'behavior_summary.json',behavior_summary)
    report=f'''# 环路预测完整原型：训练与验证报告

完成时间：{r['completed_utc']}（UTC）。本轮在 RTX 5070 台式机完成36次训练、三折留出测试、行为读出和串联预测。结果已下载到本机，所有神经/行为预测的记录级指标独立重算通过。

## 交付结论

真实解剖图模型的神经测试 MSE 为 **{s['anatomical']['mse_mean']:.4f}**，相对保持最后值改善 **{s['anatomical']['improvement_over_persistence']*100:.1f}%**。本轮测试数值最低的是 **{LABELS[neural_best]}**。没有使用测试结果重新调参或修改训练停止点。

{graph_conclusion}

行为串联的四项RMSE均高于“行为自身历史线性预测”对照。即使输入真实未来神经活动，当前简单读出器在部分行为上的误差也很大，说明问题不只来自第一段神经预测的误差传播。当前版本适合作为研究基线与教学案例，不能宣称行为预测已经达到实用优势。

完成的是公开数据上的研究原型：神经历史 + 参考解剖图 → 未来神经活动 → 未来行为。它是可调用、可复核的完整软件实验；未完成“任意外部刺激→活动”模型和导师实验室材料的验证，不能把这两项说成已经解决。纯软件预测也不等于湿实验验证。

## 已完成的组成

| 部分 | 本轮实际完成 |
|---|---|
| 数据 | 21条同条件记录、49个有名称的神经元、同步四通道行为、缺失掩码、真实时间戳与来源哈希 |
| 环路 | Cook2019/2020修订参考图，215条有向化学边、42对电突触边，类型与方向分别处理 |
| 对照 | 相同网络参数量的自身图、全连接图、真实图、保度随机图；每类3种子 |
| 泛化 | 按11个采集日期三折留出，每条记录只进入外层测试一次；同日不跨训练/验证/测试 |
| 个体差异 | 初始160帧仅校准荧光均值/标准差，不用留出记录更新模型参数，不使用其未来数据或行为做校准 |
| 行为 | 同时刻真实神经→行为读出；预测神经→行为串联；行为自身历史与神经历史直接预测的对照 |
| 使用 | 保存所有模型与线性读出，predict.py支持给定命名神经元的历史进行串联推理 |
| 验证 | 哈希、分组、时间/名字/掩码、随机图度数和权重集合、重载、独立指标复算、真实留出记录推理示例 |

## 神经预测结果

| 模型 | 留出记录 MSE | 相对保持最后值改善 |
|---|---:|---:|
{chr(10).join(neural_rows)}

神经网络为三个种子的均值±标准差，指标先对每条记录有效目标计算，再在21条记录间平均。输入32帧、输出8帧；后者约4.8秒。这里的“神经强度”指作者提取的钙指示器荧光，不是膜电位，也没有推出脉冲。

本轮按初始160帧校准，而v1使用每条记录前70%归一化；同时本轮整条记录及采集日期留出。因此 **不能把v1与v2的MSE直接相减，宣称模型升级或退步**。某些节点校准期变化很小、后续变化很大，标准化幅度最高约95，说明跨状态尺度漂移仍然明显。

![神经预测模型比较](figures/neural_comparison.png)

## 是否用到了有意义的环路先验

以下差值为“真实图MSE − 对照MSE”，负数有利于真实图。区间按采集日期整组配对bootstrap，不把窗口当成独立动物样本。

| 对照差值 | 平均差 | 日期分组bootstrap 95%区间 | 真实图误差更低的记录数 |
|---|---:|---:|---:|
{chr(10).join(comp_rows)}

两层图注意力的query是接收神经元、key/value是输入神经元。允许化学突触前→后方向、电突触双向及自身边；化学/电突触的log权重分别作为偏置。历史活动和真实相对时间先通过MLP编码，注意力发生在神经元轴。本模型可称为带活动历史和时间输入的图Transformer，但没有声称时间轴也用了自注意力。

四种图网络参数数均为 {r['training_runs'][0]['parameters']:,}；同种子使用相同初始化和优化方法。随机化保留化学入/出度、电突触度数及各类型权重集合，三个随机图都在训练前固定。图是多动物拼接参考，不是本批个体各自测量的连接图；49节点诱导子图省略其余神经元及间接路径，可能使真实图约束过强。任何预测优势都不能单独证明生物因果，负结果也不能证明真实神经环路不重要。

![使用的解剖子图](figures/anatomical_graph.png)

## 行为模块与两段串联

下面均为实际物理单位的RMSE，越小越好；每个通道分别比较，不把不同单位的误差直接相加。真实图串联行为结果先在三个种子间平均MSE再取平方根。

| 模型/诊断 | 速度 mm/s | 头角 rad | 角速度 rad/s | 咽泵 1/s |
|---|---:|---:|---:|---:|
{chr(10).join(behavior_rows)}

“真实未来神经活动→行为”使用观测到的目标时刻神经活动，仅检验读出器，**不是未来预测方法或可部署的预测输入**。它也不是理论性能上限；简单读出器本身可能不足。其他预测方法只使用历史信息。串联模型将真实图预测出的未来49维活动交给固定线性读出，不偷用真实未来神经值。

真实图串联 seed17 的咽泵预测负值比例为 {negative_pumping*100:.2f}%；保留未裁剪的回归结果和指标。若非零，说明线性读出尚有物理范围限制，不能将这些值视作真实负频率或直接作为实验刺激参数。

![行为误差比较](figures/behavior_comparison.png)

![固定留出记录上的行为预测](figures/heldout_behavior_example.png)

图中每个点都重新读取此前真实历史，再预测第8帧；没有独自闭环运行两分钟。展示记录沿用事先固定的2023-01-23-08，使用前32个测试窗口，未按预测好坏选曲线。

## 资源与执行

36次神经网络训练合计 **{r['training_seconds']/60:.2f}分钟**，PyTorch张量分配峰值 **{r['peak_allocated_mib']:.1f}MiB**（不包括驱动、CUDA上下文和其他应用）。模型设置：宽32、4头、2层、batch64、AdamW学习率0.001，最多60轮，验证连续8轮不改善停止。三个外层折的全部训练和内层选型结束后才统一外层测试，冻结证据为 selection_frozen.json。

训练实际位置是台式机 `C:/projects/worm-circuit/experiments/circuit_forecast_v2`；本机副本 `C:/projects/bio/experiments/circuit_forecast_v2`。VPS未用于GPU训练，也没有改其代理。后台任务为一次性、无周期触发器，确认完成后移除。

## 数据来源与仍未解决的边界

活动/行为来自 [WormWideWeb](https://wormwideweb.org/about/datasets/) 的 [Atanas & Kim 2023](https://doi.org/10.1016/j.cell.2023.07.035)。连接图来自 [WormWiring修订版SI5](https://wormwiring.org/pages/adjacency.html)，对应 [Cook et al. 2019](https://doi.org/10.1038/s41586-019-1352-7)。完整文件、校验值、单位与解释见 SOURCE_AUDIT.md 和 prepared/graph_manifest.json。

本地没有可核实的独立动物/重复测量ID关系，因此称未见记录与日期，不称已经验证任意未见动物。作者上游影像提取、去噪或校正是否使用未来帧未逐文件验证，仍是离线评估。这批数据已在v1探索过，本轮不能冒充完全未接触过的外部验证数据。

baseline条件缺乏可训练任意外界输入的刺激描述；其他热刺激/再喂食条件尚需独立协议，不能与baseline静默混合。用户尚未提供导师实验室材料、任务输出定义及相应实验条件，实验室迁移验证不可能由当前数据代替。模型可预测相关性，尚不能保证敲除/激活神经元后的因果反应；软件中屏蔽节点也不等于真实消融实验。

## 使用与后续教学

`predict.Predictor`加载某一外层折的已训练模型和对应行为读出。输入原始提取荧光 `[time,channel]`、逐列神经元名称、秒数时间轴、窗口起点；先用最初160帧校准，然后从起点取32帧，返回 `[8,49]` 神经预测、可观测性掩码和 `[8,4]` 行为预测。原始荧光单位输出对未观测节点标为NaN，不冒充完整302神经元预测。具体可复用Python例子见 README.md。

教学将按数据张量→分组与泄漏→损失/优化→线性与MLP→图注意力→行为串联→证据与局限展开。这是已完成的实验基础，不要求用户再手动部署或练系统指令，也不表示用户已经掌握这些知识。
'''
    (out/'REPORT_完整原型与验证.md').write_text(report,encoding='utf-8')
    print(json.dumps({'verified':True,'neural_summary':s,'graph_comparisons':r['graph_comparisons'],
                      'behavior_summary':behavior_summary,'training_minutes':r['training_seconds']/60},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);a=p.parse_args();main(a.experiment)
