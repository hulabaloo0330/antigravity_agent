"""三折日期分组：全部内层选型完成后才统一外层测试。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import random
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from common import read_json,write_json,sha256,neural_metrics,behavior_metrics
from models import make_model,loss_fn
from baselines import fit_all,persistence,predict_neural,decode,behavior_baselines

HERE=Path(__file__).resolve().parent

def stamp():return datetime.now(timezone.utc).isoformat()
def status(out,phase,**fields):
    value={'time':stamp(),'phase':phase,**fields};write_json(out/'status.json',value)
    print(__import__('json').dumps(value,ensure_ascii=False),flush=True)

def subset(data,ids):
    selected=np.isin(data['record'],ids)
    return {k:v[selected] for k,v in data.items()}

def to_gpu(data):
    return {k:torch.as_tensor(data[k],device='cuda') for k in ['x','y','present','time_x']}

@torch.inference_mode()
def predict(model,tensors):
    model.eval()
    return torch.cat([model(tensors['x'][i:i+128],tensors['present'][i:i+128],tensors['time_x'][i:i+128])
                      for i in range(0,len(tensors['x']),128)]).cpu().numpy()

def mask_check(model,tensors):
    model.eval()
    x,p,tx=(tensors[k][:8].clone() for k in ['x','present','time_x'])
    with torch.inference_mode():
        prediction=model(x,p,tx)
        torch.testing.assert_close(model(x.masked_fill(~p[:,None,:],9999),p,tx),prediction,rtol=0,atol=0)
        y=tensors['y'][:8]
        torch.testing.assert_close(loss_fn(prediction,y,p),loss_fn(prediction,y.masked_fill(~p[:,None,:],-9999),p),rtol=0,atol=0)

def fit_network(fold,variant,seed,graph_seed,config,manifest,graphs,train,val,out):
    name=f"fold{fold}_{variant}_s{seed}";directory=out/'runs'/name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return read_json(directory/'complete.json')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=make_model(config,manifest['names'],graphs,variant,graph_seed).to('cuda')
    tensors,validation=to_gpu(train),to_gpu(val)
    mask_check(model,tensors)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    curve=[];best=float('inf');stale=0;N=len(train['x']);start_time=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,config['max_epochs']+1):
        model.train();order=torch.randperm(N,device='cuda');total=torch.zeros((),device='cuda')
        for left in range(0,N,config['batch_size']):
            idx=order[left:left+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            pred=model(tensors['x'][idx],tensors['present'][idx],tensors['time_x'][idx])
            loss=loss_fn(pred,tensors['y'][idx],tensors['present'][idx])
            if not torch.isfinite(loss):raise FloatingPointError(name)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            total+=loss.detach()*len(idx)
        score=neural_metrics(predict(model,validation),val)['mse']
        curve.append({'epoch':epoch,'train_mse':float((total/N).item()),'validation_mse':score})
        if score<best:
            best=score;best_epoch=epoch;stale=0
            torch.save({'state_dict':model.state_dict(),'variant':variant,'seed':seed,'graph_seed':graph_seed,
                        'config':config,'names':manifest['names'],'fold':fold,'epoch':epoch},directory/'best.pt')
        else:stale+=1
        write_json(directory/'curve.json',curve)
        if epoch==1 or epoch%5==0 or stale>=config['patience']:
            status(out,'training',run=name,epoch=epoch,best_validation=best,validation_mse=score)
        if stale>=config['patience']:break
    torch.cuda.synchronize();duration=time.perf_counter()-start_time
    peak=torch.cuda.max_memory_allocated()/1024**2
    checkpoint=torch.load(directory/'best.pt',map_location='cuda',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);val_pred=predict(model,validation)
    mask_check(model,validation)
    reread=make_model(config,manifest['names'],graphs,variant,graph_seed).to('cuda')
    reread.load_state_dict(checkpoint['state_dict'])
    np.testing.assert_array_equal(predict(reread,validation),val_pred)
    assert abs(neural_metrics(val_pred,val)['mse']-best)<1e-8
    result={'run':name,'fold':fold,'variant':variant,'seed':seed,'graph_seed':graph_seed,'seconds':duration,
            'peak_allocated_mib':peak,'epochs':epoch,'best_epoch':best_epoch,'validation_mse':best,
            'parameters':sum(p.numel() for p in model.parameters()),
            'checks':['finite loss and gradient','trained mask invariance','checkpoint exact reload']}
    write_json(directory/'complete.json',result)
    status(out,'run_complete',run=name,seconds=duration,validation_mse=best)
    return result

def grouped_ci(a,b,manifest,seed=20260917):
    # 指标差a-b；重采样日期组，组中保留全部记录，再计算记录宏平均。
    dates=sorted(set(r['date'] for r in manifest['records']))
    idx=[np.array([i for i,r in enumerate(manifest['records']) if r['date']==d]) for d in dates]
    difference=np.asarray(a)-np.asarray(b);sums=np.array([difference[i].sum() for i in idx]);counts=np.array([len(i) for i in idx])
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(dates),(10000,len(dates)))
    samples=sums[draw].sum(1)/counts[draw].sum(1)
    return {'record_macro_mse_difference':float(difference.mean()),'date_bootstrap_95_interval':np.quantile(samples,[.025,.975]).tolist(),
            'records_a_better':int((difference<0).sum()),'interpretation':'negative favors first model; descriptive same-cohort grouped bootstrap'}

def main(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    config=read_json(HERE/'config.json');manifest=read_json(HERE/'prepared/manifest.json')
    arrays=dict(np.load(HERE/'prepared/windows.npz'));frames=dict(np.load(HERE/'prepared/frames.npz'))
    graphs=dict(np.load(HERE/'prepared/graphs.npz'))
    source_hashes={p.name:sha256(p) for p in HERE.iterdir() if p.suffix in {'.py','.json','.md'}}
    prepared_hashes={p.name:sha256(p) for p in (HERE/'prepared').iterdir() if p.is_file()}
    if (out/'provenance.json').exists():
        previous=read_json(out/'provenance.json')
        assert previous['source_sha256']==source_hashes and previous['prepared_sha256']==prepared_hashes
    # 对远端数据副本再次核验，不依赖“已经传过”的假设。
    raw_root=HERE.parents[1]/'source_data/neural_forecast_v2_2026-09-15/records'
    for record in manifest['records']:
        for name,digest in record['files_sha256'].items():assert sha256(raw_root/record['key']/name)==digest
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    write_json(out/'provenance.json',{'started_utc':stamp(),'source_sha256':source_hashes,'prepared_sha256':prepared_hashes,
        'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'device':torch.cuda.get_device_name(0),
        'precision':'float32 deterministic, TF32 disabled','raw_hashes_verified':True})
    status(out,'source_verified',records=21,windows=len(arrays['x']))
    all_runs=[];fold_selections=[]
    for fold in manifest['folds']:
        k=fold['fold'];fd=out/f'fold{k}';fd.mkdir(exist_ok=True)
        train,val=(subset(arrays,fold[s]) for s in ['train','validation'])
        frame_train,frame_val=(subset(frames,fold[s]) for s in ['train','validation'])
        weights,baseline_selection=fit_all(train,val,frame_train,frame_val,config)
        np.savez(fd/'linear_weights.npz',**weights);write_json(fd/'baseline_selection.json',baseline_selection)
        runs=[]
        for variant in config['variants']:
            for seed,graph_seed in zip(config['seeds'],config['graph_seeds']):
                runs.append(fit_network(k,variant,seed,graph_seed,config,manifest,graphs,train,val,out))
        assert len(set(r['parameters'] for r in runs))==1
        all_runs.extend(runs)
        val_mse={v:float(np.mean([r['validation_mse'] for r in runs if r['variant']==v])) for v in config['variants']}
        val_mse['ridge']=min(r['validation_mse'] for r in baseline_selection['neural'])
        val_mse['persistence']=neural_metrics(persistence(val),val)['mse']
        chosen=min(val_mse,key=val_mse.get)
        neural_choice=min(config['variants'],key=val_mse.get)
        chosen_seed=min([r for r in runs if r['variant']==neural_choice],key=lambda r:r['validation_mse'])['seed']
        fold_selections.append({'fold':k,'validation_mse':val_mse,'selected_predictor':chosen,
            'selected_network':neural_choice,'selected_network_seed':chosen_seed})
    write_json(out/'selection_frozen.json',{'frozen_utc':stamp(),'folds':fold_selections,'outer_test_evaluated':False})
    status(out,'selection_frozen',folds=fold_selections)
    # 至此全部折的内层训练已完成，以下才进行外层测试。
    pred_dir=out/'predictions';pred_dir.mkdir(exist_ok=True)
    neural_ids=['persistence','ridge']+[f'{v}_s{s}' for v in config['variants'] for s in config['seeds']]
    neural_predictions={n:np.zeros_like(arrays['y']) for n in neural_ids}
    behavior_predictions={n:np.zeros_like(arrays['behavior_y']) for n in
        ['behavior_persistence','behavior_ar','direct_neural_history','observed_future_neural_readout']+[f'cascade_{n}' for n in neural_ids]}
    behavior_scales=np.zeros((len(manifest['records']),4))
    for fold in manifest['folds']:
        k=fold['fold'];selected=np.isin(arrays['record'],fold['test']);test={a:b[selected] for a,b in arrays.items()}
        tensors=to_gpu(test);weights=dict(np.load(out/f'fold{k}/linear_weights.npz'))
        for rid in fold['test']:behavior_scales[rid]=weights['behavior_std']
        for n,pred in behavior_baselines(weights,test).items():behavior_predictions[n][selected]=pred
        local={'persistence':persistence(test),'ridge':predict_neural(weights['neural_ridge'],test)}
        for variant in config['variants']:
            for seed,graph_seed in zip(config['seeds'],config['graph_seeds']):
                checkpoint=torch.load(out/'runs'/f'fold{k}_{variant}_s{seed}'/'best.pt',map_location='cuda',weights_only=True)
                model=make_model(config,manifest['names'],graphs,variant,graph_seed).to('cuda')
                model.load_state_dict(checkpoint['state_dict']);local[f'{variant}_s{seed}']=predict(model,tensors)
        for name,pred in local.items():
            neural_predictions[name][selected]=pred
            behavior_predictions['cascade_'+name][selected]=decode(pred,test['present'],weights['decoder'],weights['behavior_mean'],weights['behavior_std'])
    neural_scores={}
    for name,pred in neural_predictions.items():
        assert np.isfinite(pred).all();neural_scores[name]=neural_metrics(pred,arrays)
        np.save(pred_dir/f'neural_{name}.npy',pred)
    groups={'persistence':['persistence'],'ridge':['ridge'],**{v:[f'{v}_s{s}' for s in config['seeds']] for v in config['variants']}}
    group_scores={};record_mses={}
    for name,ids in groups.items():
        scores=[neural_scores[n]['mse'] for n in ids]
        record_mses[name]=np.mean([[r['mse'] for r in neural_scores[n]['per_record']] for n in ids],0)
        group_scores[name]={'mse_mean':float(np.mean(scores)),'mse_seed_std':float(np.std(scores,ddof=1)) if len(scores)>1 else None,
            'horizon_mse':np.mean([neural_scores[n]['horizon_mse'] for n in ids],0).tolist(),
            'per_record_mse':record_mses[name].tolist(),'improvement_over_persistence':float(1-np.mean(scores)/neural_scores['persistence']['mse'])}
    comparisons={f'anatomical_minus_{name}':grouped_ci(record_mses['anatomical'],record_mses[name],manifest)
                 for name in ['self','dense','rewired','ridge','persistence']}
    behavior_scores={}
    for name,pred in behavior_predictions.items():
        assert np.isfinite(pred).all();rows=[]
        for rid in range(len(manifest['records'])):
            s=arrays['record']==rid
            metric=behavior_metrics(pred[s],arrays['behavior_y'][s],arrays['record'][s],behavior_scales[rid])
            rows.extend(metric['per_record'])
        mse=np.mean([r['mse'] for r in rows],0);norm=np.mean([r['normalized_mse'] for r in rows],0)
        behavior_scores[name]={'rmse':np.sqrt(mse).tolist(),'mse':mse.tolist(),'normalized_mse':norm.tolist(),
                               'score':float(norm.mean()),'per_record':rows}
        np.save(pred_dir/f'behavior_{name}.npy',pred)
    np.save(out/'behavior_scales.npy',behavior_scales)
    results={'completed_utc':stamp(),'neural_summary':group_scores,'neural_scores':neural_scores,
             'graph_comparisons':comparisons,'behavior_scores':behavior_scores,'fold_selections':fold_selections,
             'training_runs':all_runs,'training_seconds':sum(r['seconds'] for r in all_runs),
             'peak_allocated_mib':max(r['peak_allocated_mib'] for r in all_runs),
             'scope':'date-held-out recording forecasts after neural prefix calibration; no test weight adaptation'}
    write_json(out/'results.json',results)
    status(out,'complete',neural_mse={k:v['mse_mean'] for k,v in group_scores.items()},training_seconds=results['training_seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args()
    try:main(a.out)
    except BaseException:
        a.out.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.out/'error.txt').write_text(error,encoding='utf-8')
        status(a.out,'failed',error=error);raise
