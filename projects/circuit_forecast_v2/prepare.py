"""准备日期分组、校准后的活动与行为数据，以及经校验的真实/随机连接图。"""
import itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import read_json, write_json, sha256

HERE=Path(__file__).resolve().parent

def rewired_graph(a, seed, directed):
    rng=np.random.default_rng(seed)
    candidates=np.argwhere(a>0)
    if not directed: candidates=candidates[candidates[:,0]<candidates[:,1]]
    edges=[tuple(map(int,e)) for e in candidates]
    weights=[float(a[e]) for e in edges]
    occupied=set(edges); swaps=0; target=20*len(edges)
    for _ in range(1000*max(len(edges),1)):
        if swaps>=target: break
        i,j=rng.choice(len(edges),2,replace=False)
        u,v=edges[i];x,y=edges[j]
        if len({u,v,x,y})<4: continue
        if not directed and rng.random()<.5: x,y=y,x
        e1,e2=(u,y),(x,v)
        if not directed: e1,e2=tuple(sorted(e1)),tuple(sorted(e2))
        if e1 in occupied or e2 in occupied:continue
        occupied.remove(edges[i]);occupied.remove(edges[j]);occupied.update([e1,e2])
        edges[i],edges[j]=e1,e2;swaps+=1
    b=np.zeros_like(a)
    for (u,v),w in zip(edges,weights):
        b[u,v]=w
        if not directed:b[v,u]=w
    np.testing.assert_array_equal((a>0).sum(0),(b>0).sum(0))
    np.testing.assert_array_equal((a>0).sum(1),(b>0).sum(1))
    np.testing.assert_array_equal(np.sort(a[a>0]),np.sort(b[b>0]))
    assert swaps>=target and not np.trace(b)
    return b,{'swaps':swaps,'edge_overlap_fraction':float(((a>0)&(b>0)).sum()/(a>0).sum())}

def prepare_graph(names, out, config):
    import openpyxl
    source=HERE/'sources/Cook2019_corrected2020.xlsx'
    digest=sha256(source)
    assert digest=='1f4fdbf84746b69b49a8da0816f52787860ce349b638dce37924ba80f90c70c9'
    workbook=openpyxl.load_workbook(source,read_only=True,data_only=True)
    matrices={}
    for kind,sheet in [('chemical','hermaphrodite chemical'),('electrical','hermaphrodite gap jn symmetric')]:
        rows=list(workbook[sheet].values)
        row_map={v[2]:i for i,v in enumerate(rows[3:],3) if isinstance(v[2],str)}
        col_map={v:i for i,v in enumerate(rows[2][3:],3) if isinstance(v,str)}
        missing=[n for n in names if n not in row_map or n not in col_map]
        if missing:raise ValueError(f'Unmatched graph names {missing}')
        raw=np.array([[rows[row_map[a]][col_map[b]] or 0 for b in names] for a in names],dtype=np.float32)
        assert (raw>=0).all() and np.isfinite(raw).all()
        # 注意力行=接收者、列=输入者；源表行=突触前、列=突触后。
        matrices[kind]=raw.T.copy()
        np.fill_diagonal(matrices[kind],0)
    np.testing.assert_array_equal(matrices['electrical'],matrices['electrical'].T)
    report={'source_url':'https://wormwiring.org/pages/adjacency.html','source_sha256':digest,
            'reference_doi':'10.1038/s41586-019-1352-7','names':names,'matrix_orientation':'target_row_source_column',
            'chemical_directed_edges':int((matrices['chemical']>0).sum()),
            'electrical_undirected_edges':int((matrices['electrical']>0).sum()//2),
            'weight_meaning':'total EM serial sections, not conductance or excitatory/inhibitory sign',
            'rewiring':{}}
    for seed in config['graph_seeds']:
        for kind in ('chemical','electrical'):
            a,stats=rewired_graph(matrices[kind],seed,kind=='chemical')
            matrices[f'{kind}_{seed}']=a;report['rewiring'][f'{kind}_{seed}']=stats
    np.savez(out/'graphs.npz',**matrices)
    write_json(out/'graph_manifest.json',report)
    return report

def date_folds(records, seed):
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r['date']].append(i)
    rng=np.random.default_rng(seed)
    keys=list(groups);rng.shuffle(keys);keys.sort(key=lambda d:-len(groups[d]))
    buckets=[[],[],[]];sizes=[0,0,0]
    for date in keys:
        k=int(np.argmin(sizes));buckets[k].append(date);sizes[k]+=len(groups[date])
    folds=[]
    for k,test_dates in enumerate(buckets):
        rest=[d for d in groups if d not in test_dates];rng.shuffle(rest)
        choices=list(itertools.combinations(rest,2))
        val_dates=min(choices,key=lambda ds:abs(sum(len(groups[d]) for d in ds)-4))
        train_dates=[d for d in rest if d not in val_dates]
        folds.append({'fold':k,'train_dates':train_dates,'validation_dates':list(val_dates),'test_dates':test_dates,
            **{s:sorted(i for d in ds for i in groups[d]) for s,ds in
               [('train',train_dates),('validation',val_dates),('test',test_dates)]}})
    assert sorted(i for f in folds for i in f['test'])==list(range(len(records)))
    for f in folds:
        for a,b in itertools.combinations(('train','validation','test'),2):
            assert not(set(f[a])&set(f[b])) and not(set(f[a+'_dates'])&set(f[b+'_dates']))
    return folds

def prepare(data_root, out):
    data_root,out=Path(data_root),Path(out);out.mkdir(parents=True,exist_ok=True)
    config=read_json(HERE/'config.json')
    old=read_json(HERE.parent/'neural_forecast_v1/prepared/manifest.json')
    names=old['names'];L,H,S,C=(config[k] for k in ['history','horizon','stride','calibration_frames'])
    parts={k:[] for k in ['x','y','present','record','left','time_x','time_y','behavior_x','behavior_y']}
    frame_parts={k:[] for k in ['neural','behavior','present','record']}
    records=[];means=[];stds=[]
    for rid,previous in enumerate(old['records']):
        path=data_root/'records'/previous['key'];m=read_json(path/'metadata.json')
        cells=read_json(path/'neurons.json');mapping={n['neuron_name']:n['column_index'] for n in cells if n.get('neuron_name') in names}
        assert len(mapping)==len([n for n in cells if n.get('neuron_name') in names])
        raw=np.load(path/'signal.npy');valid=np.load(path/'valid.npy');times=np.load(path/'time_s.npy')
        a=np.zeros((len(raw),len(names)),np.float32);present=np.zeros(len(names),bool)
        for j,name in enumerate(names):
            if name in mapping:
                ix=mapping[name];assert valid[:,ix].all() and np.isfinite(raw[:,ix]).all()
                a[:,j]=raw[:,ix];present[j]=True
        mean=a[:C].mean(0,dtype=np.float64).astype(np.float32)
        std=a[:C].std(0,dtype=np.float64).astype(np.float32);std[std<1e-6]=1
        z=np.where(present[None,:],(a-mean)/std,0).astype(np.float32)
        changed=a.copy();changed[C:]+=99999
        np.testing.assert_array_equal(changed[:C].mean(0,dtype=np.float64).astype(np.float32),mean)
        assert np.isfinite(z).all() and np.max(np.diff(times))<1.5*np.median(np.diff(times))
        behavior=np.load(path/'behavior.npy').astype(np.float32)
        channel_names=[v['name'] for v in read_json(path/'behavior_channels.json')]
        assert channel_names==config['behavior_names'] and np.load(path/'behavior_valid.npy').all()
        assert behavior.shape==(len(a),4) and np.isfinite(behavior).all()
        behavior[:,0]*=.1
        for left in range(C,len(z)-L-H+1,S):
            mid=left+L;right=mid+H
            for key,value in [('x',z[left:mid]),('y',z[mid:right]),('present',present),('record',rid),
                              ('left',left),('time_x',times[left:mid]-times[mid-1]),('time_y',times[mid:right]),
                              ('behavior_x',behavior[left:mid]),('behavior_y',behavior[mid:right])]:parts[key].append(value)
        for key,value in [('neural',z[C:]),('behavior',behavior[C:]),('present',np.tile(present,(len(z)-C,1))),
                          ('record',np.full(len(z)-C,rid,dtype=np.int64))]:frame_parts[key].append(value)
        means.append(mean);stds.append(std)
        records.append({'record_index':rid,'key':path.name,'id':m['source_record_id'],'date':m['source_record_id'][:10],
            'animal_id':m['animal_id'],'source_group_id':m['source_group_id'],'n_frames':len(a),
            'present_neurons':int(present.sum()),'calibration_frames':C,'max_abs_normalized':float(np.abs(z).max()),
            'files_sha256':{f:sha256(path/f) for f in ['signal.npy','valid.npy','time_s.npy','neurons.json','metadata.json',
                                                    'behavior.npy','behavior_valid.npy','behavior_channels.json']}})
    arrays={k:np.stack(v) for k,v in parts.items()};frames={k:np.concatenate(v) for k,v in frame_parts.items()}
    np.savez(out/'windows.npz',**arrays);np.savez(out/'frames.npz',**frames)
    np.savez(out/'calibration.npz',mean=np.stack(means),std=np.stack(stds))
    manifest={'config':config,'names':names,'records':records,'folds':date_folds(records,config['split_seed']),
              'n_windows':len(arrays['x']),'n_dates':len(set(r['date'] for r in records)),
              'checks':['source identities aligned','finite original data','initial-only neural calibration',
                        'held-out dates disjoint','every recording tested once','behavior synchronized, no test behavior calibration']}
    write_json(out/'manifest.json',manifest)
    g=prepare_graph(names,out,config)
    print(__import__('json').dumps({'records':len(records),'dates':manifest['n_dates'],'windows':manifest['n_windows'],
        'fold_counts':[{k:len(f[k]) for k in ['train','validation','test']} for f in manifest['folds']],
        'graph':g,'max_abs_normalized':max(r['max_abs_normalized'] for r in records)},ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();prepare(a.data,a.out)
