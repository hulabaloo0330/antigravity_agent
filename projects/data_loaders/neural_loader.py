"""NumPy loader; yields [time, neuron] windows for CNN/Transformer input.

Default split is chronological WITHIN each recording, not cross-animal evaluation.
No windows overlap split boundaries or timestamp gaps. No full-trace normalization.
"""
import json,pathlib
import numpy as np

def load_record(path,neuron_names=None):
    p=pathlib.Path(path);meta=json.loads((p/'metadata.json').read_text())
    if not meta['training_eligible']:raise ValueError('Record is a control or needs review')
    a=np.load(p/'signal.npy',mmap_mode='r',allow_pickle=False)
    mask=np.load(p/'valid.npy',mmap_mode='r',allow_pickle=False)
    neurons=json.loads((p/'neurons.json').read_text())
    if neuron_names is not None:
        aligned=np.full((len(a),len(neuron_names)),np.nan,dtype=np.float32);valid=np.zeros(aligned.shape,dtype=bool)
        for k,name in enumerate(neuron_names):
            matched=[n['column_index'] for n in neurons if n['neuron_name']==name]
            if len(matched)>1:raise ValueError('Ambiguous neuron identity: '+name)
            if matched:aligned[:,k]=a[:,matched[0]];valid[:,k]=mask[:,matched[0]]
        a,mask=aligned,valid
    segments=json.loads((p/'segments.json').read_text())['intervals']
    channel_ids=list(neuron_names) if neuron_names is not None else [n['neuron_name'] or meta['record_key']+':'+n['local_channel_id'] for n in neurons]
    return {'signal':a,'valid':mask,'time_s':np.load(p/'time_s.npy',mmap_mode='r',allow_pickle=False),'metadata':meta,'segments':segments,'neurons':neurons,'channel_ids':channel_ids}

def partition(segment,split):
    start,stop=segment['start'],segment['stop'];n=stop-start
    bounds=[start,start+int(.70*n),start+int(.85*n),stop]
    if split=='all':return start,stop
    ix={'train':0,'validation':1,'test':2}[split]
    return bounds[ix],bounds[ix+1]

def fit_training_scaler(record):
    """Only the training portions contribute; future validation/test data excluded."""
    n=record['signal'].shape[1];count=np.zeros(n);total=np.zeros(n);squares=np.zeros(n)
    for seg in record['segments']:
        a,b=partition(seg,'train');x=record['signal'][a:b];valid=record['valid'][a:b]
        safe=np.where(valid,x,0).astype(np.float64)
        count+=valid.sum(0);total+=safe.sum(0);squares+=(safe*safe).sum(0)
    mean=np.divide(total,count,out=np.zeros(n),where=count>0)
    variance=np.divide(squares,count,out=np.zeros(n),where=count>0)-mean*mean
    std=np.sqrt(np.maximum(variance,0));std[std<1e-8]=1
    return {'mean':mean.astype(np.float32),'std':std.astype(np.float32),'n_training_observations':count.astype(np.int64)}

def windows(record,history=32,horizon=8,stride=8,split='train',scaler=None):
    if min(history,horizon,stride)<1:raise ValueError('Positive window lengths required')
    for seg in record['segments']:
        start,stop=partition(seg,split)
        for left in range(start,stop-history-horizon+1,stride):
            mid=left+history;right=mid+horizon
            x=np.asarray(record['signal'][left:mid]);y=np.asarray(record['signal'][mid:right])
            mx=np.asarray(record['valid'][left:mid]);my=np.asarray(record['valid'][mid:right])
            if not mx.any() or not my.any():continue
            if scaler is not None:x=(x-scaler['mean'])/scaler['std'];y=(y-scaler['mean'])/scaler['std']
            yield {'x':x,'y':y,'x_valid':mx,'y_valid':my,'time_x_s':record['time_s'][left:mid],
                'time_y_s':record['time_s'][mid:right],'record_key':record['metadata']['record_key'],
                'source_group_id':record['metadata']['source_group_id'],'channel_ids':record['channel_ids'],'indices':(left,mid,right)}

def batch_windows(items):
    """Pad neurons only, retaining masks. Model tensors use zero placeholders;
    original files and unbatched windows retain NaN. Compute losses on y_valid only.
    Unaligned channels require an identity-aware model; fixed-channel CNNs should
    first pass the SAME neuron_names list to load_record for every animal.
    """
    if not items:raise ValueError('Empty batch')
    h=items[0]['x'].shape[0];f=items[0]['y'].shape[0];n=max(x['x'].shape[1] for x in items)
    x=np.zeros((len(items),h,n),np.float32);y=np.zeros((len(items),f,n),np.float32)
    mx=np.zeros(x.shape,bool);my=np.zeros(y.shape,bool)
    for b,item in enumerate(items):
        count=item['x'].shape[1]
        if item['x'].shape[0]!=h or item['y'].shape!=(f,count):raise ValueError('Incompatible window dimensions')
        x[b,:,:count]=np.where(item['x_valid'],item['x'],0)
        y[b,:,:count]=np.where(item['y_valid'],item['y'],0)
        mx[b,:,:count]=item['x_valid'];my[b,:,:count]=item['y_valid']
    return {'x':x,'y':y,'x_valid':mx,'y_valid':my,'channel_ids':[i['channel_ids'] for i in items],
        'record_keys':[i['record_key'] for i in items],'time_x_s':np.stack([i['time_x_s'] for i in items]),'time_y_s':np.stack([i['time_y_s'] for i in items])}

if __name__=='__main__':
    import sys
    r=load_record(sys.argv[1]);scale=fit_training_scaler(r)
    item=next(windows(r,scaler=scale))
    print(json.dumps({'x_shape':list(item['x'].shape),'y_shape':list(item['y'].shape),'record_key':item['record_key'],'split':'within_record_chronological_train','normalization':'training_partition_only'}))
