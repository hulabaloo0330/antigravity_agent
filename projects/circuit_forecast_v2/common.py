import hashlib
import json
from pathlib import Path
import numpy as np

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def neural_metrics(pred, data):
    rows=[]
    for rid in np.unique(data['record']):
        s=data['record']==rid
        delta=pred[s].astype(np.float64)-data['y'][s]
        mask=np.broadcast_to(data['present'][s,None,:],delta.shape)
        rows.append({'record_index':int(rid),'mse':float(np.mean(delta[mask]**2)),
            'mae':float(np.mean(np.abs(delta[mask]))),
            'horizon_mse':(np.where(mask,delta**2,0).sum((0,2))/mask.sum((0,2))).tolist()})
    return {'mse':float(np.mean([r['mse'] for r in rows])), 'per_record':rows,
            'horizon_mse':np.mean([r['horizon_mse'] for r in rows],axis=0).tolist()}

def behavior_metrics(pred, target, record_ids, scale):
    rows=[]
    for rid in np.unique(record_ids):
        s=record_ids==rid
        d=pred[s].astype(np.float64)-target[s]
        mse=np.mean(d**2,axis=tuple(range(d.ndim-1)))
        rows.append({'record_index':int(rid),'mse':mse.tolist(),
                     'normalized_mse':(mse/scale**2).tolist()})
    mse=np.mean([r['mse'] for r in rows],axis=0)
    nmse=np.mean([r['normalized_mse'] for r in rows],axis=0)
    return {'mse':mse.tolist(),'rmse':np.sqrt(mse).tolist(),'normalized_mse':nmse.tolist(),
            'score':float(nmse.mean()),'per_record':rows}
