"""加载已训练图模型进行神经→行为串联推理；不读未来目标、不更新权重。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse
import numpy as np
import torch
from common import read_json
from models import make_model
from baselines import decode

class Predictor:
    def __init__(self,experiment,fold=0,variant='anatomical',seed=17,device='cpu'):
        self.root=Path(experiment);self.fold=fold;self.device=device
        self.config=read_json(self.root/'config.json')
        self.manifest=read_json(self.root/'prepared/manifest.json');self.names=self.manifest['names']
        checkpoint=torch.load(self.root/'results/runs'/f'fold{fold}_{variant}_s{seed}'/'best.pt',map_location=device,weights_only=True)
        self.model=make_model(self.config,self.names,dict(np.load(self.root/'prepared/graphs.npz')),variant,checkpoint['graph_seed']).to(device)
        self.model.load_state_dict(checkpoint['state_dict']);self.model.eval()
        self.linear=dict(np.load(self.root/f'results/fold{fold}/linear_weights.npz'))

    def forecast(self,signal,neuron_names,time_s,left):
        L,H,C=(self.config[k] for k in ['history','horizon','calibration_frames'])
        signal=np.asarray(signal);time_s=np.asarray(time_s)
        if signal.ndim!=2 or signal.shape[1]!=len(neuron_names) or time_s.shape!=(len(signal),):
            raise ValueError('Expected signal [time,channel], one name per channel, one timestamp per timepoint')
        if left<C or left+L>len(signal):raise ValueError('Need separate initial calibration and at least 32 historical frames')
        if len(set(n for n in neuron_names if n in self.names))!=sum(n in self.names for n in neuron_names):
            raise ValueError('Duplicate identified neuron columns')
        mapping={n:i for i,n in enumerate(neuron_names)}
        present=np.array([n in mapping for n in self.names],bool)
        if not present.any():raise ValueError('No recognized neuron identities')
        calibration=np.zeros((C,len(self.names)),np.float32);history=np.zeros((L,len(self.names)),np.float32)
        for k,name in enumerate(self.names):
            if present[k]:
                calibration[:,k]=signal[:C,mapping[name]]
                history[:,k]=signal[left:left+L,mapping[name]]
        if not np.isfinite(calibration).all() or not np.isfinite(history).all():raise ValueError('Partial missingness requires a separate preprocessing protocol')
        mean=calibration.mean(0,dtype=np.float64).astype(np.float32)
        std=calibration.std(0,dtype=np.float64).astype(np.float32);std[std<1e-6]=1
        z=np.where(present[None],(history-mean)/std,0).astype(np.float32)
        times=np.asarray(time_s[left:left+L])-time_s[left+L-1]
        if not np.all(np.diff(times)>0):raise ValueError('Time must increase')
        dt=np.diff(times);median_dt=float(np.median(dt))
        if not .54<=median_dt<=.66 or dt.max()>1.5*median_dt:
            raise ValueError('This checkpoint was trained near 0.6 seconds/frame without gaps; use a separately validated sampling protocol')
        with torch.inference_mode():
            pred=self.model(torch.from_numpy(z[None]).to(self.device),torch.from_numpy(present[None]).to(self.device),
                            torch.from_numpy(times[None]).to(self.device)).cpu().numpy()
        behavior=decode(pred,present[None],self.linear['decoder'],self.linear['behavior_mean'],self.linear['behavior_std'])[0]
        raw_pred=pred[0]*std+mean;raw_pred[:,~present]=np.nan
        return {'neural_normalized':pred[0],'neural_source_units':raw_pred,'present':present,
                'behavior':behavior,'neuron_names':np.array(self.names),'behavior_names':np.array(self.config['behavior_names']),
                'behavior_units':np.array(self.config['behavior_units']),
                'forecast_step':np.arange(1,H+1),'estimated_elapsed_s':np.arange(1,H+1)*np.median(np.diff(times))}

def record_input(path):
    path=Path(path);neurons=read_json(path/'neurons.json')
    names=[None]*len(neurons)
    for cell in neurons:names[cell['column_index']]=cell.get('neuron_name')
    return np.load(path/'signal.npy'),names,np.load(path/'time_s.npy')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fold',type=int,default=0);p.add_argument('--variant',default='anatomical');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--left',type=int,default=160);p.add_argument('--device',default='cpu')
    a=p.parse_args();predictor=Predictor(Path(__file__).parent,a.fold,a.variant,a.seed,a.device)
    result=predictor.forecast(*record_input(a.record),a.left)
    a.out.parent.mkdir(parents=True,exist_ok=True);np.savez(a.out,**result)
    print(f'Saved neural {result["neural_normalized"].shape} and behavior {result["behavior"].shape} predictions to {a.out}')
