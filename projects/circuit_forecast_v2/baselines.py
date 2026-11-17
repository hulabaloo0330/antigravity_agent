import numpy as np
from common import neural_metrics, behavior_metrics

def persistence(data):return np.repeat(data['x'][:,-1:,:],data['y'].shape[1],axis=1)

def features(x):
    return np.concatenate([x-x[:,-1:],x[:,-1:],np.ones((len(x),1))],axis=1).astype(np.float64)

def solve(x,y,alpha):
    x,y=np.asarray(x,np.float64),np.asarray(y,np.float64)
    penalty=np.eye(x.shape[1])*alpha;penalty[-1,-1]=0
    return np.linalg.solve(x.T@x+penalty,x.T@y)

def fit_neural(train,alpha):
    N,H=train['x'].shape[-1],train['y'].shape[1]
    coef=np.zeros((N,train['x'].shape[1]+2,H))
    for n in range(N):
        s=train['present'][:,n]
        assert s.any()
        coef[n]=solve(features(train['x'][s,:,n]),train['y'][s,:,n]-train['x'][s,-1:,n],alpha)
    return coef

def predict_neural(coef,data):
    pred=persistence(data).copy()
    for n in range(pred.shape[-1]):pred[:,:,n]+=features(data['x'][:,:,n])@coef[n]
    return np.where(data['present'][:,None,:],pred,0)

def decoder_features(neural,present):
    return np.concatenate([neural*present,present.astype(np.float32),np.ones((*neural.shape[:-1],1))],axis=-1)

def decode(neural,present,coef,mean,std):
    mask=np.broadcast_to(present[:,None,:],neural.shape) if neural.ndim==3 else present
    return (decoder_features(neural,mask)@coef)*std+mean

def history_summary(data):
    x=data['x'];p=data['present'].astype(np.float32)
    return np.concatenate([x[:,-1,:],x.mean(1),x[:,-1,:]-x[:,0,:],p,np.ones((len(x),1))],1)

def fit_all(train,val,frame_train,frame_val,config):
    trials=[];best=float('inf');weights=None
    for alpha in config['ridge_alphas']:
        c=fit_neural(train,alpha);score=neural_metrics(predict_neural(c,val),val)['mse']
        trials.append({'alpha':alpha,'validation_mse':score})
        if score<best:best=score;weights=c;chosen_alpha=alpha
    mean=frame_train['behavior'].mean(0,dtype=np.float64)
    std=frame_train['behavior'].std(0,dtype=np.float64);assert (std>0).all()
    x=decoder_features(frame_train['neural'],frame_train['present'])
    y=(frame_train['behavior']-mean)/std
    decoder_trials=[];best=float('inf')
    for alpha in config['ridge_alphas']:
        c=solve(x,y,alpha)
        pred=decode(frame_val['neural'],frame_val['present'],c,mean,std)
        score=behavior_metrics(pred,frame_val['behavior'],frame_val['record'],std)['score']
        decoder_trials.append({'alpha':alpha,'validation_score':score})
        if score<best:best=score;decoder=c;decoder_alpha=alpha
    behavior_candidates=[];best=float('inf')
    bx=(train['behavior_x']-mean)/std;by=(train['behavior_y']-mean)/std
    vx=(val['behavior_x']-mean)/std
    for alpha in config['ridge_alphas']:
        coef=np.stack([solve(features(bx[:,:,k]),by[:,:,k]-bx[:,-1:,k],alpha) for k in range(4)])
        pred=np.stack([vx[:,-1:,k]+features(vx[:,:,k])@coef[k] for k in range(4)],axis=-1)*std+mean
        score=behavior_metrics(pred,val['behavior_y'],val['record'],std)['score']
        behavior_candidates.append({'alpha':alpha,'validation_score':score})
        if score<best:best=score;behavior_ar=coef;behavior_alpha=alpha
    direct_trials=[];best=float('inf')
    for alpha in config['ridge_alphas']:
        coef=solve(history_summary(train),by.reshape(len(by),-1),alpha)
        pred=(history_summary(val)@coef).reshape(val['behavior_y'].shape)*std+mean
        score=behavior_metrics(pred,val['behavior_y'],val['record'],std)['score']
        direct_trials.append({'alpha':alpha,'validation_score':score})
        if score<best:best=score;direct=coef;direct_alpha=alpha
    return {'neural_ridge':weights,'decoder':decoder,'behavior_ar':behavior_ar,'direct':direct,
            'behavior_mean':mean,'behavior_std':std}, {
            'neural':trials,'neural_alpha':chosen_alpha,'decoder':decoder_trials,'decoder_alpha':decoder_alpha,
            'behavior_ar':behavior_candidates,'behavior_alpha':behavior_alpha,'direct':direct_trials,'direct_alpha':direct_alpha}

def behavior_baselines(weights,data):
    mean,std=weights['behavior_mean'],weights['behavior_std']
    bx=(data['behavior_x']-mean)/std;H=data['behavior_y'].shape[1]
    ar=np.stack([bx[:,-1:,k]+features(bx[:,:,k])@weights['behavior_ar'][k] for k in range(4)],axis=-1)*std+mean
    direct=(history_summary(data)@weights['direct']).reshape(data['behavior_y'].shape)*std+mean
    return {'behavior_persistence':np.repeat(data['behavior_x'][:,-1:,:],H,axis=1),
            'behavior_ar':ar,'direct_neural_history':direct,
            'observed_future_neural_readout':decode(data['y'],data['present'],weights['decoder'],mean,std)}
