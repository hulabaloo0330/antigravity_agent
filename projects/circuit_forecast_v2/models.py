"""相同网络，四种固定图；注意力行=接受输入的神经元。"""
import math
import numpy as np
import torch
from torch import nn

class GraphBlock(nn.Module):
    def __init__(self,width,heads,dropout):
        super().__init__();self.heads=heads;self.dim=width//heads
        self.qkv=nn.Linear(width,3*width);self.project=nn.Linear(width,width)
        self.edge_coefficient=nn.Parameter(torch.ones(heads,2))
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))
        self.dropout=nn.Dropout(dropout)

    def forward(self,h,present,support,edge_bias):
        B,N,W=h.shape
        q,k,v=self.qkv(h).reshape(B,N,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-1,-2))/math.sqrt(self.dim)
        scores=scores+torch.einsum('hk,kij->hij',self.edge_coefficient,edge_bias)[None]
        eye=torch.eye(N,dtype=torch.bool,device=h.device)
        allowed=(support[None]&present[:,None,:])|eye[None]
        attention=torch.softmax(scores.masked_fill(~allowed[:,None],float('-inf')),dim=-1)
        update=(self.dropout(attention)@v).transpose(1,2).reshape(B,N,W)
        h=self.norm1(h+self.dropout(self.project(update)))
        h=self.norm2(h+self.dropout(self.ff(h)))
        return h.masked_fill(~present[:,:,None],0)

class CircuitForecast(nn.Module):
    def __init__(self,config,names,chemical,electrical,variant):
        super().__init__();N=len(names);W=config['width'];self.variant=variant
        self.encoder=nn.Sequential(nn.Linear(2*config['history']+1,W),nn.GELU())
        self.identity=nn.Embedding(N,W)
        self.blocks=nn.ModuleList([GraphBlock(W,config['heads'],config['dropout']) for _ in range(config['layers'])])
        self.output=nn.Linear(W,config['horizon']);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        chemical=np.asarray(chemical,np.float32);electrical=np.asarray(electrical,np.float32)
        if variant=='self':support=np.eye(N,dtype=bool);bias=np.zeros((2,N,N),np.float32)
        elif variant=='dense':support=np.ones((N,N),bool);bias=np.zeros((2,N,N),np.float32)
        else:
            support=(chemical>0)|(electrical>0)|np.eye(N,dtype=bool)
            bias=np.stack([np.log1p(a)/max(float(np.log1p(a).max()),1) for a in [chemical,electrical]])
        self.register_buffer('support',torch.as_tensor(support))
        self.register_buffer('edge_bias',torch.as_tensor(bias))

    def forward(self,x,present,time_x):
        x=x.masked_fill(~present[:,None,:],0);last=x[:,-1,:]
        relative=(x-last[:,None,:]).transpose(1,2)
        times=(time_x/20).to(x.dtype)[:,None,:].expand(-1,x.shape[-1],-1)
        h=self.encoder(torch.cat([relative,last[:,:,None],times],dim=-1))+self.identity.weight[None]
        h=h.masked_fill(~present[:,:,None],0)
        for block in self.blocks:h=block(h,present,self.support,self.edge_bias)
        prediction=last[:,None,:]+self.output(h).transpose(1,2)
        return prediction.masked_fill(~present[:,None,:],0)

def loss_fn(pred,target,present):
    error=(pred-target).square().masked_fill(~present[:,None,:],0)
    return (error.sum((1,2))/(present.sum(1)*target.shape[1])).mean()

def make_model(config,names,graphs,variant,graph_seed):
    suffix=f'_{graph_seed}' if variant=='rewired' else ''
    return CircuitForecast(config,names,graphs['chemical'+suffix],graphs['electrical'+suffix],variant)
