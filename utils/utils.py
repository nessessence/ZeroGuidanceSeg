import torch
import numpy as np
import torch.nn.functional as F


def compute_cosine_sim(x,y,pairwise=False,norm=True):  
    x = x.clone().detach().float(); y = y.clone().detach().float()
    if norm:
        x /= x.norm(dim=-1, keepdim=True)
        y /= y.norm(dim=-1, keepdim=True)

    if pairwise: 
        similarity = torch.sum(x * y, dim=1)
    else: similarity = x @ y.T
    return similarity


def kl_div_uniformtest(t):
    if isinstance(t,np.ndarray) or isinstance(t,list): t = torch.tensor(t, dtype=torch.float32)
    else:  t = t.float()
    norm_t =  F.log_softmax(t,dim=t.ndim-1)
    target_uniform = F.softmax(torch.ones(t.shape),dim=t.ndim-1).to(t.get_device() if t.get_device() >=0 else 'cpu')
    return F.kl_div(norm_t,target=target_uniform,reduction='batchmean')

