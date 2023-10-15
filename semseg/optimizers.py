from torch import nn
from torch.optim import AdamW, SGD
import torch


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01, log_vars=torch.ones((2,), requires_grad=True), bgm=0):
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.requires_grad:
            if p.dim() == 1:
                nwd_params.append(p)
            else:
                wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if bgm == 1:
        for log_var in log_vars:
            params.append(dict(params=log_var, lr=lr))

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)