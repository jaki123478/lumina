"""Adattatori addestrabili locali; i pesi del modello base restano congelati."""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        for p in self.base.parameters():p.requires_grad_(False)

    def forward(self,x):
        original=self.base(x)
        delta=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.lora_A),self.lora_B)
        return original+delta.to(original.dtype)*self.scale


def attach(model,rank=8,alpha=16):
    for p in model.parameters():p.requires_grad_(False)
    selected=[(name,module) for name,module in model.named_modules()
              if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in ('q_proj','v_proj')]
    for name,module in selected:
        parent,attr=name.rsplit('.',1)
        setattr(model.get_submodule(parent),attr,LoRALinear(module,rank,alpha))
    return [name for name,module in selected]


def adapter_state(model):
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if 'lora_A' in name or 'lora_B' in name}


def apply_state(model,state):
    parameters=dict(model.named_parameters())
    expected={n for n in parameters if 'lora_A' in n or 'lora_B' in n}
    if set(state)!=expected:raise ValueError('Adattatore incompatibile con il modello base')
    with torch.no_grad():
        for name,value in state.items():parameters[name].copy_(value.to(parameters[name].device))
