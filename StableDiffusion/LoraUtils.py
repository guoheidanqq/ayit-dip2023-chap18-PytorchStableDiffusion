
import torch
import torch.nn as nn
from typing import Tuple
from .LoraLayer import LoraLayer


def injectLora(model:nn.Module,filterTuple:Tuple[str,...]=('in_proj','out_proj'),device='cuda')->None:
    for name,layer in model.named_modules():        
        filterName =filterTuple
        if name.endswith(filterName) and isinstance(layer,nn.Linear):            
            partNames = name.split('.')
            preName = '.'.join(partNames[:-1])
            layerName = partNames[-1]
            #print(f'preName = {preName}')
            if preName =='':
                parent = model
            else:
                parent = model.get_submodule(preName)    
            injectedLoraLayer = LoraLayer(layer,rank =8,alpha=16,device=device)
            setattr(parent,layerName,injectedLoraLayer)
    for name,params in model.named_parameters():
        print(f' {name} {params.size()}')


def freezeModelWeights(moduel:nn.Module)->None:
    for name,param in moduel.named_parameters():
        if 'loraA' in name or 'loraB' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    

def checkModuleStatus(model: nn.Module) -> None:
    print(f"{'Parameter Name':<60} | {'Status':<10} | {'Shape'}")
    print("-" * 85)
    
    for name, param in model.named_parameters():
        status = "✅ TRAIN" if param.requires_grad else "❄️ FROZEN"
        print(f"{name:<60} | {status:<10} | {list(param.shape)}")

    