
import torch
import torch.nn as nn
from typing import Tuple
from .LoraLayer import LoraLayer


def injectLora(model:nn.Module,filterTuple:Tuple[str,...]=('in_proj','out_proj'))->None:
    for name,layer in model.named_modules():        
        filterName =filterTuple
        if name.endswith(filterName) and isinstance(layer,nn.Linear):            
            partNames = name.split('.')
            preName = '.'.join(partNames[:-1])
            layerName = partNames[-1]
            print(f'preName = {preName}')
            if preName =='':
                parent = model
            else:
                parent = model.get_submodule(preName)    
            injectedLoraLayer = LoraLayer(layer,rank =8,alpha=16)
            setattr(parent,layerName,injectedLoraLayer)
    for name,params in model.named_parameters():
        print(f' {name} {params.size()}')