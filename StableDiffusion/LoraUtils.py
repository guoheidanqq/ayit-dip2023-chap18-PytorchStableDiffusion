
import torch
import torch.nn as nn
from typing import Tuple
from .LoraLayer import LoraLayer


def injectLora(model:nn.Module,rank:int =8,alpha:int = 16,filterTuple:Tuple[str,...]=('in_proj','out_proj'),device='cuda')->None:
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
            injectedLoraLayer = LoraLayer(layer,rank =rank,alpha=alpha,device=device)
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

def writeLoraToFile(model:nn.Module)->None:
    import os    
    for name,param in model.named_parameters():
        print(name,param.shape)
    loraDict={}
    for name,param in model.named_parameters():
        if 'loraA' in name or 'loraB' in name:
            print(name)
            loraDict[name]=param
            
    for name,param in loraDict.items():
        print(name,param.shape)    
    torch.save(loraDict,'./lora.ckpt.tmp')
    os.replace('./lora.ckpt.tmp','./lora.ckpt') 


def loadLoraFromFile(model:nn.Module)->None:
    loraDict =torch.load('./lora.ckpt')
    model.load_state_dict(loraDict,strict=False)
    print(f"{'*'*30}lora loaded")    


