import torch
import torch.nn as nn

class LoraLayer(nn.Module):
    def __init__(self,originLinear:nn.Module,rank:int = 8,alpha:int =16,device:str = 'cuda'):
        super(LoraLayer,self).__init__()
        outFeatures,inFeatures = originLinear.weight.shape
        self.originLinear = originLinear.to(device)
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.alpha = alpha 
        self.rank = rank
        self.scale= self.alpha / float(self.rank)
        self.loraA= nn.Linear(self.inFeatures ,self.rank,bias=False).to(device)
        self.loraB = nn.Linear(self.rank,self.outFeatures,bias=False).to(device)  
        
        nn.init.kaiming_normal_(self.loraA.weight,a=5.0**0.5)
        nn.init.zeros_(self.loraB.weight)
    
    
    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        x = inputs 
        mainPart = self.originLinear(x)
        x = x@self.loraA(x)
        x = x @ self.loraB(x)
        x = self.scale * x        
        x = x + mainPart    
        return x
        
        