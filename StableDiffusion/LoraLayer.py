import torch
import torch.nn as nn

class LoraLayer(nn.Module):
    def __init__(self,originLinear:nn.Module,rank:int = 8,alpha:int =16):
        super(LoraLayer,self).__init__()
        outFeatures,inFeatures = originLinear.weight.shape
        self.originLinear = originLinear
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.alpha = alpha 
        self.rank = rank
        self.scale= self.alpha / float(self.rank)
        self.loraA= nn.Parameter(torch.randn(self.inFeatures,self.rank),requires_grad=True)
        self.loraB = nn.Parameter(torch.randn(self.rank,self.outFeatures),requires_grad=True)
        
        nn.init.zeros_(self.loraB)
        nn.init.kaiming_normal_(self.loraA,a=5.0**0.5)
    
    
    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        x = inputs 
        mainPart = self.originLinear(x)
        x = x@self.loraA
        x = x @ self.loraB
        x = self.scale * x        
        x = x + mainPart    
        return x
        
        