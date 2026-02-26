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
        self.loraA= nn.Parameter(torch.randn(self.inFeatures,self.rank),requires_grad=True).to(device)
        self.loraB = nn.Parameter(torch.randn(self.rank,self.outFeatures),requires_grad=True).to(device) # 初始化权重矩阵为零，并使用Kaiming Normal分布进行随机采样。具体步骤如下：
        
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
        
        