import torch
import torch.nn as nn
from .DitConfig import DitConfig


class DitClassEmbedding(nn.Module):
    def __init__(self,ditConfig:DitConfig):
        super().__init__()
        self.classNum = ditConfig.classNum
        self.classDropoutRate = ditConfig.classDropoutRate
        self.hiddenSize = ditConfig.hiddenSize
        self.useCfg = ditConfig.useCfg
        self.classEmbedding=nn.Embedding(self.classNum+1,self.hiddenSize)

    
    def forward(self,classIdBatch:torch.Tensor)-> torch.Tensor:
        if self.training and self.useCfg:            
            device = classIdBatch.device
            B, = classIdBatch.shape
            mask = torch.rand(B).to(device)        
            mask = mask > self.classDropoutRate
            classIdBatch = torch.where(mask,classIdBatch,self.classNum)       
        
        x = classIdBatch    
        x = self.classEmbedding(x)
        return x
    
    
    
        