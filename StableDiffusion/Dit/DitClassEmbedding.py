import torch
import torch.nn as nn
from .DitConfig import DitConfig


class DitClassEmbedding(nn.Module):
    def __init__(self,ditConfig:DitConfig):
        super().__init__()
        self.classNum = ditConfig.classNum
        self.hiddenSize = ditConfig.hiddenSize
        self.useCfg = ditConfig.useCfg
        self.classEmbedding=nn.Embedding(self.classNum+1,self.hiddenSize)

    
    def forward(self,classIdBatch:torch.Tensor)-> torch.Tensor:
        x = classIdBatch    
        x = self.classEmbedding(x)
        return x
    
    
    
        