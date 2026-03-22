import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from .DitConfig import DitConfig
class DitTimeEmbedding(nn.Module):
    
    def __init__(self,ditConfig:DitConfig):
        super().__init__()
        self.timeEmbeddingDimension = ditConfig.timeEmbeddingDimension
        self.hiddenDimension = ditConfig.hiddenSize
        self.linear_1  = nn.Linear(self.timeEmbeddingDimension, self.hiddenDimension)
        #diffusion_model.time_embed.0  1280 320
        self.linear_2  = nn.Linear(self.hiddenDimension, self.hiddenDimension)
        #diffusion_model.time_embed.2  1280 1280
        
    def getTimeEmbeddingBatchTorch(self,timeStep:torch.Tensor,device='cuda')->torch.Tensor:  
        device = device
        t = timeStep
        t = t[:,None]
        #print(t.shape)
        HalfTimeDim = self.timeEmbeddingDimension//2
        frequency = torch.arange(0,HalfTimeDim,device=device) 
        frequency = -frequency/HalfTimeDim
        frequency = 10000**frequency    
        xk = t *frequency 
        #print(xk.shape)
        timeEmbeddingCos = torch.cos(xk)
        timeEmbeddingSin = torch.sin(xk)
        timeEmbedding = torch.cat([timeEmbeddingCos,timeEmbeddingSin],axis=1)
        return timeEmbedding
    
    def forward(self,inputTimeBatch:torch.Tensor)->torch.Tensor:        
        timeEmbed = self.getTimeEmbeddingBatchTorch(inputTimeBatch)
        x = timeEmbed
        # 1,320
        x = self.linear_1(x)
        #1,1280
        x = F.silu(x)
        #add 20260218
        x = self.linear_2(x)
        #1,1280
        return x 