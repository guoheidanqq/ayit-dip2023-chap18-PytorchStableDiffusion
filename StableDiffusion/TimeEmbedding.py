import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
class TimeEmbedding(nn.Module):
    def __init__(self,embeddingDimension = 320):
        super(TimeEmbedding,self).__init__()
        self.embeddingDimension = embeddingDimension
        self.linear_1  = nn.Linear(self.embeddingDimension, 4 * self.embeddingDimension)
        #diffusion_model.time_embed.0  1280 320
        self.linear_2  = nn.Linear(4 * self.embeddingDimension, 4 * self.embeddingDimension)
        #diffusion_model.time_embed.2  1280 1280
    def forward(self,inputTime):
        x = inputTime
        # 1,320
        x = self.linear_1(x)
        #1,1280
        x = F.silu(x)
        #add 20260218
        x = self.linear_2(x)
        #1,1280
        return x 