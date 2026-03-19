
import torch
import torch.nn as nn
from .DitConfig import DitConfig

class DitMlp(nn.Module):
    def __init__(self, config: DitConfig):
        super(DitMlp,self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.hiddenSize,self.config.intermediateSize,bias = True) # 1152->4304
        self.fc2 = nn.Linear(self.config.intermediateSize,self.config.hiddenSize,bias = True)  #4304->1152
    
    def forward(self,inputs:torch.Tensor):
        x = inputs  # B  256 1152
        x = self.fc1(x) # B 256 4304
        x = nn.functional.gelu(x,approximate='tanh')
        x = self.fc2(x) # B 256 1152
        return x 
    
        