from .DitConfig import DitConfig
from .DitEncoderLayer import DitEncoderLayer
import torch
from torch import nn
from typing import Any,Tuple,Optional

class DitEncoder(nn.Module):

    def __init__(self, ditConfig:DitConfig):
        super(DitEncoder,self).__init__()
        #super().__init__()
        self.ditConfig = ditConfig
        self.numLayers = ditConfig.numHiddenLayers # 27
        self.layers = nn.ModuleList([DitEncoderLayer(self.ditConfig) for i in range(self.numLayers)])
        
    
    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        hiddenStates = inputs  
        # B 256 1152 
        for layer in self.layers:
            hiddenStates = layer(hiddenStates)
        # B 256 1152 -> B 256 1152    
        return hiddenStates