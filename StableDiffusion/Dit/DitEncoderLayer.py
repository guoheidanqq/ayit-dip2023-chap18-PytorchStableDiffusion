import torch 
from torch import nn
from .DitConfig import DitConfig
from typing import Tuple,Optional
from .DitAttention import DitAttention
from .DitMlp import DitMlp

class DitEncoderLayer(nn.Module):
    def __init__(self,ditConfig:DitConfig):
        super(DitEncoderLayer, self).__init__()
        self.ditConfig = ditConfig
        self.hiddenSize = ditConfig.hiddenSize  #1152
        self.self_attn = DitAttention(ditConfig) # B 256 1152 
        self.layer_norm1 = nn.LayerNorm(ditConfig.hiddenSize,ditConfig.layerNormEps)
        self.layer_norm2 = nn.LayerNorm(ditConfig.hiddenSize,ditConfig.layerNormEps)
        self.mlp = DitMlp(ditConfig)# B 256 1152
    
    def forward(self,hiddenStates:torch.Tensor)->torch.Tensor:
        
        residual0 = hiddenStates # B 256 1152
        hiddenStates = self.layer_norm1(hiddenStates)
        hiddenStates,weights = self.self_attn(hiddenStates) # B 256 1152
        hiddenStates = residual0 + hiddenStates
        residual1 = hiddenStates
        hiddenStates = self.layer_norm2(hiddenStates)
        hiddenStates = self.mlp(hiddenStates) # B 256 1152
        hiddenStates = residual1 + hiddenStates
        
        return hiddenStates
        