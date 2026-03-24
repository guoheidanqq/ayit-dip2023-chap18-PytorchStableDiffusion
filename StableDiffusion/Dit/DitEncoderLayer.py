import torch 
from torch import nn
from .DitConfig import DitConfig
from typing import Tuple,Optional
from .DitAttention import DitAttention
from .DitMlp import DitMlp

class DitEncoderLayer(nn.Module):
    def __init__(self,ditConfig:DitConfig):
        super().__init__()
        self.ditConfig = ditConfig
        self.hiddenSize = ditConfig.hiddenSize  #1152
        
        self.layer_norm1 = nn.LayerNorm(ditConfig.hiddenSize,ditConfig.layerNormEps)
        
        self.self_attn = DitAttention(ditConfig) # B 256 1152 
        
        self.layer_norm2 = nn.LayerNorm(ditConfig.hiddenSize,ditConfig.layerNormEps)
        
        self.mlp = DitMlp(ditConfig)# B 256 1152
        
        self.adaptiveLinearNorm = nn.Sequential(nn.SiLU(),
                                            nn.Linear(self.hiddenSize,self.hiddenSize * 6,bias = True))
        
    def modulate(self,x:torch.Tensor,scaleControl:torch.Tensor,shiftControl:torch.Tensor)->torch.Tensor:
        # x:B,N,D 
        # scaleControl: B,D 
        # shiftControl: B,D 
        scaleControl = scaleControl[:,None,:]
        shiftControl = shiftControl[:,None,:]
        x = x*(1.0 + scaleControl) + shiftControl
        return x
        
        
        
    
    def forward(self,hiddenStates:torch.Tensor,classEmbedBatch:torch.Tensor,timeEmbedBatch:torch.Tensor)->torch.Tensor:
        
        controlBatch = classEmbedBatch + timeEmbedBatch # B  1152
        controlBatch = self.adaptiveLinearNorm(controlBatch) # B  1152
        norm1Scale,norm1Shift,attentionGateScale,norm2Scale,norm2Shift,mlpGateScale = controlBatch.chunk(6,dim = 1)
        attentionGateScale =attentionGateScale[:,None,:]
        mlpGateScale = mlpGateScale[:,None,:]
        
        residual0 = hiddenStates # B 256 1152
        hiddenStates = self.layer_norm1(hiddenStates)
        hiddenStates = self.modulate(hiddenStates,norm1Scale,norm1Shift)
        hiddenStates,weights = self.self_attn(hiddenStates) # B 256 1152
        hiddenStates = hiddenStates*attentionGateScale
        
        hiddenStates = residual0 + hiddenStates
        residual1 = hiddenStates
        hiddenStates = self.layer_norm2(hiddenStates)
        hiddenStates = self.modulate(hiddenStates,norm2Scale,norm2Shift)
        hiddenStates = self.mlp(hiddenStates) # B 256 1152
        hiddenStates = hiddenStates*mlpGateScale
        hiddenStates = residual1 + hiddenStates
        
        return hiddenStates
        