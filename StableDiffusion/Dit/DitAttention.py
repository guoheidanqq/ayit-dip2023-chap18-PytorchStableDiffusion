import torch 
from torch import nn
from .DitConfig import DitConfig
from typing import Tuple,Optional

class DitAttention(nn.Module):
    def __init__(self,ditConfig:DitConfig):
        super().__init__()
        self.ditConfig = ditConfig
        self.numHeads = ditConfig.numAttenionHeads #16
        self.hiddenSize = ditConfig.hiddenSize #1152
        self.headDims = self.hiddenSize // self.numHeads #1152//16=72
        self.scaleFactor = self.headDims ** -0.5 #  multiplier factor 
        self.attentionDropoutRate = ditConfig.attentionDropoutRate #0.0
        self.k_proj = nn.Linear(self.hiddenSize,self.hiddenSize,bias = True) # 1152->1152
        self.v_proj = nn.Linear(self.hiddenSize,self.hiddenSize,bias = True) # 1152->1152
        self.q_proj = nn.Linear(self.hiddenSize,self.hiddenSize,bias = True) # 1152->1152 
        self.out_proj = nn.Linear(self.hiddenSize,self.hiddenSize,bias = True) # 1152->11152
        
    
    def forward(self,hiddenStates:torch.Tensor)->Tuple[torch.Tensor,Optional[torch.Tensor]]:
        #
        BatchSize,SeqLen,HiddenSize = hiddenStates.shape # (B,256,1152)
        queryStates = self.q_proj(hiddenStates) # B 256 1152
        queryStates = queryStates.reshape(BatchSize,SeqLen,self.numHeads,self.headDims) # B 256 16 72
        queryStates = queryStates.permute(0,2,1,3) # B 16 256 72
        keyStates = self.k_proj(hiddenStates)
        keyStates = keyStates.reshape(BatchSize,SeqLen,self.numHeads,self.headDims) # B 256 16 72
        keyStates = keyStates.permute(0,2,1,3) # B 16 256 72
        valueStates  = self.v_proj(hiddenStates)
        valueStates = valueStates.reshape(BatchSize,SeqLen,self.numHeads,self.headDims) # B 256 16 72
        valueStates = valueStates.permute(0,2,1,3) # B 16 256 72
        attentionWeights = queryStates @ keyStates.permute(0,1,3,2) 
        #B 16 256 72 @ B 16 72 256 -> B 16 256 256
        attentionWeights = attentionWeights * self.scaleFactor  # B 16 256 256
        attentionWeights = nn.functional.softmax(attentionWeights,dim=-1,dtype=torch.float).to(queryStates.dtype)
        attentionWeights = nn.functional.dropout(attentionWeights,p=self.attentionDropoutRate,training=self.training)
        attentionOutputs = attentionWeights @ valueStates
        # B 16 256 256 @ B 16 256 72 -> B 16 256 72
        attentionOutputs = attentionOutputs.permute(0,2,1,3)
        # B 16 256 72 -> B 256 16 72
        attentionOutputs = attentionOutputs.reshape(BatchSize,SeqLen,self.hiddenSize)
        # B 256 16 72-> B 256 1152
        attentionOutputs = self.out_proj(attentionOutputs)
        # B SeqLen HiddenSize   B 256 1152     
        
        return attentionOutputs,attentionWeights
        #attentionOutputs B 256 1152, attentionWeights B 16 256 256